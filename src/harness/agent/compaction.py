"""Context compaction: summarize old turns to free context space.

Design note (why summaries are never re-summarized):
  Each compaction drops a contiguous block of REAL turns that sits right
  after the leading run of summary messages, and inserts ONE new summary
  message there. Existing summaries are preserved, so every summary in
  memory covers a disjoint chunk of turns and there is at most a growing
  list of them at the front. This keeps the live memory exactly consistent
  with Session.replay(), whose rule is "system + ALL summaries in order +
  all msg/tool events after the LAST summary". If we folded old summaries
  into a new one, replay would double-count them.

The cut always lands either on a real user-message turn start or on a
round's assistant message (a round is that message plus its tool results),
so an assistant + tool_calls + tool group is never split (chat templates
break if those are separated).

When a single long tool-call chain outgrows the budget and there are no
older user turns to drop, the cut falls inside the current turn: everything
before the last `keep_recent_turns` complete rounds — older turns included,
and the turn's own user message — is folded into one summary. The kept
rounds are pure assistant/tool groups, which is the same shape memory already
has mid-turn, so replay stays consistent (system + all summaries + all
msg/tool events after the last summary).
"""

from __future__ import annotations

import json
import threading

from .. import config as cfgmod
from ..backend.client import BackendError, estimate_prompt_tokens
from .session import SUMMARY_MARKER, summary_message

SUMMARIZER_SYSTEM = (
    "You compact an agent's conversation history to free context space. "
    "Write a concise, factual summary that lets the agent continue seamlessly. "
    "Preserve: decisions made, files created/edited (with exact paths), commands "
    "run and their outcomes, errors and how they were resolved, open tasks, and "
    "facts the user stated. Keep paths, names, and values exact. Do not invent "
    "new information. Output plain text only."
)


def _is_summary(m: dict) -> bool:
    return m.get("role") == "user" and \
        str(m.get("content") or "").startswith(SUMMARY_MARKER)


def _front_summary_end(memory: list[dict]) -> int:
    """Index just past the leading run of summary messages (after system)."""
    i = 1  # memory[0] is the system message
    while i < len(memory) and _is_summary(memory[i]):
        i += 1
    return i


def _real_turn_starts(memory: list[dict]) -> list[int]:
    """Indices of real user messages (turn starts), excluding summaries."""
    return [i for i, m in enumerate(memory) if m.get("role") == "user"
            and not _is_summary(m)]


def _find_cut(memory: list[dict], keep: int) -> int | None:
    """Index of the oldest real turn we keep, so the tail has exactly `keep`
    complete turns. None when there is nothing old to drop."""
    base = _front_summary_end(memory)
    turns = _real_turn_starts(memory)
    if len(turns) <= keep:
        return None
    cut = turns[-keep]
    if cut <= base:      # first kept turn is the first real turn: nothing before it
        return None
    return cut


def _find_round_cut(memory: list[dict], base: int, keep: int) -> int | None:
    """Cut index so the tail keeps the last `keep` rounds of the current
    turn, walking backwards from the end. A round is an assistant message
    with tool_calls plus its tool results; an in-flight round (results not
    all back yet) counts as one and stays in the tail as-is. A trailing
    plain assistant message (the turn's answer) is always kept. The cut
    lands on a round's assistant message, so no assistant/tool group is
    ever split. None when there is no round strictly before the kept
    window — dropping an empty span would grow memory instead of shrinking
    it (summarizer call in, no content out)."""
    i = len(memory) - 1
    if i >= base and memory[i].get("role") == "assistant" \
            and not (memory[i].get("tool_calls") or []):
        i -= 1                             # trailing answer: always kept
    starts: list[int] = []                 # round assistant indices, newest first
    while i >= base:
        m = memory[i]
        if m.get("role") == "tool":
            # the walk lands here between rounds: back up to this round's
            # assistant message (always the one whose results these are)
            while i >= base and memory[i].get("role") == "tool":
                i -= 1
        m = memory[i]
        if m.get("role") != "assistant" or not (m.get("tool_calls") or []):
            break                          # user msg, summary, or plain assistant
        starts.append(i)
        i -= 1
    if len(starts) <= keep:
        return None                        # kept window covers every round
    cut = starts[keep - 1]                 # earliest kept round
    return cut if cut > base else None     # cut == base: nothing before it to drop


def _short(s, n: int = 60) -> str:
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _arg_detail(name: str, args_str: str) -> str:
    """Pick the most identifying argument of a tool call for the summary."""
    try:
        a = json.loads(args_str) if args_str else {}
    except (json.JSONDecodeError, TypeError):
        return _short(args_str)
    if not isinstance(a, dict):
        return _short(a)
    for key in ("path", "command", "url", "pattern", "query", "text"):
        v = a.get(key)
        if isinstance(v, str) and v:
            return _short(v)
    for v in a.values():
        if isinstance(v, str) and v:
            return _short(v)
    return _short(json.dumps(a))


def _flatten(span: list[dict]) -> str:
    """Flatten dropped wire messages for the summarizer. Tool calls fold into
    one line each:  called write_file(path) -> ok, 2,100 chars."""
    results = {m.get("tool_call_id"): (m.get("content") or "")
               for m in span if m.get("role") == "tool"}
    out: list[str] = []
    for m in span:
        role = m.get("role")
        if role == "user":
            out.append(f"user: {m.get('content', '')}")
        elif role == "assistant":
            content = (m.get("content") or "").strip()
            if content:
                out.append(f"assistant: {content}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name", "?")
                res = results.get(tc.get("id"), "")
                status = "err" if res.startswith("error") else "ok"
                out.append(f"  called {name}({_arg_detail(name, fn.get('arguments') or '')})"
                           f" -> {status}, {len(res):,} chars")
        # tool results are folded into their assistant line above
    return "\n".join(out)


def compact_once(memory: list[dict], client, cfg: cfgmod.Config, session,
                 keep: int | None = None) -> list[dict] | None:
    """One compaction pass: drop the oldest real turns (keeping the most
    recent `keep`), summarize them, insert the summary. Returns the new
    memory, or None when there is nothing old to drop. Raises BackendError
    if the summarizer call fails or returns nothing."""
    if keep is None:
        keep = cfg.context.keep_recent_turns
    base = _front_summary_end(memory)
    cut = _find_cut(memory, keep)
    if cut is None:
        # No older user turns to drop (e.g. a single long tool-call chain):
        # fall back to cutting complete rounds, keeping the last `keep`.
        cut = _find_round_cut(memory, base, keep)
    if cut is None:
        return None

    flattened = _flatten(memory[base:cut])
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        {"role": "user",
         "content": f"Summarize this dropped conversation span:\n\n{flattened}"},
    ]
    text = client.complete(messages, max_tokens=cfg.context.summary_max_tokens)
    if not text or not text.strip():
        raise BackendError("summarizer returned an empty summary")
    text = text.strip()

    session.record_summary(text)
    _record_kept_tail(session, memory[cut:])
    return memory[:base] + [summary_message(text)] + memory[cut:]


def _record_kept_tail(session, tail: list[dict]) -> None:
    """Re-record the kept tail in the session log, AFTER the summary event
    (tagged dup: True so it's obvious in the audit log).

    replay() keeps "all summaries + all msg/tool events after the LAST
    summary", so anything a cut leaves in memory must exist in the log
    after the summary event — a resumed session otherwise loses it (e.g. the
    current turn's rounds in a mid-turn cut, or kept prior turns that predate
    the summary). The original events stay in the log unmodified as the
    unsummarized audit trail. The summary and its tail are written
    synchronously with no preemption point in between — that order is what
    replay depends on."""
    for m in tail:
        role = m.get("role")
        if role == "user":
            session.append({"type": "msg", "role": "user",
                            "content": m.get("content") or "", "dup": True})
        elif role == "assistant":
            ev = {"type": "msg", "role": "assistant",
                  "content": m.get("content") or "", "dup": True}
            if m.get("tool_calls"):
                ev["tool_calls"] = m["tool_calls"]
            session.append(ev)
        elif role == "tool":
            session.append({"type": "tool", "id": m.get("tool_call_id"),
                            "result": m.get("content") or "",
                            "ok": True, "dup": True})


def ensure_budget(memory: list[dict], client, cfg: cfgmod.Config,
                  session, cancel: "threading.Event | None" = None) -> list[dict] | None:
    """Auto path: compact until the estimate is under the trigger. Returns
    the (possibly new) memory, or None when even the smallest recent window
    exceeds the budget (an error event is recorded). An interrupt request
    (cancel) stops the loop at the next iteration and returns memory as-is.

    Safety: every pass must strictly shrink the memory (a summary replaces
    >=2 messages with one). If that invariant ever fails we bail instead of
    looping forever."""
    target = int(cfg.context.max_tokens * cfg.context.summarize_threshold)
    keep = cfg.context.keep_recent_turns
    while estimate_prompt_tokens(memory) > target:
        if cancel is not None and cancel.is_set():
            return memory
        new_mem = compact_once(memory, client, cfg, session, keep=keep)
        if new_mem is None:
            if keep > 1:            # recent window too big: keep fewer, retry
                keep = max(1, keep // 2)
                continue
            session.record_error("context exceeded, start a new session")
            return None
        if len(new_mem) >= len(memory):
            session.record_error("compaction made no progress, aborting")
            return None
        memory = new_mem
    return memory

