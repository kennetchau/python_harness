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

The cut always lands on a real user-message turn start, so an
assistant + tool_calls + tool group is never split (chat templates break
if those are separated).
"""

from __future__ import annotations

import json

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
    return memory[:base] + [summary_message(text)] + memory[cut:]


def ensure_budget(memory: list[dict], client, cfg: cfgmod.Config,
                  session) -> list[dict] | None:
    """Auto path: compact until the estimate is under the trigger. Returns
    the (possibly new) memory, or None when even the smallest recent window
    exceeds the budget (an error event is recorded)."""
    target = int(cfg.context.max_tokens * cfg.context.summarize_threshold)
    keep = cfg.context.keep_recent_turns
    while estimate_prompt_tokens(memory) > target:
        new_mem = compact_once(memory, client, cfg, session, keep=keep)
        if new_mem is None:
            if keep > 1:            # recent window too big: keep fewer, retry
                keep = max(1, keep // 2)
                continue
            session.record_error("context exceeded, start a new session")
            return None
        memory = new_mem
    return memory

