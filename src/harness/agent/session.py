"""Append-only JSONL session store + replay.

One file per session under /state/sessions/<id>.jsonl. Events are the
audit log (git-tracked in /state); replay() rebuilds the working memory
the model sees. Contract:

  memory = system message
         + all summary texts, in order
         + all msg/tool events after the LAST summary

Everything before the last summary is intentionally dropped from memory
(it lives on in the JSONL, which is what you audit). Only msg and tool
events become wire messages; everything else (approval, turn_end, error,
interrupted, ...) is metadata and is skipped by replay.

This module never raises on I/O or corrupt lines: a bad line is skipped
so a hiccup can't kill a session.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config as cfgmod

SESSIONS_DIR = cfgmod.STATE_DIR / "sessions"

# Marker for a compacted span, per requirements. The loop and replay both
# build the summary user message through summary_message() so they can't
# drift apart.
SUMMARY_MARKER = "[Summary of prior conversation]"


def new_session_id(agent: str, now: float | None = None) -> str:
    """YYYYMMDD-HHMM-<agent>, local time. e.g. 20260709-1430-coder."""
    return time.strftime("%Y%m%d-%H%M", time.localtime(now)) + f"-{agent}"


def summary_message(text: str) -> dict:
    """The single user message a compacted span is replaced by."""
    return {"role": "user", "content": f"{SUMMARY_MARKER}\n{text}"}


def _event_to_message(ev: dict) -> dict | None:
    """Map a msg/tool event to an OpenAI wire message; None otherwise."""
    t = ev.get("type")
    if t == "msg":
        msg: dict = {"role": ev.get("role"), "content": ev.get("content") or ""}
        if ev.get("tool_calls"):
            msg["tool_calls"] = ev["tool_calls"]
        return msg
    if t == "tool":
        # name/args/ok are audit-only; the wire format needs id + result.
        return {"role": "tool", "tool_call_id": ev.get("id"),
                "content": ev.get("result") or ""}
    return None


def _read_all(path: Path) -> list[dict]:
    """Read every parseable event. Missing file -> []; bad line -> skipped."""
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


class Session:
    """A single session: an append-only event log plus derived memory."""

    def __init__(self, path: Path, agent: str, model: str,
                 events: list[dict] | None = None):
        self.path = path
        self._agent0 = agent
        self.model = model
        self._events: list[dict] = events if events is not None else _read_all(path)

    # ---------- construction ----------

    @classmethod
    def create(cls, agent: str, model: str,
               sessions_dir: Path = SESSIONS_DIR) -> "Session":
        sessions_dir.mkdir(parents=True, exist_ok=True)
        base = new_session_id(agent)
        path = sessions_dir / f"{base}.jsonl"
        n = 1
        while path.exists():          # same-minute collision: suffix -1, -2...
            path = sessions_dir / f"{base}-{n}.jsonl"
            n += 1
        s = cls(path, agent, model, events=[])
        s.append({"type": "session", "id": s.id, "agent": agent, "model": model})
        return s

    @classmethod
    def open(cls, path: Path) -> "Session":
        """Resume an existing session file, recovering agent/model from it."""
        events = _read_all(path)
        agent, model = "coder", ""
        for ev in events:
            if ev.get("type") == "session":
                agent = ev.get("agent", agent)
                model = ev.get("model", model)
                break
        return cls(path, agent, model, events=events)

    # ---------- identity / derived state ----------

    @property
    def id(self) -> str:
        return self.path.stem

    @property
    def agent(self) -> str:
        """Current agent: latest `agent` event, else the session's agent."""
        name = self._agent0
        for ev in self._events:
            if ev.get("type") == "agent":
                name = ev.get("name", name)
        return name

    @property
    def turn_number(self) -> int:
        """Completed turns (last turn_end.n, 0 for a fresh session)."""
        n = 0
        for ev in self._events:
            if ev.get("type") == "turn_end":
                n = ev.get("n", n)
        return n

    @property
    def last_prompt_tokens(self) -> int:
        """Last known prompt_tokens (last turn_end), 0 if none yet."""
        tokens = 0
        for ev in self._events:
            if ev.get("type") == "turn_end":
                tokens = ev.get("prompt_tokens", tokens)
        return tokens

    @property
    def last_commit(self) -> str | None:
        commit = None
        for ev in self._events:
            if ev.get("type") == "turn_end":
                commit = ev.get("commit")
        return commit

    # ---------- persistence ----------

    def append(self, event: dict) -> None:
        """Append one event to memory and to the JSONL file. Never raises."""
        event = dict(event)
        self._events.append(event)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ---------- typed event helpers (keep the loop clean) ----------

    def record_user(self, content: str) -> None:
        self.append({"type": "msg", "role": "user", "content": content})

    def record_assistant(self, content: str,
                         tool_calls: list[dict] | None = None) -> None:
        ev: dict = {"type": "msg", "role": "assistant", "content": content}
        if tool_calls:
            ev["tool_calls"] = tool_calls
        self.append(ev)

    def record_tool(self, id: str, name: str, args: dict,
                    result: str, ok: bool) -> None:
        self.append({"type": "tool", "id": id, "name": name, "args": args,
                     "result": result, "ok": ok})

    def record_approval(self, tool: str, decision: str) -> None:
        self.append({"type": "approval", "tool": tool, "decision": decision})

    def record_summary(self, text: str) -> None:
        self.append({"type": "summary", "text": text})

    def record_agent(self, name: str) -> None:
        self.append({"type": "agent", "name": name})

    def record_interrupted(self) -> None:
        self.append({"type": "interrupted"})

    def record_error(self, text: str) -> None:
        self.append({"type": "error", "text": text})

    def record_turn_aborted(self, reason: str) -> None:
        self.append({"type": "turn_aborted", "reason": reason})

    def record_turn_end(self, n: int, commit: str | None,
                        prompt_tokens: int, completion_tokens: int = 0) -> None:
        self.append({"type": "turn_end", "n": n, "commit": commit,
                     "prompt_tokens": prompt_tokens,
                     "completion_tokens": completion_tokens})

    # ---------- memory ----------

    def replay(self, system_prompt: str) -> list[dict]:
        """Rebuild working memory per the replay rule (see module docstring)."""
        memory: list[dict] = [{"role": "system", "content": system_prompt}]

        last_summary_idx = -1
        for i, ev in enumerate(self._events):
            if ev.get("type") == "summary":
                last_summary_idx = i

        for ev in self._events:                       # all summaries, in order
            if ev.get("type") == "summary":
                memory.append(summary_message(ev.get("text", "")))

        for ev in self._events[last_summary_idx + 1:]:  # post-last-summary only
            msg = _event_to_message(ev)
            if msg is not None:
                memory.append(msg)

        return memory

