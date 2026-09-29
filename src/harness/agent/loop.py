"""The agent turn state machine.

One run_turn(user_message) call drives a full turn:

    user message -> budget check (compaction) -> streaming request ->
    [tool calls -> approval -> execute -> tool results]* -> final message ->
    wrap-up (final diff + the two independent auto-commits)

Memory is a plain list[dict] in OpenAI wire format, kept in lockstep with
the session's JSONL log: every append to memory is paired with a session
record, so a resumed session replays to exactly the same memory.

Tool errors are data for the model; only backend/loop failures abort the
turn. An aborted or interrupted turn still commits its side effects.
"""

from __future__ import annotations

import difflib
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console

from .. import config as cfgmod
from ..backend.client import (BackendClient, BackendError, Done, ReasoningDelta,
                              TextDelta, ToolCallDelta, Usage,
                              estimate_prompt_tokens)
from ..state import gitstore
from ..tools import registry
from .compaction import ensure_budget
from .session import Session

DEFAULT_AGENT = "coder"

SYSTEM_PROMPTS: dict[str, str] = {
    DEFAULT_AGENT: (
        "You are coder, a software engineering agent running in a Linux container. "
        "All file paths are relative to your workspace (the current directory).\n\n"
        "Working style:\n"
        "- Inspect before you modify: read/grep before write/edit.\n"
        "- Make minimal, correct changes; verify your work when it is cheap to do so "
        "(run the program or a relevant test).\n"
        "- If a tool call is denied by the user, do not retry the identical call. "
        "Adjust the approach or explain what you need instead.\n"
        "- Follow existing project conventions; do not add files or dependencies "
        "that were not asked for.\n"
        "- Keep the final reply short: what you did, plus any follow-ups."
    ),
}


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one run_turn. The loop never raises for turn-level failures."""
    ok: bool
    reason: str = ""
    commit: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------- small helpers ----------

def _git_out(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return proc.stdout


def _short_text(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    nl = cut.rfind("\n")
    if nl > limit // 2:
        cut = cut[:nl]
    return cut + f"\n…[truncated at {limit} chars]"


def _first_line(text: str, limit: int = 60) -> str:
    """First non-empty line of the final reply, for the commit message."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "(no summary)"
    line = lines[0]
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _edit_preview(args: dict, workspace: Path) -> str:
    """Unified diff for edit_file when the file exists; old/new strings otherwise."""
    path = str(args.get("path") or args.get("file") or "")
    old = str(args.get("old_string") or args.get("old") or "")
    new = str(args.get("new_string") or args.get("new") or "")
    candidate = (workspace / path) if path else None
    if candidate is not None and candidate.is_file():
        try:
            original = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            original = None
        if original is not None and old:
            count = -1 if args.get("replace_all") else 1
            replaced = original.replace(old, new, count)
            diff = "".join(difflib.unified_diff(
                original.splitlines(True), replaced.splitlines(True),
                fromfile=f"a/{path}", tofile=f"b/{path}"))
            if diff:
                return _short_text(f"{path}\n{diff}", 4000)
    return _short_text(f"{path}\n-find:\n{old}\n+replace:\n{new}", 2000)


def make_preview(name: str, args: dict, workspace: Path) -> str:
    """What the user approves: diff for edits, command text for exec, URL for web."""
    if name == "write_file":
        return _short_text(f"{args.get('path', '')}\n{args.get('content', '')}", 2000)
    if name == "edit_file":
        return _edit_preview(args, workspace)
    if name == "run_command":
        return f"$ {args.get('command', '')}"
    if name == "web_fetch":
        return f"GET {args.get('url', '')}"
    if name == "web_search":
        return f"search: {args.get('query', '')}"
    if name == "delete_file":
        return f"delete {args.get('path', '')}"
    return _short_text(json.dumps(args, indent=2), 1500)


def _default_ask(name: str, args: dict, preview: str) -> str:
    """Console approval prompt. EOF / Ctrl-C deny, so unattended runs never hang."""
    console = Console()
    console.rule(f"[yellow]{name}[/yellow] — approval requested")
    console.print(preview, markup=False, highlight=False)
    while True:
        try:
            answer = console.input("[y]es / [n]o / [a]lways-allow this tool: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "n"
        if answer in ("y", "yes"):
            return "y"
        if answer in ("a", "always"):
            return "a"
        if answer in ("n", "no", ""):
            return "n"


def _make_default_emit(console: Console) -> Callable[[str, dict], None]:
    """Headless renderer. The TUI later passes its own emit() instead."""
    def emit(kind: str, data: dict) -> None:
        if kind == "text":
            console.print(data["text"], end="", markup=False, highlight=False)
        elif kind == "reasoning":
            console.print(data["text"], end="", markup=False, highlight=False,
                          style="dim italic")
        elif kind == "tool":
            args = data.get("args") or {}
            shown = args if isinstance(args, dict) else {"raw": args}
            console.print(f"\n[bold cyan]⚙ {data['name']}[/bold cyan] "
                          f"{_short_text(json.dumps(shown), 200)}")
        elif kind == "tool_result":
            mark = "[green]✓[/green]" if data["ok"] else "[red]✗[/red]"
            console.print(f"  {mark} {_short_text(str(data['result']), 400)}")
        elif kind == "error":
            console.print(f"\n[bold red]error: {data['text']}[/bold red]")
        else:  # info
            console.print(f"\n[dim]{data['text']}[/dim]")
    return emit


# ---------- the loop ----------

class AgentLoop:
    def __init__(self, cfg: cfgmod.Config, client: BackendClient, session: Session,
                 ask_approval: Callable[[str, dict, str], str] | None = None,
                 emit: Callable[[str, dict], None] | None = None,
                 console: Console | None = None):
        self.cfg = cfg
        self.client = client
        self.session = session
        self.console = console or Console()
        self._emit = emit or _make_default_emit(self.console)
        self._ask = ask_approval or _default_ask
        self.ctx = registry.ToolContext(cfg, Path(cfg.workspace.path))
        self._schemas = registry.schemas(cfg)
        self._enabled = set(registry.enabled_names(cfg))
        self._always_allow: set[str] = set()      # session-scoped "a" decisions
        self._cancel = threading.Event()
        system = SYSTEM_PROMPTS.get(session.agent, SYSTEM_PROMPTS[DEFAULT_AGENT])
        self.memory: list[dict] = session.replay(system)

    # -- one turn -----------------------------------------------------------

    def run_turn(self, user_message: str) -> TurnResult:
        cfg = self.cfg
        turn_n = self.session.turn_number + 1
        self._cancel.clear()

        # 1. user message -> memory + JSONL
        self.session.record_user(user_message)
        self.memory.append({"role": "user", "content": user_message})

        # 2. budget check -> compaction pass
        fail = self._check_budget(turn_n)
        if fail is not None:
            return fail

        max_rounds = cfg.tools.limits.max_tool_rounds
        rounds = 0
        malformed_streak = 0
        prompt_tokens = self.session.last_prompt_tokens
        completion_tokens = 0

        while True:
            # 3. one streaming request. Esc (KeyboardInterrupt) stops
            #    iteration; the client's finally closes the stream and the
            #    partial output is simply never appended.
            try:
                content, tool_calls, usage = self._stream()
            except KeyboardInterrupt:
                self.session.record_interrupted()
                self._emit("info", {"text": "interrupted — partial output discarded"})
                self._wrapup(turn_n, prompt_tokens, completion_tokens,
                             "interrupted", completed=False)
                return TurnResult(False, "interrupted", None, prompt_tokens,
                                  completion_tokens)
            except BackendError as e:
                self.session.record_error(str(e))
                self._emit("error", {"text": str(e)})
                self._wrapup(turn_n, prompt_tokens, completion_tokens,
                             "aborted: backend error", completed=False)
                return TurnResult(False, f"backend error: {e}", None, prompt_tokens,
                                  completion_tokens)
            if self._cancel.is_set():
                self.session.record_interrupted()
                self._emit("info", {"text": "interrupted — partial output discarded"})
                self._wrapup(turn_n, prompt_tokens, completion_tokens,
                             "interrupted", completed=False)
                return TurnResult(False, "interrupted", None, prompt_tokens,
                                  completion_tokens)

            if usage and usage.prompt_tokens:
                prompt_tokens = usage.prompt_tokens
            else:
                prompt_tokens = estimate_prompt_tokens(self.memory)
            if usage:
                completion_tokens += usage.completion_tokens

            # 5. no tool calls -> final assistant message -> wrap-up
            if not tool_calls:
                self.memory.append({"role": "assistant", "content": content})
                self.session.record_assistant(content)
                commit = self._wrapup(turn_n, prompt_tokens, completion_tokens,
                                      _first_line(content), completed=True)
                return TurnResult(True, "", commit, prompt_tokens, completion_tokens)

            # guard: max tool rounds per turn. The dangling assistant+tool_calls
            # message is deliberately NOT appended, so memory stays valid.
            if rounds >= max_rounds:
                self.session.record_turn_aborted(f"max_tool_rounds ({max_rounds}) exceeded")
                self._emit("error", {"text": f"max tool rounds ({max_rounds}) exceeded — stopping"})
                self._wrapup(turn_n, prompt_tokens, completion_tokens,
                             "aborted: max tool rounds", completed=False)
                return TurnResult(False, f"max tool rounds ({max_rounds}) exceeded",
                                  None, prompt_tokens, completion_tokens)
            rounds += 1

            # 4. assistant message with tool_calls -> memory + JSONL, then execute
            self.memory.append({"role": "assistant", "content": content,
                                "tool_calls": tool_calls})
            self.session.record_assistant(content, tool_calls)

            try:
                for tc in tool_calls:
                    malformed_streak, stop_reason = self._execute_call(tc, malformed_streak)
                    if stop_reason:
                        self._wrapup(turn_n, prompt_tokens, completion_tokens,
                                     "aborted: malformed tool calls", completed=False)
                        return TurnResult(False, stop_reason, None, prompt_tokens,
                                          completion_tokens)
            except KeyboardInterrupt:
                self._backfill_interrupted(tool_calls)
                self.session.record_interrupted()
                self._emit("info", {"text": "interrupted during tool execution"})
                self._wrapup(turn_n, prompt_tokens, completion_tokens,
                             "interrupted", completed=False)
                return TurnResult(False, "interrupted", None, prompt_tokens,
                                  completion_tokens)

            if self._cancel.is_set():
                self._backfill_interrupted(tool_calls)
                self.session.record_interrupted()
                self._emit("info", {"text": "interrupted during tool execution"})
                self._wrapup(turn_n, prompt_tokens, completion_tokens,
                             "interrupted", completed=False)
                return TurnResult(False, "interrupted", None, prompt_tokens,
                                  completion_tokens)

            # 6. mid-turn budget check: a long tool chain grows memory every
            #    round, so compact inside the loop too, not just at turn start.
            #    Memory is always valid here: the assistant+tool_calls+tool
            #    group from this round is complete. prompt_tokens is the live
            #    in-turn count (backend usage when available), so growth from
            #    this turn's tool results actually trips the threshold.
            fail = self._check_budget(turn_n, known_tokens=prompt_tokens)
            if fail is not None:
                return fail

            # otherwise: loop back to 3 — request-driven, approval pauses
            # cost nothing on the wire

    # -- interruption ---------------------------------------------------------

    def cancel(self) -> None:
        """Request interruption of the running turn (thread-safe). The loop
        stops at the next safe point (between stream chunks, between tool
        calls, between rounds), discards the partial output, records an
        interrupted event, and wraps up like any other turn. A tool that is
        already executing runs on to completion or its own timeout."""
        self._cancel.set()

    def _check_budget(self, turn_n: int, known_tokens: int | None = None) -> TurnResult | None:
        """Auto compaction pass. Returns None when memory fits the budget (or
        summarization is off) — otherwise a TurnResult that aborts the turn.
        Called at turn start AND after each tool round, so a long tool chain
        compacts consistently instead of only at the next turn's start.

        The gate must look at the CURRENT memory, not a count from the last
        request: a tool round appends the assistant+tool message and every
        tool result, so the request's reported prompt_tokens is always one
        round behind the memory that actually gets sent next. We therefore
        max() the in-turn count, the last turn's count, and a fresh local
        estimate of the current memory. Over-gating is safe — ensure_budget
        re-checks the live estimate and is a no-op when memory already fits."""
        if self.cfg.context.summarize != "auto":
            return None
        known = max(known_tokens or 0, self.session.last_prompt_tokens,
                    estimate_prompt_tokens(self.memory))
        if known <= self.cfg.context.max_tokens * self.cfg.context.summarize_threshold:
            return None
        try:
            new_mem = ensure_budget(self.memory, self.client, self.cfg, self.session,
                                    cancel=self._cancel)
        except BackendError as e:
            self.session.record_error(f"compaction failed: {e}")
            self._emit("error", {"text": f"compaction failed: {e}"})
            self._wrapup(turn_n, known, 0, "aborted: compaction failed",
                         completed=False)
            return TurnResult(False, f"compaction failed: {e}", None, known, 0)
        if new_mem is None:
            self._wrapup(turn_n, known, 0, "aborted: context exceeded",
                         completed=False)
            return TurnResult(False, "context exceeded, start a new session",
                              None, known, 0)
        if new_mem is not self.memory:
            self.memory = new_mem
            self._emit("info", {"text": "context compacted"})
        return None

    # -- internals ----------------------------------------------------------

    def _stream(self) -> tuple[str, list[dict], Usage | None]:
        """Consume one streamed response. Returns (content, tool_calls, usage).
        Raises BackendError (aborts the turn) or propagates KeyboardInterrupt
        (the partial accumulation here is discarded)."""
        content_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage: Usage | None = None
        completion_chars = 0
        for ev in self.client.stream_chat(self.memory, tools=self._schemas, cancel=self._cancel):
            if isinstance(ev, ReasoningDelta):
                completion_chars += len(ev.text)
                self._emit("reasoning", {"text": ev.text,
                                         "tokens": completion_chars // 4})
            elif isinstance(ev, TextDelta):
                content_parts.append(ev.text)
                completion_chars += len(ev.text)
                self._emit("text", {"text": ev.text,
                                    "tokens": completion_chars // 4})
            elif isinstance(ev, ToolCallDelta):
                c = calls.setdefault(ev.index, {"id": None, "name": None, "args": []})
                if ev.id:
                    c["id"] = ev.id
                if ev.name:
                    c["name"] = ev.name
                if ev.arguments:
                    c["args"].append(ev.arguments)
            elif isinstance(ev, Done):
                usage = ev.usage
        tool_calls: list[dict] = []
        for idx in sorted(calls):
            c = calls[idx]
            tool_calls.append({
                "id": c["id"] or f"call_{len(tool_calls)}",
                "type": "function",
                "function": {"name": c["name"] or "unknown",
                             "arguments": "".join(c["args"])},
            })
        return "".join(content_parts), tool_calls, usage

    def _execute_call(self, tc: dict, malformed_streak: int) -> tuple[int, str | None]:
        """One tool call: parse -> disabled? -> approval? -> execute -> record.
        Returns (new_malformed_streak, abort_reason_or_None)."""
        name = tc["function"]["name"]
        call_id = tc["id"]
        raw_args = tc["function"]["arguments"]

        # malformed tool JSON is data for the model; 2 consecutive aborts
        args: dict | str = raw_args
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            if not isinstance(parsed, dict):
                raise ValueError("arguments is not a JSON object")
            args = parsed
            malformed_streak = 0
        except (json.JSONDecodeError, ValueError):
            malformed_streak += 1
            self._record_tool(call_id, name, raw_args,
                              registry.err(f"error: malformed tool call JSON: {raw_args[:200]}"))
            if malformed_streak >= 2:
                self.session.record_turn_aborted("two consecutive malformed tool calls")
                self._emit("error", {"text": "two consecutive malformed tool calls — stopping"})
                return malformed_streak, "two consecutive malformed tool calls"
            return malformed_streak, None

        if name not in self._enabled:
            self._record_tool(call_id, name, args,
                              registry.err(f"error: tool {name} is disabled"))
            return malformed_streak, None

        if registry.approval_for(name, self.cfg) == "ask" and name not in self._always_allow:
            preview = make_preview(name, args, self.ctx.workspace)
            decision = self._ask(name, args, preview)
            self.session.record_approval(name, decision)
            if decision == "a":
                self._always_allow.add(name)
            if decision not in ("y", "a"):
                self._record_tool(call_id, name, args, registry.err("denied by user"))
                return malformed_streak, None

        self._emit("tool", {"name": name, "args": args})
        result = registry.execute(name, args, self.ctx)
        self._record_tool(call_id, name, args, result)
        return malformed_streak, None

    def _record_tool(self, call_id: str, name: str, args,
                     result: "registry.ToolResult") -> None:
        self.session.record_tool(call_id, name, args, result.result, result.ok)
        self.memory.append({"role": "tool", "tool_call_id": call_id,
                            "content": result.result})
        self._emit("tool_result", {"name": name, "ok": result.ok,
                                   "result": result.result})

    def _backfill_interrupted(self, tool_calls: list[dict]) -> None:
        """If the turn died mid-batch, give every unexecuted call an error
        result so the assistant+tool_calls+tool group stays complete."""
        recorded = {e.get("id") for e in self.session._events
                    if e.get("type") == "tool"}
        for tc in tool_calls:
            if tc["id"] not in recorded:
                self._record_tool(tc["id"], tc["function"]["name"],
                                  tc["function"]["arguments"],
                                  registry.err("error: interrupted before execution"))

    def _show_diff(self, ws: Path) -> None:
        parts: list[str] = []
        new_files = _git_out(["ls-files", "--others", "--exclude-standard"], ws).strip()
        if new_files:
            parts.append("new files:\n" + new_files)
        diff = _git_out(["diff", "HEAD"], ws).strip()
        if diff:
            parts.append(diff)
        self._emit("info", {"text": "workspace changes this turn:\n"
                                     + _short_text("\n\n".join(parts), 4000)})

    def _wrapup(self, turn_n: int, prompt_tokens: int, completion_tokens: int,
                summary: str, completed: bool) -> str | None:
        """Final diff + the two independent auto-commits. Returns the
        workspace commit hash (None when the workspace was unchanged)."""
        commit: str | None = None
        if self.cfg.workspace.auto_git:
            ws = Path(self.cfg.workspace.path)
            if _git_out(["status", "--porcelain"], ws).strip():
                self._show_diff(ws)
                commit = gitstore.commit_all(
                    ws, f"harness: {self.session.agent} {summary} [session:{self.session.id}]")
        if completed:
            self.session.record_turn_end(turn_n, commit, prompt_tokens, completion_tokens)
            state_msg = f"session {self.session.id}: turn {turn_n}"
        else:
            state_msg = f"session {self.session.id}: turn {turn_n} (aborted)"
        gitstore.commit_all(cfgmod.STATE_DIR, state_msg)
        bits = [f"turn {turn_n} {'done' if completed else 'aborted'}",
                f"prompt_tokens={prompt_tokens}",
                f"completion_tokens={completion_tokens}"]
        if commit:
            bits.append(f"workspace@{commit}")
        self._emit("info", {"text": "  ".join(bits)})
        return commit

    # -- manual compaction (summarize = "manual"; the TUI will call this) ---

    def compact_now(self) -> bool:
        try:
            new_mem = ensure_budget(self.memory, self.client, self.cfg, self.session)
        except BackendError as e:
            self._emit("error", {"text": f"compaction failed: {e}"})
            return False
        if new_mem is None or new_mem is self.memory:
            self._emit("info", {"text": "nothing to compact"})
            return False
        self.memory = new_mem
        self._emit("info", {"text": "context compacted"})
        return True

