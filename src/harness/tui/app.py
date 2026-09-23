"""Textual TUI for the harness.

Deliberately a thin renderer: the agent layer (agent/loop.py) is complete
and headless. This app plugs in through the AgentLoop constructor seams:

  - emit(kind, data)     -> conversation panel / status bar updates
  - ask_approval(...)    -> modal with y / n / a (blocks a worker thread,
                            resolved on the UI thread)

run_turn executes in a Textual worker thread so the UI stays responsive.
Ctrl-C during a turn calls loop.cancel() — cooperative and thread-safe; the
loop stops at the next safe point, records an interrupted event, and wraps
up. (Headless mode still uses KeyboardInterrupt, handled natively by the
loop.) A tool already executing runs on to completion or its own timeout.
"""

from __future__ import annotations

import json
import re
import time
import threading
from pathlib import Path

from rich.markup import escape
from rich.segment import Segment
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.geometry import Size
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Footer, Header, Input, Label, RichLog, Static

from .. import config as cfgmod
from ..agent import DEFAULT_AGENT, AgentLoop, Session, TurnResult
from ..backend.client import BackendClient
from ..state.seed import seed

PREVIEW_CAP = 4000
RESULT_CAP = 600
REASONING_STYLE = "dim italic"


# ---------- messages ----------

class TurnDone(Message):
    def __init__(self, result: TurnResult) -> None:
        self.result = result
        super().__init__()


class Notice(Message):
    """Cross-thread UI note (kind = an emit kind, e.g. info / error)."""
    def __init__(self, kind: str, text: str) -> None:
        self.kind = kind
        self.text = text
        super().__init__()


# ---------- widgets ----------

class Conversation(ScrollView):
    """The conversation document.

    Keeps its own document model — completed blocks plus an in-progress
    block that grows with each streamed delta — and renders it as a single
    wrapping Text.

    It is a plain ScrollView, not a RichLog: RichLog.write() re-renders the
    *entire* document into strips on every call, so a streamed answer
    (throttled to ~20/s) re-rendered the whole conversation 20x/second and
    the full-screen repaint flickered. Here only the visible window is
    rendered (render_line), and a refresh is scheduled at most once per
    frame, so the frame is painted exactly once.
    """

    def __init__(self) -> None:
        super().__init__()
        self._blocks: list[Text] = []
        self._current: Text | None = None
        self._cur_style: str | None = None
        self._lines: list[Strip] = []
        self._line_width = 0
        self._last_render = 0.0

    def stream(self, style: str, text: str) -> None:
        """Grow the in-progress block; re-renders at most ~20x/second."""
        if not text:
            return
        if self._cur_style != style:
            self.flush()
            self._cur_style = style
            self._current = Text(style=style)
        self._current.append(text)
        now = time.monotonic()
        if now - self._last_render >= 0.05:
            self._rerender()

    def flush(self) -> None:
        """Commit the in-progress block, if any."""
        if self._current is not None:
            if self._current.plain:
                self._blocks.append(self._current)
                self._blocks.append(Text("\n"))
            self._current = None
            self._cur_style = None
        self._rerender()

    def write(self, line: str | Text) -> None:
        """Commit a discrete line (tool call, result, info, error, header)."""
        self.flush()
        if isinstance(line, str):
            self._blocks.append(Text.from_markup(line))
        else:
            self._blocks.append(line)
        self._rerender()

    def clear(self) -> None:
        self._blocks.clear()
        self._current = None
        self._cur_style = None
        self._rerender()

    def _rerender(self) -> None:
        self._last_render = time.monotonic()
        self._update_lines(self._build_doc())
        self.refresh()

    def _update_lines(self, doc: Text) -> None:
        """Wrap the document into strips and update the virtual size.

        Wrapping is done here (throttled, ~20/s) rather than in
        render_line, so each visible row is a pre-built Strip and the
        compositor's per-row render is an O(1) lookup.
        """
        if not self.size:
            return
        width = self.scrollable_content_region.width
        if width <= 0:
            return
        was_at_bottom = self.is_vertical_scroll_end
        console = self.app.console
        segments = console.render(doc, console.options.update_width(width))
        self._lines = Strip.from_lines(list(Segment.split_lines(segments)))
        self._line_width = width
        self.virtual_size = Size(width, len(self._lines))
        if was_at_bottom:
            # Deferred (like RichLog.write): the scrollbar state is only
            # settled during the layout pass, so the scroll lands after
            # refresh but before the frame is drawn.
            self.scroll_end(animate=False, x_axis=False)

    def render_line(self, y: int) -> Strip:
        """Render one visible row from the pre-wrapped document.

        `y` is relative to the content region; the scroll offset is added
        here (mirroring RichLog.render_line).
        """
        row = y + self.scroll_offset.y
        if row < len(self._lines):
            return self._lines[row].apply_style(self.rich_style)
        return Strip.blank(self._line_width, self.rich_style)

    def on_resize(self, event) -> None:
        """Re-wrap the document when the terminal width changes."""
        if self.size and self._line_width != self.scrollable_content_region.width:
            self._update_lines(self._build_doc())

    def _build_doc(self) -> Text:
        doc = Text()
        for i, part in enumerate(self._blocks):
            if i:
                doc.append("\n")
            doc.append_text(part)
        if self._current is not None:
            if self._blocks or doc.plain:
                doc.append("\n")
            doc.append_text(self._current)
        return doc

class StatusBar(Static):
    model = reactive("--")
    tokens = reactive(0)
    completion_tokens = reactive(0)

    def __init__(self) -> None:
        super().__init__()
        self.session_id = "--"
        self.commit = ""

    def render(self) -> Text:
        text = Text()
        text.append(" model ", style="cyan bold")
        text.append(self.model)
        text.append(f"  ·  prompt tokens {self.tokens:,}")
        text.append(f"  ·  completion tokens {self.completion_tokens:,}")
        text.append(f"  ·  session {self.session_id}", style="green")
        if self.commit:
            text.append(f"  ·  workspace@{self.commit}", style="blue")
        return text


# ---------- modals ----------

class ApprovalModal(ModalScreen[str]):
    """y / n / a. Esc denies (never hang the worker)."""

    BINDINGS = [
        Binding("y", "yes", "allow once"),
        Binding("a", "always", "always allow"),
        Binding("n", "no", "deny"),
        Binding("escape", "no", "deny", show=False),
    ]

    def __init__(self, name: str, preview: str) -> None:
        super().__init__()
        self._name = name
        self._preview = preview

    def compose(self) -> ComposeResult:
        with Container(id="approval-modal"):
            yield Label(f"approval requested — {self._name}")
            yield RichLog(id="approval-preview", wrap=True,
                          markup=False, highlight=False)
            yield Label(escape("[y]es once    [a]lways allow this tool    "
                        "[n]o / esc — deny"))

    def on_mount(self) -> None:
        self.query_one("#approval-preview", RichLog).write(
            self._preview[:PREVIEW_CAP])

    def action_yes(self) -> None:
        self.dismiss("y")

    def action_always(self) -> None:
        self.dismiss("a")

    def action_no(self) -> None:
        self.dismiss("n")


class ModelModal(ModalScreen[str | None]):
    """Pick a model from the backend's /models list."""

    def __init__(self, models: list[str], current: str) -> None:
        super().__init__()
        self._models = models
        self._current = current

    def compose(self) -> ComposeResult:
        with Container(id="model-modal"):
            yield Label(f"choose model (current: {self._current})")
            for i, m in enumerate(self._models):
                mark = "●" if m == self._current else " "
                yield Label(f"  {i + 1}) {mark} {m}")
            yield Label("number key to select · esc to cancel")

    def on_key(self, event) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
        elif event.key.isdigit():
            i = int(event.key) - 1
            if 0 <= i < len(self._models):
                event.stop()
                self.dismiss(self._models[i])


# ---------- the app ----------

class HarnessApp(App):
    """Harness TUI. The loop is the brain; this is the face."""

    TITLE = "harness"
    CSS_PATH = Path(__file__).with_name("app.tcss")
    BINDINGS = [Binding("ctrl+c", "quit", "interrupt / quit", show=False)]

    def __init__(self) -> None:
        super().__init__()
        self.cfg: cfgmod.Config | None = None
        self.client: BackendClient | None = None
        self.session: Session | None = None
        self.loop: AgentLoop | None = None
        self._turn_active = False
        self._quitting = False
        self._approval_event = threading.Event()
        self._approval_result = "n"

    # -- lifecycle --

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-area"):
            yield Conversation()
            yield StatusBar()
        yield Footer()
        yield Input(placeholder='type a request…   (commands: /help, /models, '
                                 '/model <name>, /new, /compact, /quit)',
                    id="prompt")

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()

    def on_ready(self) -> None:
        cfg = cfgmod.load_config()
        if seed(cfg):
            cfg = cfgmod.load_config()
        self.cfg = cfg
        self.client = BackendClient(cfg)
        self.session = Session.create(DEFAULT_AGENT, cfg.backend.model)
        self._make_loop()
        bar = self.query_one(StatusBar)
        bar.model = self.client.model
        bar.session_id = self.session.id
        self._conversation().write(
            f"[dim]session {self.session.id} · model {self.client.model} · "
            f"tools {len(self.loop._enabled)}[/dim]")

    def _make_loop(self) -> None:
        self.loop = AgentLoop(self.cfg, self.client, self.session,
                              ask_approval=self._ask_approval,
                              emit=self._emit, console=None)

    def _conversation(self) -> Conversation:
        return self.query_one(Conversation)

    def action_quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._approval_result = "n"           # unblock a waiting worker
        self._approval_event.set()
        if isinstance(self.screen, ApprovalModal):
            self.screen.dismiss("n")
        self._emit("info", {"text": "bye"})
        self.exit()
 
    # -- input / commands --

    def on_key(self, event) -> None:
        if event.key == "ctrl+c":
            event.stop()
            if isinstance(self.screen, ApprovalModal):
                self._on_approval_answered("n")
                return
            if self._turn_active:
                self.loop.cancel()
                self._emit("info", {"text": "interrupt requested (takes effect "
                                            "at the next safe point)"})
                return
            self.action_quit()
            return
        input_w = self.query_one("#prompt", Input)
        value = input_w.value
        if not value.startswith("/"):
            return
        event.stop()
        input_w.value = ""
        cmd, _, arg = value[1:].partition(" ")
        self._dispatch(cmd.strip().lower(), arg.strip())

    def _dispatch(self, cmd: str, arg: str) -> None:
        if cmd == "help":
            self._emit("info", {"text": (
                "/models — list backend models · /model <name> — switch "
                "(persists) · /new — new session · /compact — compact now "
                "(manual mode) · /quit — exit · Ctrl-C — interrupt turn / quit")})
        elif cmd == "new":
            if self._turn_active:
                self._emit("info", {"text": "a turn is running — finish it first"})
                return
            self.session = Session.create(DEFAULT_AGENT, self.client.model)
            self._make_loop()
            bar = self.query_one(StatusBar)
            bar.session_id = self.session.id
            bar.tokens = 0
            bar.completion_tokens = 0
            bar.commit = ""
            self._conversation().clear()
            self._conversation().write(f"[dim]new session {self.session.id}[/dim]")
        elif cmd == "models":
            self._list_models()
        elif cmd == "model":
            if not arg:
                self._emit("info", {"text": f"current model: {self.client.model}"})
            else:
                self._switch_model(arg)
        elif cmd == "compact":
            if self._turn_active:
                self._emit("info", {"text": "a turn is running — finish it first"})
                return
            self.run_worker(self._compact_worker, description="compact",
                            group="turns", thread=True)
            self._emit("info", {"text": "compacting…"})
        elif cmd == "quit":
            self.action_quit()
        else:
            self._emit("info", {"text": f"unknown command /{cmd} — try /help"})

    # -- turns --

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._turn_active:
            self._emit("info", {"text": "already running — Ctrl-C to interrupt"})
            return
        prompt = event.value.strip()
        event.input.value = ""
        if not prompt:
            return
        conv = self._conversation()
        conv.flush()
        conv.write(Text("─ turn " + str(self.session.turn_number + 1) + " " * 24,
                        style="dim"))        
        self._turn_active = True
        self.run_worker(lambda: self._run_turn_worker(prompt), name="turn", group="turns", thread=True)

    def _run_turn_worker(self, prompt: str) -> None:
        try:
            result = self.loop.run_turn(prompt)
        except Exception as e:  # belt-and-braces: run_turn should not raise
            result = TurnResult(False, f"loop crashed: {type(e).__name__}: {e}")
        finally:
            self.call_from_thread(self._turn_inactive)
        self.post_message(TurnDone(result))

    def _compact_worker(self) -> None:
        try:
            self.loop.compact_now()
        except Exception as e:
            self.post_message(Notice("error", f"compaction failed: {e}"))

    def _turn_inactive(self) -> None:
        self._turn_active = False

    @on(TurnDone)
    def _on_turn_done(self, msg: TurnDone) -> None:
        bar = self.query_one(StatusBar)
        bar.tokens = msg.result.prompt_tokens
        bar.completion_tokens = msg.result.completion_tokens
        if msg.result.commit:
            bar.commit = msg.result.commit
        if msg.result.ok:
            self._emit("info", {"text": f"turn done · prompt_tokens="
                                        f"{msg.result.prompt_tokens:,} · "
                                        f"completion_tokens="
                                        f"{msg.result.completion_tokens:,}"
                                        + (f" · workspace@{msg.result.commit}"
                                           if msg.result.commit else "")})
        else:
            self._emit("error", {"text": f"turn failed: {msg.result.reason}"})

    @on(Notice)
    def _on_notice(self, msg: Notice) -> None:
        self._emit(msg.kind, {"text": msg.text})

    # -- emit seam (called from worker threads) --

    def _emit(self, kind: str, data: dict) -> None:
        conv = self._conversation()
        if kind == "text":
            conv.stream("text", data["text"])
            if "tokens" in data:
                self.query_one(StatusBar).completion_tokens = data["tokens"]
        elif kind == "reasoning":
            conv.stream(REASONING_STYLE, data["text"])
            if "tokens" in data:
                self.query_one(StatusBar).completion_tokens = data["tokens"]
        elif kind == "tool":
            conv.flush()
            args = data.get("args") or {}
            if not isinstance(args, dict):
                args = {"raw": args}
            conv.write(f"[cyan]⚙ {escape(data['name'])} "
                       f"{escape(json.dumps(args)[:200])}[/cyan]")
        elif kind == "tool_result":
            mark = "[green]✓[/green]" if data["ok"] else "[red]✗[/red]"
            conv.write(f"  {mark} {escape(str(data['result'])[:RESULT_CAP])}")
        elif kind == "error":
            conv.flush()
            conv.write(f"[bold red]error: {escape(data['text'])}[/bold red]")
        else:  # info
            conv.flush()
            conv.write(f"[dim]{escape(data['text'])}[/dim]")

    # -- approval seam (called from the worker thread) --
        # -- approval seam (called from the worker thread) --

    def _ask_approval(self, name: str, args: dict, preview: str) -> str:
        self._approval_event.clear()
        self._approval_result = "n"
        self.call_from_thread(self._push_approval, name, preview)
        try:
            while not self._approval_event.wait(timeout=0.5):
                if self._quitting:
                    self._approval_result = "n"
                    break
        finally:
            self.call_from_thread(self._close_approval_if_open)
        return self._approval_result

    def _push_approval(self, name: str, preview: str) -> None:
        self.push_screen(ApprovalModal(name, preview),
                         callback=self._on_approval_answered)

    def _close_approval_if_open(self) -> None:
        if isinstance(self.screen, ApprovalModal):
            self.screen.dismiss("n")

    def _on_approval_answered(self, decision: str | None) -> None:
        self._approval_result = decision if decision in ("y", "n", "a") else "n"
        self._approval_event.set()

    # -- model management --

    def _list_models(self) -> None:
        self._emit("info", {"text": "querying /models …"})

        def worker() -> None:
            try:
                models = self.client.list_models()
            except Exception as e:
                self.post_message(Notice("error", f"model list failed: {e}"))
                return
            self.call_from_thread(self._open_model_modal, models)

        self.run_worker(worker, description="models", thread=True)

    def _open_model_modal(self, models: list[str]) -> None:
        if not models:
            self._emit("error", {"text": "backend returned no models"})
            return
        self.push_screen(ModelModal(models, self.client.model),
                         callback=self._on_model_picked)

    def _on_model_picked(self, model: str | None) -> None:
        if not model or model == self.client.model:
            return
        self._switch_model(model)

    def _switch_model(self, model: str) -> None:
        self.client.model = model
        try:
            _persist_model(cfgmod.CONFIG_PATH, model)
            persisted = " · saved to config.toml"
        except OSError as e:
            persisted = f" · persist failed: {e}"
        self.session.append({"type": "model", "name": model})
        self.query_one(StatusBar).model = model
        self._emit("info", {"text": f"model → {model}{persisted}"})


def _persist_model(path: Path, model: str) -> None:
    """Set [backend] model in config.toml (line replace, or append)."""
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?m)^model\s*=", text):
            new = re.sub(r"(?m)^model\s*=\s*.+$", f'model = "{model}"', text,
                         count=1)
        else:
            new = text.rstrip("\n") + f'\n\n[backend]\nmodel = "{model}"\n'
        path.write_text(new, encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[backend]\nmodel = "{model}"\n', encoding="utf-8")


def harness_tui() -> None:
    HarnessApp().run()


if __name__ == "__main__":
    harness_tui()

