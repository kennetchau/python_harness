"""Entry point: seed state, load config, print a summary.

Pass a prompt to run one headless turn:
    python -m harness "create hello.txt containing: hi"

The Textual TUI replaces this entry point once it lands.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from . import config as cfgmod
from .state.seed import seed

console = Console()


def _print_config(cfg: cfgmod.Config) -> None:
    lines = [
        f"backend    {cfg.backend.base_url}  model={cfg.backend.model}",
        f"context    max={cfg.context.max_tokens}  summarize={cfg.context.summarize} "
        f"(threshold={cfg.context.summarize_threshold}, keep={cfg.context.keep_recent_turns})",
        f"workspace  {cfg.workspace.path}  auto_git={cfg.workspace.auto_git}",
        f"tools      enabled={cfg.tools.enabled}  "
        f"ask={[n for n, lvl in cfg.tools.approval.items() if lvl == 'ask']}",
        f"search     engine={cfg.search.engine}  max_results={cfg.search.max_results}",
    ]
    console.print(Panel("\n".join(lines), title="harness config", border_style="cyan"))


def _run_one_turn(cfg: cfgmod.Config, prompt: str) -> int:
    from .agent import DEFAULT_AGENT, AgentLoop, Session
    from .backend.client import BackendClient

    session = Session.create(DEFAULT_AGENT, cfg.backend.model)
    console.print(f"session [cyan]{session.id}[/cyan] -> {session.path}")
    client = BackendClient(cfg)
    try:
        loop = AgentLoop(cfg, client, session)
        result = loop.run_turn(prompt)
    finally:
        client.close()
    if not result.ok:
        console.print(f"[red]turn failed: {result.reason}[/red]")
        return 1
    bits = ["[green]turn ok[/green]", f"session={session.id}"]
    if result.commit:
        bits.append(f"workspace@{result.commit}")
    bits.append(f"prompt_tokens={result.prompt_tokens}")
    console.print("  ".join(bits))
    return 0


def main() -> int:
    cfg = cfgmod.load_config()
    if seed(cfg):
        cfg = cfgmod.load_config()  # reload: the file just written may differ from defaults
    _print_config(cfg)
    if len(sys.argv) > 1:
        return _run_one_turn(cfg, " ".join(sys.argv[1:]))
    console.print('Headless agent loop ready — pass a prompt to run a turn: '
                  'python -m harness "do something"   (next: TUI)')
    if len(sys.argv) > 1 and sys.argv[1] == "tui":
        try:
            from .tui import harness_tui
        except ImportError as e:
            console.print(f"[red]TUI unavailable: {e}[/red]")
            return 1
        harness_tui()
        return 0


    return 0


if __name__ == "__main__":
    sys.exit(main())

