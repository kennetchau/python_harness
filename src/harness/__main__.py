"""Entry point: seed state, load config, print a summary.

The Textual TUI replaces the summary print once the agent loop lands.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from . import config as cfgmod
from .state.seed import seed

console = Console()


def main() -> int:
    cfg = cfgmod.load_config()
    if seed(cfg):
        cfg = cfgmod.load_config()  # reload: the file just written may differ from defaults

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
    console.print("Scaffold OK — next: tools/ (registry + files), backend client, agent loop, TUI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

