# harness

An AI harness for code creation and project updates. A headless agent loop drives an
OpenAI-compatible backend to work on a git-tracked workspace, with per-tool approval,
append-only JSONL session logs, automatic context compaction, and a Textual TUI on top.

Built with Python 3.14 (managed with uv) and backed by any OpenAI-compatible chat
endpoint. It can be run in a container via `run.sh` or directly as a headless one-shot
(`python -m harness "prompt"`) or an interactive TUI (`python -m harness.tui`).

## Key features

- **Agent loop** — one `run_turn(user_message)` call drives a full turn: context-budget
  check (compaction) → streaming request → repeated tool calls with per-tool approval →
  final message → wrap-up (final diff + auto-commits). Tool errors are returned as data
  to the model; only backend/loop failures abort a turn.
- **Sessions** — append-only JSONL event logs under the state directory that serve as an
  audit log; `replay()` deterministically rebuilds working memory, so a resumed session
  replays to exactly the same memory.
- **Context compaction** — when the context budget is exceeded, the oldest turns are
  summarized into a single summary message, cutting on turn boundaries and checking the
  budget after every tool round, not just at turn start.
- **Auto-commits** — with `auto_git`, each turn commits workspace and state changes to git.
- **Jailed, approvable tools** — filesystem (list/read/write/edit/grep/glob/delete),
  shell (`run_command`, optionally network-isolated via `unshare -n`), git diff, and web
  tools (`web_search` via ddgs, `web_fetch` via httpx + trafilatura). Read-only tools run
  freely by default; mutating tools prompt the user for approval.
- **Configuration** — TOML config (`config.toml` in the state dir) over built-in defaults,
  covering backend endpoint, context/compaction, workspace, tool approval levels, tool
  limits, and web search settings; plus environment variable overrides.

## Project layout

```
src/harness/
├── __main__.py      # headless entry point: seed, config summary, one turn
├── config.py        # config loading + defaults
├── agent/           # agent loop, session store, context compaction
├── backend/         # OpenAI-compatible streaming client
├── state/           # first-run seeding, git helpers
├── tools/           # tool registry + file/shell/web tools
└── tui/             # Textual TUI (renderer on top of the headless loop)
```
