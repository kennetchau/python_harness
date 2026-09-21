# harness

An AI harness for code creation and project updates. A headless agent loop
drives an OpenAI-compatible backend to work on a git-tracked workspace, with
per-tool approval, append-only JSONL session logs, automatic context
compaction, and a Textual TUI on top.

## Requirements

- Python 3.14 (managed with [uv](https://docs.astral.sh/uv/); see `.python-version`)
- An OpenAI-compatible chat backend (default: `http://localhost:8080/v1`)

## Setup

```sh
uv sync
```

On first run, `harness` seeds its state directory (default `/state`) with a
default `config.toml` and initializes git repos for both the state directory
and the workspace (default `/workspace`). Seeding is idempotent.

## Usage

Headless — print the config summary and run one agent turn:

```sh
uv run python -m harness "create hello.txt containing: hi"
```

Textual TUI:

```sh
uv run python -m harness.tui
```

TUI commands (typed in the prompt input):

| Command | Effect |
| --- | --- |
| `/help` | Show available commands |
| `/models` | List models served by the backend |
| `/model <name>` | Switch model (persisted to `config.toml`) |
| `/new` | Start a new session |
| `/compact` | Compact the context now (manual mode) |
| `/quit` | Exit |

`Ctrl-C` interrupts a running turn (the loop stops at the next safe point and
discards partial output) or quits when idle.

## Configuration

Config is loaded from `$HARNESS_STATE/config.toml` (default
`/state/config.toml`) over built-in defaults. Unknown keys are ignored;
invalid enum values fall back to the default with a stderr warning. The
default file is written on first run and is git-tracked.

| Section | Keys | Notes |
| --- | --- | --- |
| `[backend]` | `base_url`, `api_key`, `model`, `temperature`, `max_tokens` | OpenAI-compatible endpoint |
| `[context]` | `max_tokens`, `summarize`, `summarize_threshold`, `keep_recent_turns`, `summary_max_tokens` | `summarize` = `off` \| `auto` \| `manual`; auto compaction triggers when prompt tokens exceed `max_tokens * summarize_threshold` |
| `[workspace]` | `path`, `auto_git` | Workspace root; auto-commit workspace changes each turn |
| `[tools]` | `enabled` | Master switch for all tools |
| `[tools.approval]` | per-tool level | `off` (hidden) \| `ask` (prompt) \| `auto` (run freely) |
| `[tools.limits]` | `max_read_lines`, `exec_timeout_sec`, `exec_output_chars`, `web_max_chars`, `exec_network`, `max_tool_rounds` | `exec_network = false` runs commands in a fresh network namespace via `unshare -n` |
| `[search]` | `engine`, `max_results` | Web search settings |

Environment variables:

| Variable | Effect |
| --- | --- |
| `HARNESS_STATE` | State directory (default `/state`) |
| `HARNESS_WORKSPACE` | Workspace path (default `/workspace`) |
| `HARNESS_CA_BUNDLE` | CA bundle for backend TLS verification |
| `HARNESS_TLS_VERIFY` | Set to `0`/`false`/`no`/`off` to disable TLS verification |

## How it works

**Agent loop** (`src/harness/agent/loop.py`) — one `run_turn(user_message)`
call drives a full turn: budget check (compaction) → streaming request →
(tool calls → approval → execute → tool results)\* → final message → wrap-up
(final diff + auto-commits). Memory is a plain list of OpenAI wire-format
messages kept in lockstep with the session log, so a resumed session replays
to exactly the same memory. Tool errors are data for the model; only
backend/loop failures abort the turn, and an aborted or interrupted turn
still commits its side effects.

**Sessions** (`src/harness/agent/session.py`) — append-only JSONL event logs
under `/state/sessions/<id>.jsonl` (id: `YYYYMMDD-HHMM-<agent>`). Events are
the audit log; `replay()` rebuilds working memory as: system message + all
summaries in order + all message/tool events after the last summary.

**Compaction** (`src/harness/agent/compaction.py`) — when the context budget
is exceeded, the oldest real turns (keeping the most recent
`keep_recent_turns`) are summarized into a single summary message. Cuts land
on turn boundaries so assistant/tool-call groups are never split, and
existing summaries are never re-summarized, keeping memory consistent with
replay.

**Auto-commits** (`src/harness/state/gitstore.py`) — with `auto_git`, each
turn commits workspace changes (message: first line of the final reply) and
state changes (`session <id>: turn <n>`). Git failures warn but never kill a
session.

**Backend client** (`src/harness/backend/client.py`) — streaming
chat/completions plus one-shot completions (used for compaction) and
`GET /models`. One retry with backoff on 5xx/connection errors; the single
exception type, `BackendError`, aborts the turn but not the session.

## Tools

All paths are workspace-relative and jailed to the workspace root. Tool
errors are returned as data, never raised.

| Tool | Description | Default approval |
| --- | --- | --- |
| `list_dir` | List directory entries | `auto` |
| `read_file` | Read a text file, optionally a line range | `auto` |
| `grep` | Regex search, returns `path:line: text` hits | `auto` |
| `glob` | List files matching a glob pattern | `auto` |
| `git_diff` | Unified diff of the workspace vs HEAD | `auto` |
| `web_search` | Web search via `ddgs` | `auto` |
| `web_fetch` | Fetch a URL as plain text (`httpx` + `trafilatura`) | `auto` |
| `write_file` | Create or overwrite a file | `ask` |
| `edit_file` | Replace an exact string (unique match required unless `replace_all`) | `ask` |
| `run_command` | Run a shell command in the workspace | `ask` |

When approval is `ask`, the user sees a preview (diff for edits, command
text for exec, URL for web) and answers yes / no / always-allow.

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
