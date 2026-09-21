"""Harness configuration.

Loads /state/config.toml (git-tracked) over built-in defaults.
Unknown keys are ignored; invalid enum values fall back to the
default with a stderr warning. This module never raises.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

STATE_DIR = Path(os.environ.get("HARNESS_STATE", "/state"))
CONFIG_PATH = STATE_DIR / "config.toml"

APPROVAL_LEVELS = ("off", "ask", "auto")
SUMMARIZE_MODES = ("off", "auto", "manual")

DEFAULT_APPROVAL: dict[str, str] = {
    "list_dir": "auto",
    "read_file": "auto",
    "grep": "auto",
    "glob": "auto",
    "git_diff": "auto",
    "web_search": "auto",
    "web_fetch": "auto",
    "write_file": "ask",
    "edit_file": "ask",
    "run_command": "ask",
}


@dataclass(frozen=True)
class BackendConfig:
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "none"
    model: str = "default"
    temperature: float = 0.2
    max_tokens: int = 50000
    thinking_budget: int = 8192


@dataclass(frozen=True)
class ContextConfig:
    max_tokens: int = 240000
    summarize: str = "auto"
    summarize_threshold: float = 0.7
    keep_recent_turns: int = 8
    summary_max_tokens: int = 4096


@dataclass(frozen=True)
class WorkspaceConfig:
    path: str = "/workspace"
    auto_git: bool = True


@dataclass(frozen=True)
class LimitsConfig:
    max_read_lines: int = 2000
    exec_timeout_sec: int = 300
    exec_output_chars: int = 20000
    web_max_chars: int = 50000
    exec_network: bool = True
    max_tool_rounds: int = 25


@dataclass(frozen=True)
class ToolsConfig:
    enabled: bool = True
    approval: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_APPROVAL))
    limits: LimitsConfig = field(default_factory=LimitsConfig)


@dataclass(frozen=True)
class SearchConfig:
    engine: str = "ddgs"
    max_results: int = 5


@dataclass(frozen=True)
class Config:
    backend: BackendConfig
    context: ContextConfig
    workspace: WorkspaceConfig
    tools: ToolsConfig
    search: SearchConfig


def pick(data: dict, cls: type) -> dict:
    """Keep only keys that are fields of dataclass cls."""
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in allowed}


def enum_check(value, allowed, default, where) -> str:
    if value not in allowed:
        print(f"[config] {where}: {value!r} invalid (want {allowed}), using {default!r}",
              file=sys.stderr)
        return default
    return value


def build_config(raw: dict) -> Config:
    ctx = pick(raw.get("context", {}), ContextConfig)
    ctx["summarize"] = enum_check(
        ctx.get("summarize", "auto"), SUMMARIZE_MODES, "auto", "context.summarize")

    tools_raw = raw.get("tools", {})
    limits = LimitsConfig(**pick(tools_raw.get("limits", {}), LimitsConfig))

    approval = dict(DEFAULT_APPROVAL)
    for name, level in tools_raw.get("approval", {}).items():
        approval[name] = enum_check(
            level, APPROVAL_LEVELS, approval.get(name, "ask"), f"tools.approval.{name}")

    tools = pick(tools_raw, ToolsConfig)
    tools["approval"] = approval
    tools["limits"] = limits

    ws = pick(raw.get("workspace", {}), WorkspaceConfig)
    ws.setdefault("path", os.environ.get("HARNESS_WORKSPACE", "/workspace"))

    return Config(
        backend=BackendConfig(**pick(raw.get("backend", {}), BackendConfig)),
        context=ContextConfig(**ctx),
        workspace=WorkspaceConfig(**ws),
        tools=ToolsConfig(**tools),
        search=SearchConfig(**pick(raw.get("search", {}), SearchConfig)),
    )

def load_config(path: Path = CONFIG_PATH) -> Config:
    raw: dict = {}
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)
    return build_config(raw)


DEFAULT_CONFIG_TEMPLATE = """\
# Harness configuration. Git-tracked in /state.

[backend]
base_url = "http://localhost:8080/v1"
api_key = "none"
model = "default"
temperature = 0.2
max_tokens = 50000
thinking_budget = 8192

[context]
max_tokens = 240000
summarize = "auto"            # off | auto | manual
summarize_threshold = 0.7     # compact when prompt tokens exceed max_tokens * threshold
keep_recent_turns = 8
summary_max_tokens = 4096

[workspace]
# path = "/workspace"
auto_git = true

[tools]
enabled = true

[tools.approval]              # off | ask | auto
list_dir = "auto"
read_file = "auto"
grep = "auto"
glob = "auto"
git_diff = "auto"
web_search = "auto"
web_fetch = "auto"
write_file = "ask"
edit_file = "ask"
run_command = "ask"

[tools.limits]
max_read_lines = 2000
exec_timeout_sec = 300
exec_output_chars = 20000
web_max_chars = 50000
exec_network = true           # false wraps commands in `unshare -n`
max_tool_rounds = 25

[search]
engine = "ddgs"
max_results = 5
"""

