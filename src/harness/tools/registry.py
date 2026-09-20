"""Tool registry: schemas, dispatch, path jail, output caps.

Tools return ToolResult — errors are data for the model, never
exceptions. The dispatch catch is the only last-resort guard in
the tools layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import config as cfgmod


@dataclass(frozen=True)
class ToolResult:
    result: str
    ok: bool = True


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable


REGISTRY: dict[str, Tool] = {}


def res(text: str) -> ToolResult:
    return ToolResult(text)


def err(text: str) -> ToolResult:
    return ToolResult(text, ok=False)


def register(name: str, description: str, parameters: dict,
             fn: Callable[[dict, "ToolContext"], ToolResult]) -> Tool:
    tool = Tool(name=name, description=description, parameters=parameters, fn=fn)
    REGISTRY[name] = tool
    return tool


def enabled_names(cfg: cfgmod.Config) -> list[str]:
    """Registration-ordered names the model may see: tools on, approval != off."""
    if not cfg.tools.enabled:
        return []
    return [n for n in REGISTRY if cfg.tools.approval.get(n, "auto") != "off"]


def schemas(cfg: cfgmod.Config) -> list[dict]:
    """OpenAI tools array for the backend request."""
    return [
        {"type": "function",
         "function": {"name": t.name, "description": t.description,
                      "parameters": t.parameters}}
        for n in enabled_names(cfg)
        for t in (REGISTRY[n],)
    ]


def approval_for(name: str, cfg: cfgmod.Config) -> str:
    return cfg.tools.approval.get(name, "auto")


def execute(name: str, args: dict, ctx: "ToolContext") -> ToolResult:
    """Run a tool by name. Never raises for tool-level problems."""
    tool = REGISTRY.get(name)
    if tool is None:
        return err(f"error: unknown tool {name!r}")
    try:
        return tool.fn(args or {}, ctx)
    except Exception as exc:
        return err(f"error: {type(exc).__name__}: {exc}")


def safe_path(root: Path, raw: str) -> Path | None:
    """Resolve a workspace-relative path. None when it escapes the root."""
    if not raw:
        return None
    candidate = (root / raw).resolve()
    root_resolved = root.resolve()
    if candidate == root_resolved or root_resolved in candidate.parents:
        return candidate
    return None


def cap(text: str, limit: int) -> str:
    """Truncate to limit chars, preferring a line boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    newline = cut.rfind("\n")
    if newline > limit // 2:
        cut = cut[:newline]
    return cut + f"\n...[truncated at {limit} chars]"


@dataclass(frozen=True)
class ToolContext:
    cfg: cfgmod.Config
    workspace: Path

    @property
    def limits(self) -> cfgmod.LimitsConfig:
        return self.cfg.tools.limits

    def safe(self, raw: str) -> Path | None:
        return safe_path(self.workspace, raw)

    def jail_error(self, raw: str) -> ToolResult:
        return err(f"error: {raw or '(empty path)'} escapes the workspace")

