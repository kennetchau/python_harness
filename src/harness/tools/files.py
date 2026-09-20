"""File tools: list_dir, read_file, write_file, edit_file, delete_file,
grep, glob, git_diff. All paths are workspace-relative and jailed."""

from __future__ import annotations

import fnmatch
import re
import subprocess

from .registry import ToolContext, cap, err, register, res


def _read_text(path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _list_dir(args: dict, ctx: ToolContext):
    raw = args.get("path", ".")
    p = ctx.safe(raw)
    if p is None:
        return ctx.jail_error(raw)
    if not p.is_dir():
        return err(f"error: {raw}: not a directory")
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    lines = [e.name + ("/" if e.is_dir() else "") for e in entries]
    return res(cap("\n".join(lines), ctx.limits.exec_output_chars))


def _read_file(args: dict, ctx: ToolContext):
    raw = args.get("path", "")
    p = ctx.safe(raw)
    if p is None:
        return ctx.jail_error(raw)
    if not p.is_file():
        return err(f"error: {raw}: no such file")
    text = _read_text(p)
    if "\x00" in text[:8192]:
        return err(f"error: {raw}: binary file, not readable as text")
    lines = text.splitlines()
    total = len(lines)
    offset = max(1, int(args.get("offset", 1)))
    limit = min(int(args.get("limit", ctx.limits.max_read_lines)),
                ctx.limits.max_read_lines)
    if offset > total:
        return err(f"error: {raw}: offset {offset} is past end of file ({total} lines)")
    chunk = lines[offset - 1: offset - 1 + limit]
    header = f"# {raw} lines {offset}-{offset + len(chunk) - 1} of {total}"
    return res(cap(header + "\n" + "\n".join(chunk), ctx.limits.exec_output_chars))


def _write_file(args: dict, ctx: ToolContext):
    raw = args.get("path", "")
    p = ctx.safe(raw)
    if p is None:
        return ctx.jail_error(raw)
    content = args.get("content")
    if not isinstance(content, str):
        return err("error: content must be a string")
    if p.exists() and not p.is_file():
        return err(f"error: {raw}: exists and is not a file")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return res(f"wrote {len(content)} chars to {raw}")


def _edit_file(args: dict, ctx: ToolContext):
    raw = args.get("path", "")
    p = ctx.safe(raw)
    if p is None:
        return ctx.jail_error(raw)
    if not p.is_file():
        return err(f"error: {raw}: no such file")
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return err("error: old_string and new_string must be strings")
    text = _read_text(p)
    count = text.count(old)
    if count == 0:
        return err(f"error: {raw}: old_string not found (exact match required, "
                   f"including whitespace and indentation)")
    replace_all = bool(args.get("replace_all"))
    if count > 1 and not replace_all:
        return err(f"error: {raw}: old_string matches {count} times; add "
                   f"surrounding context to make it unique or set replace_all=true")
    n = count if replace_all else 1
    p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1),
                 encoding="utf-8")
    return res(f"edited {raw}: replaced {n} occurrence(s)")


def _delete_file(args: dict, ctx: ToolContext):
    raw = args.get("path", "")
    p = ctx.safe(raw)
    if p is None:
        return ctx.jail_error(raw)
    if not p.is_file():
        return err(f"error: {raw}: not a file (directories can't be deleted)")
    p.unlink()
    return res(f"deleted {raw}")


def _iter_files(base, include: str | None):
    if base.is_file():
        yield base
        return
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        if include and not fnmatch.fnmatch(f.name, include):
            continue
        yield f


def _grep(args: dict, ctx: ToolContext):
    pattern = args.get("pattern", "")
    if not pattern:
        return err("error: pattern is required")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return err(f"error: invalid regex: {e}")
    raw = args.get("path", ".")
    base = ctx.safe(raw)
    if base is None:
        return ctx.jail_error(raw)
    if not base.exists():
        return err(f"error: {raw}: no such path")
    include = args.get("include")
    hits: list[str] = []
    files = list(_iter_files(base, include))
    for f in files:
        rel = f.relative_to(ctx.workspace).as_posix()
        text = _read_text(f)
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {line}")
                if len("\n".join(hits)) > ctx.limits.exec_output_chars:
                    break
        if len("\n".join(hits)) > ctx.limits.exec_output_chars:
            break
    if not hits:
        return res(f"no matches for {pattern!r} in {raw or '.'}")
    return res(cap("\n".join(hits), ctx.limits.exec_output_chars))


def _glob(args: dict, ctx: ToolContext):
    pattern = args.get("pattern", "")
    if not pattern:
        return err("error: pattern is required")
    matches = [p for p in ctx.workspace.glob(pattern) if p.is_file()]
    matches.sort(key=lambda p: p.relative_to(ctx.workspace).as_posix())
    if not matches:
        return res(f"no files match {pattern!r}")
    lines = [p.relative_to(ctx.workspace).as_posix() for p in matches]
    return res(cap("\n".join(lines), ctx.limits.exec_output_chars))


def _git_diff(args: dict, ctx: ToolContext):
    proc = subprocess.run(["git", "diff", "HEAD"], cwd=ctx.workspace,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return err(f"error: git diff failed: {proc.stderr.strip()}")
    if not proc.stdout.strip():
        return res("no changes vs HEAD")
    return res(cap(proc.stdout, ctx.limits.exec_output_chars))


register(
    "list_dir",
    "List directory entries (directories marked with a trailing /).",
    {"type": "object",
     "properties": {"path": {"type": "string",
                             "description": "Directory relative to the workspace root (default '.')"}}},
    _list_dir)

register(
    "read_file",
    "Read a text file, optionally a line range.",
    {"type": "object",
     "properties": {
         "path": {"type": "string", "description": "File relative to the workspace root"},
         "offset": {"type": "integer", "minimum": 1,
                    "description": "First line to read, 1-based (default 1)"},
         "limit": {"type": "integer", "minimum": 1,
                   "description": "Max lines to read (default: configured max_read_lines)"}},
     "required": ["path"]},
    _read_file)

register(
    "write_file",
    "Create or overwrite a file with the given content. Parent dirs are created.",
    {"type": "object",
     "properties": {
         "path": {"type": "string", "description": "File relative to the workspace root"},
         "content": {"type": "string", "description": "Full file content"}},
     "required": ["path", "content"]},
    _write_file)

register(
    "edit_file",
    "Replace an exact string in a file. Fails unless old_string matches exactly "
    "once, unless replace_all is true.",
    {"type": "object",
     "properties": {
         "path": {"type": "string", "description": "File relative to the workspace root"},
         "old_string": {"type": "string", "description": "Exact text to find"},
         "new_string": {"type": "string", "description": "Replacement text"},
         "replace_all": {"type": "boolean",
                         "description": "Replace every occurrence (default false)"}},
     "required": ["path", "old_string", "new_string"]},
    _edit_file)

register(
    "delete_file",
    "Delete a file.",
    {"type": "object",
     "properties": {"path": {"type": "string",
                             "description": "File relative to the workspace root"}},
     "required": ["path"]},
    _delete_file)

register(
    "grep",
    "Search files for a regular expression. Returns 'path:line: text' hits.",
    {"type": "object",
     "properties": {
         "pattern": {"type": "string", "description": "Regular expression"},
         "path": {"type": "string",
                  "description": "File or directory to search (default '.')"},
         "include": {"type": "string",
                     "description": "Filename glob filter, e.g. '*.py' (default: all)"}},
     "required": ["pattern"]},
    _grep)

register(
    "glob",
    "List files matching a glob pattern, relative to the workspace root.",
    {"type": "object",
     "properties": {"pattern": {"type": "string",
                                "description": "Glob pattern, e.g. 'src/**/*.py'"}},
     "required": ["pattern"]},
    _glob)

register(
    "git_diff",
    "Show the unified diff of the workspace against HEAD (staged + unstaged).",
    {"type": "object", "properties": {}},
    _git_diff)

