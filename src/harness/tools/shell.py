"""run_command: execute shell commands in the workspace.

When limits.exec_network is false, commands run inside a fresh
network namespace (unshare -n). If unshare is unavailable we refuse
to run with network instead of silently relaxing the sandbox.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess

from .registry import ToolContext, cap, err, res, register


def _shell_argv(ctx: ToolContext, command: str, sandboxed: bool) -> list[str]:
    if not sandboxed:
        return ["bash", "-c", command]
    return ["unshare", "-n", "bash", "-c", command]


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the whole process group, not just the direct child.

    A `bash -c` command may spawn its own children (builds, servers,
    pipelines). Killing only the direct child leaks those, so pids keep
    piling up until the process limit is hit. The group only exists
    because we started it (start_new_session=True), so signaling it
    cannot reach the harness itself.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def _run(argv: list[str], ctx: ToolContext, command_display: str):
    timeout = ctx.limits.exec_timeout_sec
    proc = subprocess.Popen(argv, cwd=ctx.workspace,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True,
                            start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
        code = proc.returncode
        suffix = ""
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        # Bounded second wait: communicate can only re-raise if a child
        # that survived the group kill still holds the pipe open.
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            output = ""
        code, suffix = 124, f" (killed after {timeout}s)"
    output = (output or "").strip()
    body = cap(output, ctx.limits.exec_output_chars) if output else "(no output)"
    return (f"exit {code}{suffix}\n${command_display}\n{body}", code == 0)


def _run_command(args: dict, ctx: ToolContext):
    command = args.get("command", "")
    if not command:
        return err("error: command is required")
    command_display = cap(command, 500)
    limits = ctx.limits
    if limits.exec_network:
        text, ok = _run(_shell_argv(ctx, command, sandboxed=False), ctx, command_display)
        return res(text) if ok else err(text)
    if shutil.which("unshare") is None:
        return err("error: exec_network=false but unshare is unavailable; "
                   "refusing to run with network")
    probe = subprocess.run(["unshare", "-n", "true"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        detail = (probe.stderr or "").strip()
        return err(f"error: network sandbox unavailable ({detail}); "
                   "refusing to run with network")
    text, ok = _run(_shell_argv(ctx, command, sandboxed=True), ctx, command_display)
    return res(text) if ok else err(text)


register(
    "run_command",
    "Run a shell command in the workspace. Working directory is the workspace "
    "root; venv and system tools are on PATH. Network availability follows "
    "the exec_network limit. Long output is truncated.",
    {"type": "object",
     "properties": {"command": {"type": "string",
                                "description": "Shell command line to run"}},
     "required": ["command"]},
    _run_command)

