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
import threading

from .registry import ToolContext, cap, err, res, register

_REAPER: threading.Thread | None = None


def _ensure_reaper() -> None:
    """Start (once, and only as PID 1) a daemon thread that reaps orphans.

    In the container the harness process *is* PID 1 (run.sh starts it
    directly), so anything it fails to wait() lingers as a zombie
    forever. On a run_command timeout we kill the command's whole
    process group, but grandchildren get re-parented to PID 1
    *asynchronously* — a one-shot reap in the kill path races against
    that re-parenting and misses them. This reaper collects them shortly
    after they die.

    Only zombies that are (a) dead children of us and (b) *not* session
    leaders are touched. The direct run_command child is a session
    leader (start_new_session, so pgrp == pid) and is reaped by
    Popen.communicate() itself; if the reaper wait()'d it first,
    CPython's Popen would observe ECHILD and record a wrong exit status.
    Orphaned grandchildren keep their dead parent's group (pgrp != pid),
    so they are the only ones collected.

    Gated on getpid() == 1: with `podman run --init` (preferred) the
    orphans are re-parented to the init binary and reaped by it, and a
    reaper thread here would have nothing to collect — so it isn't even
    started.
    """
    global _REAPER
    if _REAPER is not None and _REAPER.is_alive():
        return
    if os.getpid() != 1:
        return  # a real parent reaps our orphans for us
    our_pid = os.getpid()

    def _reap():
        while True:
            try:
                for entry in os.scandir("/proc"):
                    if not entry.name.isdigit():
                        continue
                    pid = int(entry.name)
                    try:
                        with open(f"/proc/{pid}/stat") as f:
                            data = f.read()
                    except OSError:
                        continue
                    # Fields after the closing ')' of comm (which may
                    # contain spaces): state(0) ppid(1) pgrp(2) ...
                    fields = data.rsplit(")", 1)[1].split()
                    if len(fields) < 3 or fields[0] != "Z" or fields[1] != str(our_pid):
                        continue  # not a dead child of ours
                    if int(fields[2]) == pid:
                        continue  # session leader: Popen reaps it
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            except Exception:
                pass  # never let the reaper die
            threading.Event().wait(0.5)

    _REAPER = threading.Thread(target=_reap, name="harness-reaper",
                               daemon=True)
    _REAPER.start()


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
    _ensure_reaper()
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
        # that survived the group kill still holds the pipe open. The
        # reaper thread collects any grandchildren that were re-parented
        # to us while this returns.
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

