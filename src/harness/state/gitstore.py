"""Git helpers shared by seeding and per-turn commits.

Failures print a warning to stderr and return a falsy value;
they never raise, so a git hiccup can't kill a session.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_git(args: list[str], cwd: Path) -> bool:
    """Run a git command. Prints stderr on failure. Returns True on success."""
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        print(f"[git] {' '.join(args)} failed in {cwd}: {detail}", file=sys.stderr)
        return False
    return True


def ensure_repo(path: Path, message: str) -> None:
    """Create dir, git init if needed, initial commit if there is content."""
    path.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        run_git(["init"], path)
        run_git(["config", "user.name", "harness"], path)
        run_git(["config", "user.email", "harness@localhost"], path)
    commit_all(path, message)


def commit_all(path: Path, message: str) -> str | None:
    """git add -A + commit. Returns the short hash, or None when nothing changed."""
    if not run_git(["add", "-A"], path):
        return None
    status = subprocess.run(["git", "status", "--porcelain"],
                            cwd=path, capture_output=True, text=True)
    if not status.stdout.strip():
        return None
    if not run_git(["commit", "-m", message], path):
        return None
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=path, capture_output=True, text=True)
    return proc.stdout.strip()

