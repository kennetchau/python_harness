"""First-run seeding of /state and the workspace repo. Idempotent."""

from __future__ import annotations

from pathlib import Path

from .. import config as cfgmod
from .gitstore import ensure_repo


def seed(cfg: cfgmod.Config) -> bool:
    """Seed /state and the workspace. Returns True if config.toml was created."""
    cfgmod.STATE_DIR.mkdir(parents=True, exist_ok=True)
    created = not cfgmod.CONFIG_PATH.exists()
    if created:
        cfgmod.CONFIG_PATH.write_text(cfgmod.DEFAULT_CONFIG_TEMPLATE)
        print(f"[seed] wrote default config: {cfgmod.CONFIG_PATH}")
    (cfgmod.STATE_DIR / "agents").mkdir(exist_ok=True)
    (cfgmod.STATE_DIR / "sessions").mkdir(exist_ok=True)
    ensure_repo(cfgmod.STATE_DIR, "harness: initial state")
    if cfg.workspace.auto_git:
        ensure_repo(Path(cfg.workspace.path), "harness: initial import")
    return created

