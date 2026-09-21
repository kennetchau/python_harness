"""Agent layer: JSONL session store, context compaction, turn state machine."""

from .compaction import compact_once, ensure_budget
from .loop import DEFAULT_AGENT, SYSTEM_PROMPTS, AgentLoop, TurnResult, make_preview
from .session import SUMMARY_MARKER, Session, new_session_id, summary_message

__all__ = [
    "AgentLoop", "TurnResult", "Session", "new_session_id", "summary_message",
    "SUMMARY_MARKER", "compact_once", "ensure_budget",
    "DEFAULT_AGENT", "SYSTEM_PROMPTS", "make_preview",
]

