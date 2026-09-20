"""OpenAI-compatible backend access (v1/chat/completions)."""

from .client import (BackendClient, BackendError, Done, ReasoningDelta, ToolCallDelta,
                     TextDelta, Usage)

__all__ = ["BackendClient", "BackendError", "Done", "ReasoningDelta",  "ToolCallDelta",
           "TextDelta", "Usage"]

