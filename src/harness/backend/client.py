"""OpenAI-compatible chat/completions client: streaming + one-shot.

Exactly one exception exists here: BackendError means the turn
aborts (session survives). Missing usage stats, odd SSE lines and
the like all degrade instead of raising.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Iterator

import httpx

from .. import config as cfgmod

RETRY_BACKOFF_SEC = 2.0
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 300.0


class BackendError(RuntimeError):
    """Backend failure: the loop turns this into an error event + turn abort."""


@dataclass(frozen=True)
class TextDelta:
    text: str

@dataclass(frozen=True)
class ReasoningDelta:
    text: str

@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None
    name: str | None
    arguments: str  # partial JSON fragment; the loop concatenates per index


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class Done:
    """Final stream event. usage is None when the backend didn't report it."""
    finish_reason: str = ""
    usage: Usage | None = None


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Fallback when the backend doesn't report usage (~4 chars/token)."""
    total = 0
    for m in messages:
        total += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            total += len((tc.get("function") or {}).get("arguments", ""))
    return max(1, total // 4)


class BackendClient:
    def __init__(self, cfg: cfgmod.Config):
        self._base = cfg.backend.base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if cfg.backend.api_key not in ("", "none"):
            headers["Authorization"] = f"Bearer {cfg.backend.api_key}"
        ca_bundle = os.environ.get("HARNESS_CA_BUNDLE")
        verify_off = os.environ.get("HARNESS_TLS_VERIFY", "").strip().lower() \
            in ("0", "false", "no", "off")
        self._client = httpx.Client(
            headers=headers,
            verify=ca_bundle or (not verify_off),
            timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT))
        self._model = cfg.backend.model
        self._temperature = cfg.backend.temperature
        self._max_tokens = cfg.backend.max_tokens
        self._thinking_budget = cfg.backend.thinking_budget

    def _payload(self, messages, tools, max_tokens: int, stream: bool, thinking: bool = True) -> dict:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if thinking and self._thinking_budget > 0:
            payload["thinking_budget_tokens"] = self._thinking_budget
        if tools:  # some backends 400 on an empty tools array
            payload["tools"] = tools
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _send(self, request: httpx.Request, stream: bool = False) -> httpx.Response:
        """Send a request. 5xx/connection: one retry after backoff.
        4xx: raise immediately. Returns an open response; caller closes."""
        last = ""
        for attempt in (1, 2):
            try:
                resp = self._client.send(request, stream=stream)
            except httpx.HTTPError as e:
                last = f"connection error: {type(e).__name__}: {e}"
                if attempt == 2:
                    raise BackendError(last) from None
                time.sleep(RETRY_BACKOFF_SEC)
                continue
            if resp.status_code >= 500:
                body = resp.read()[:200].decode("utf-8", "replace")
                resp.close()
                last = f"HTTP {resp.status_code}: {body}"
                if attempt == 2:
                    raise BackendError(last) from None
                time.sleep(RETRY_BACKOFF_SEC)
                continue
            if resp.status_code >= 400:
                body = resp.read()[:200].decode("utf-8", "replace")
                resp.close()
                raise BackendError(f"HTTP {resp.status_code}: {body}") from None
            return resp
        raise BackendError(last)


    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None, cancel: threading.Event | None = None
                    ) -> Iterator[TextDelta | ToolCallDelta | Done]:
        """Stream one completion. Stop iterating (Esc) and the request is
        cancelled by resp.close() in the finally block; discard the partial."""
        payload = self._payload(messages, tools, self._max_tokens, stream=True)
        request = self._client.build_request(
            "POST", f"{self._base}/chat/completions", json=payload)
        resp = self._send(request, stream=True)
        finish_reason = ""
        usage: Usage | None = None
        try:
            for raw in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    break
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue  # keep going; the loop owns turn-level parsing
                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage = Usage(u.get("prompt_tokens", 0),
                                  u.get("completion_tokens", 0))
                choices = chunk.get("choices") or []
                if not choices:
                    continue  # usage-only chunk
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    yield ReasoningDelta(delta["reasoning_content"])
                if delta.get("content"):
                    yield TextDelta(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    yield ToolCallDelta(
                        index=tc.get("index", 0),
                        id=tc.get("id"),
                        name=fn.get("name") or None,
                        arguments=fn.get("arguments") or "")
            yield Done(finish_reason, usage)
        except httpx.HTTPError as e:
            raise BackendError(f"stream interrupted: {type(e).__name__}: {e}") from None
        finally:
            resp.close()

    def complete(self, messages: list[dict], max_tokens: int) -> str:
        """One-shot completion (compaction pass). Returns the content string."""
        payload = self._payload(messages, None, max_tokens, stream=False, thinking=False)
        request = self._client.build_request(
            "POST", f"{self._base}/chat/completions", json=payload)
        resp = self._send(request, stream=False)
        try:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise BackendError("backend returned no choices")
            return (choices[0].get("message") or {}).get("content") or ""
        except (json.JSONDecodeError, httpx.DecodingError):
            raise BackendError("backend returned non-JSON response") from None
        finally:
            resp.close()

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    def list_models(self) -> list[str]:
        """GET /models: ids the backend can serve (for the TUI picker)."""
        request = self._client.build_request("GET", f"{self._base}/models")
        resp = self._send(request)
        try:
            data = resp.json()
        except (json.JSONDecodeError, httpx.DecodingError):
            raise BackendError("backend returned non-JSON response") from None
        finally:
            resp.close()
        ids: list[str] = []
        for item in data.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        return ids

    def close(self) -> None:
        self._client.close()


