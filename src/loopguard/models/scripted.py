"""Offline backend that replays predefined model turns.

``ScriptedModel`` has the same interface as :class:`~loopguard.models.claude.ClaudeModel`,
so the whole runtime (loop, budgets, retries, checkpoints, guardrails, tracing) can be
exercised deterministically without network access or an API key.

The turn to play is selected by the number of assistant turns already present in the
request's history - *not* by an internal counter. That makes scripts replay-stable: after a
crash, ``resume()`` rebuilds the history from the event log and the scripted model picks up
at exactly the right turn, the same way a real model would respond to the restored context.

A script entry is either a :class:`ModelResponse` or a callable ``(ModelRequest) ->
ModelResponse`` for turns that need to react to tool results (e.g. "if the lookup failed,
apologise").
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

from .base import ContentBlock, ModelRequest, ModelResponse, Usage

ScriptStep = ModelResponse | Callable[[ModelRequest], ModelResponse]


def text(value: str) -> ContentBlock:
    return {"type": "text", "text": value}


def tool_call(name: str, input: dict[str, Any] | None = None, id: str | None = None) -> ContentBlock:
    return {"type": "tool_use", "id": id or "", "name": name, "input": input or {}}


def thinking(signature: str = "sig") -> ContentBlock:
    # Opus 5.5 returns thinking blocks with empty text by default (display="omitted").
    return {"type": "thinking", "thinking": "", "signature": signature}


def turn(*blocks: ContentBlock, stop_reason: str | None = None, **kw: Any) -> ModelResponse:
    content = [dict(b) for b in blocks]
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
    return ModelResponse(content=content, stop_reason=stop_reason, **kw)


def assistant_turns(request: ModelRequest) -> int:
    return sum(1 for m in request.messages if m["role"] == "assistant")


def latest_tool_results(request: ModelRequest) -> list[dict[str, Any]]:
    """Tool result blocks from the most recent user message (empty if none)."""
    if not request.messages or request.messages[-1]["role"] != "user":
        return []
    content = request.messages[-1]["content"]
    if isinstance(content, str):
        return []
    return [b for b in content if b.get("type") == "tool_result"]


def _estimate_tokens(obj: Any) -> int:
    return max(1, len(json.dumps(obj, sort_keys=True, default=str)) // 4)


class ScriptedModel:
    def __init__(
        self,
        script: Sequence[ScriptStep],
        *,
        model: str = "claude-opus-5-5",
        with_thinking: bool = True,
        latency: float = 0.0,
    ) -> None:
        self.script = list(script)
        self.model = model
        self.with_thinking = with_thinking
        self.latency = latency
        self.requests: list[ModelRequest] = []
        self._cache_warm = False

    async def create(self, request: ModelRequest) -> ModelResponse:
        # Snapshot so later appends to the live history don't mutate what we recorded.
        self.requests.append(
            ModelRequest(
                request.system,
                request.tools,
                json.loads(json.dumps(request.messages)),
                request.max_tokens,
            )
        )
        if self.latency:
            await asyncio.sleep(self.latency)
        index = assistant_turns(request)
        if index < len(self.script):
            step = self.script[index]
            response = step(request) if callable(step) else step
        else:
            response = turn(text("(script exhausted)"))

        content = [dict(b) for b in response.content]
        for i, block in enumerate(content):
            if block["type"] == "tool_use" and not block.get("id"):
                block["id"] = f"toolu_{index:02d}_{i}"
        if self.with_thinking and not any(b["type"] == "thinking" for b in content):
            content.insert(0, thinking(signature=f"scripted-sig-{index}"))

        prefix = _estimate_tokens([request.system, request.tools])
        tail = _estimate_tokens(request.messages)
        if self._cache_warm:
            usage = Usage(input_tokens=tail, output_tokens=_estimate_tokens(content),
                          cache_read_input_tokens=prefix)
        else:
            usage = Usage(input_tokens=tail, output_tokens=_estimate_tokens(content),
                          cache_creation_input_tokens=prefix)
            self._cache_warm = True
        if response.usage.total_tokens:  # explicit usage in the script wins
            usage = response.usage

        return ModelResponse(
            content=content,
            stop_reason=response.stop_reason,
            usage=usage,
            model=self.model,
            stop_details=response.stop_details,
            request_id=f"scripted-{index}",
        )
