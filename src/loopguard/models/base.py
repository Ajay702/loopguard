"""Model backend interface shared by the live Claude backend and the offline scripted backend.

Conversation content is kept as plain JSON-serialisable dicts in the exact shape the
Messages API uses. That gives us three properties for free:

* the event log can store assistant turns byte-for-byte and replay them on resume,
* assistant content blocks (including ``thinking`` blocks and their signatures) are
  passed back to the API unchanged, which the API requires,
* the scripted backend and the live backend are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

ContentBlock = dict[str, Any]
Message = dict[str, Any]

STOP_REASONS = ("end_turn", "tool_use", "max_tokens", "pause_turn", "refusal", "stop_sequence")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Usage:
        d = d or {}
        return cls(
            int(d.get("input_tokens") or 0),
            int(d.get("output_tokens") or 0),
            int(d.get("cache_creation_input_tokens") or 0),
            int(d.get("cache_read_input_tokens") or 0),
        )


@dataclass
class ModelRequest:
    """Everything a backend needs for one call. Stable parts first (cache-friendly)."""

    system: str
    tools: list[dict[str, Any]]
    messages: list[Message]
    max_tokens: int


@dataclass
class ModelResponse:
    content: list[ContentBlock]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    stop_details: dict[str, Any] | None = None
    request_id: str | None = None

    @property
    def tool_uses(self) -> list[ContentBlock]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "stop_details": self.stop_details,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelResponse:
        return cls(
            content=list(d["content"]),
            stop_reason=d["stop_reason"],
            usage=Usage.from_dict(d.get("usage")),
            model=d.get("model", ""),
            stop_details=d.get("stop_details"),
            request_id=d.get("request_id"),
        )


@runtime_checkable
class ModelBackend(Protocol):
    """Anything that can turn a :class:`ModelRequest` into a :class:`ModelResponse`."""

    model: str

    async def create(self, request: ModelRequest) -> ModelResponse: ...
