"""Live backend: the Claude Messages API via the official ``anthropic`` Python SDK.

Request shape decisions (see docs/ARCHITECTURE.md for the reasoning):

* ``output_config={"effort": ...}`` is always sent explicitly. On claude-opus-5-5 the
  default is ``medium``; being explicit keeps behaviour stable across model upgrades.
* No ``thinking`` parameter, no ``temperature``, no assistant prefill and no forced
  ``tool_choice``: thinking is always on for this model (it cannot be disabled), sampling
  parameters and prefill are rejected, and ``any``/``tool`` tool choice returns a 400.
  ``tool_choice`` stays ``auto``; tools use ``strict: true`` for schema-valid arguments.
* Prompt caching: one explicit breakpoint on the static system prompt (which, because the
  render order is tools -> system -> messages, also covers the tool definitions) plus
  top-level automatic caching for the growing conversation tail.
* SDK-level retries are disabled (``max_retries=0``): loopguard owns retries so that
  every attempt is visible in traces and counted against budgets.
* Refusals: optionally enable server-side refusal fallbacks (beta) so a safety-classifier
  decline is retried on a fallback model inside the same call. The runtime still handles
  ``stop_reason == "refusal"`` when the whole chain declines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import ModelRequest, ModelResponse, Usage

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def credentials_available() -> bool:
    """Best-effort check for credentials the SDK would resolve (env vars or an `ant` profile)."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if os.environ.get("ANTHROPIC_FEDERATION_RULE_ID") and (
        os.environ.get("ANTHROPIC_IDENTITY_TOKEN") or os.environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE")
    ):
        return True
    profile_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "anthropic"
    return profile_dir.exists() and any(profile_dir.iterdir())


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Serialise an SDK content block without altering it (signatures etc. preserved)."""
    if isinstance(block, dict):
        return block
    return block.model_dump(mode="json", exclude_none=True)


class ClaudeModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "medium",
        refusal_fallback: bool = True,
        client: Any | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.effort = effort
        self.refusal_fallback = refusal_fallback
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(max_retries=0, timeout=timeout)
        self._client = client

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": request.messages,
            "output_config": {"effort": self.effort},
            # Automatic caching for the conversation tail; composes with the explicit
            # system breakpoint above.
            "cache_control": {"type": "ephemeral"},
        }
        if request.tools:
            params["tools"] = request.tools
            params["tool_choice"] = {"type": "auto"}
        return params

    async def create(self, request: ModelRequest) -> ModelResponse:
        params = self.build_params(request)
        if self.refusal_fallback:
            msg = await self._client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            msg = await self._client.messages.create(**params)

        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )
        stop_details = getattr(msg, "stop_details", None)
        return ModelResponse(
            content=[_block_to_dict(b) for b in msg.content],
            stop_reason=msg.stop_reason or "end_turn",
            usage=usage,
            model=msg.model,
            stop_details=_block_to_dict(stop_details) if stop_details is not None else None,
            request_id=getattr(msg, "_request_id", None),
        )
