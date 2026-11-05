"""Retries with exponential backoff + full jitter, and per-tool circuit breakers.

Retry classification uses the SDK's *typed* exceptions - never string matching:

* retryable: ``RateLimitError`` (429), ``InternalServerError`` and its subclasses
  (500, 503, 504, 529 overloaded), ``APIConnectionError`` (incl. ``APITimeoutError``),
  plus 408/409 which the API documents as transient.
* not retryable: every other ``APIStatusError`` - 400 invalid request, 401, 403, 404,
  413 ... Retrying those only burns budget and hides bugs.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

import anthropic

from .tools import TransientToolError

T = TypeVar("T")


def is_retryable_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError)):
        return True
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (408, 409) or exc.status_code >= 500
    return False


def is_retryable_tool_error(exc: BaseException) -> bool:
    return isinstance(exc, (TransientToolError, TimeoutError, asyncio.TimeoutError))


def retry_after_seconds(exc: BaseException) -> float | None:
    """Honour the server's ``retry-after`` header on 429/529 when present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        value = response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 20.0
    jitter: bool = True

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Full-jitter backoff: uniform(0, min(max_delay, base * 2**(attempt-1)))."""
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        if not self.jitter:
            return ceiling
        return (rng or random).uniform(0, ceiling)


async def retry_async(
    fn: Callable[[int], Awaitable[T]],
    *,
    policy: RetryPolicy,
    retryable: Callable[[BaseException], bool],
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    deadline: float | None = None,
) -> T:
    """Call ``fn(attempt)`` until it succeeds, a non-retryable error occurs, attempts run
    out, or sleeping would cross ``deadline`` (a ``time.monotonic()`` value)."""
    attempt = 1
    while True:
        try:
            return await fn(attempt)
        except Exception as exc:
            if attempt >= policy.max_attempts or not retryable(exc):
                raise
            delay = policy.delay(attempt)
            hinted = retry_after_seconds(exc)
            if hinted is not None:
                delay = min(max(delay, hinted), policy.max_delay)
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise
            if on_retry:
                on_retry(attempt, exc, delay)
            await sleep(delay)
            attempt += 1


class CircuitOpenError(Exception):
    pass


@dataclass
class CircuitBreaker:
    """Classic three-state breaker.

    closed --(N consecutive failures)--> open --(cooldown elapsed)--> half_open
    half_open --(probe succeeds)--> closed ; half_open --(probe fails)--> open
    """

    failure_threshold: int = 3
    reset_timeout: float = 30.0
    clock: Callable[[], float] = time.monotonic
    state: str = "closed"
    failures: int = 0
    opened_at: float = 0.0
    transitions: list[str] = field(default_factory=list)

    def _set(self, state: str) -> None:
        if state != self.state:
            self.transitions.append(f"{self.state}->{state}")
            self.state = state

    def allow(self) -> bool:
        if self.state == "open":
            if self.clock() - self.opened_at >= self.reset_timeout:
                self._set("half_open")
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self._set("closed")

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.opened_at = self.clock()
            self._set("open")
