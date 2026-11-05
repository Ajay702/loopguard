from __future__ import annotations

import random

import anthropic
import pytest

from loopguard.faults import api_error
from loopguard.resilience import (
    CircuitBreaker,
    RetryPolicy,
    is_retryable_api_error,
    is_retryable_tool_error,
    retry_async,
)
from loopguard.tools import ToolError, TransientToolError


@pytest.mark.parametrize("kind,retryable", [
    ("429", True), ("500", True), ("529", True), ("connection", True), ("timeout", True),
    ("400", False), ("401", False)])
def test_retry_classification_uses_typed_sdk_errors(kind, retryable):
    assert is_retryable_api_error(api_error(kind)) is retryable


def test_tool_retry_classification():
    assert is_retryable_tool_error(TransientToolError("x"))
    assert is_retryable_tool_error(TimeoutError())
    assert not is_retryable_tool_error(ToolError("order not found"))
    assert not is_retryable_tool_error(ValueError())


def test_backoff_is_exponential_capped_and_jittered():
    p = RetryPolicy(base_delay=1, max_delay=5, jitter=False)
    assert [p.delay(n) for n in range(1, 6)] == [1, 2, 4, 5, 5]
    jp = RetryPolicy(base_delay=1, max_delay=5)
    rng = random.Random(0)
    samples = [jp.delay(3, rng) for _ in range(200)]
    assert all(0 <= s <= 4 for s in samples) and len(set(samples)) > 100


async def test_retry_until_success_records_delays():
    calls, delays = [], []

    async def fn(attempt):
        calls.append(attempt)
        if attempt < 3:
            raise api_error("500")
        return "ok"

    async def sleep(d):
        delays.append(d)

    out = await retry_async(fn, policy=RetryPolicy(max_attempts=5, jitter=False, base_delay=0.1),
                            retryable=is_retryable_api_error, sleep=sleep)
    assert out == "ok" and calls == [1, 2, 3] and delays == [0.1, 0.2]


async def test_non_retryable_raises_immediately():
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        raise api_error("400")

    with pytest.raises(Exception, match="invalid request"):
        await retry_async(fn, policy=RetryPolicy(), retryable=is_retryable_api_error)
    assert calls == [1]


async def test_attempts_exhausted_reraises_last_error():
    async def fn(attempt):
        raise api_error("529")

    async def sleep(_):
        pass

    with pytest.raises(Exception, match="overloaded"):
        await retry_async(fn, policy=RetryPolicy(max_attempts=3), retryable=is_retryable_api_error, sleep=sleep)


async def test_retry_after_header_is_honoured():
    delays = []
    err = api_error("429")
    err.response.headers["retry-after"] = "3"

    async def fn(attempt):
        if attempt == 1:
            raise err
        return 1

    async def sleep(d):
        delays.append(d)

    await retry_async(fn, policy=RetryPolicy(base_delay=0.1, max_delay=10), retryable=is_retryable_api_error, sleep=sleep)
    assert delays == [3.0]


async def test_deadline_stops_retrying():
    import time

    async def fn(attempt):
        raise api_error("500")

    with pytest.raises(anthropic.InternalServerError):
        await retry_async(fn, policy=RetryPolicy(base_delay=10, jitter=False, max_delay=10),
                          retryable=is_retryable_api_error, deadline=time.monotonic() + 1)


def test_circuit_breaker_state_machine():
    now = [0.0]
    b = CircuitBreaker(failure_threshold=2, reset_timeout=10, clock=lambda: now[0])
    assert b.allow()
    b.record_failure()
    assert b.state == "closed"
    b.record_failure()
    assert b.state == "open" and not b.allow()
    now[0] = 11
    assert b.allow() and b.state == "half_open"
    b.record_failure()  # probe fails -> open again
    assert b.state == "open" and not b.allow()
    now[0] = 22
    assert b.allow()
    b.record_success()
    assert b.state == "closed" and b.failures == 0
    assert b.transitions == ["closed->open", "open->half_open", "half_open->open",
                             "open->half_open", "half_open->closed"]
