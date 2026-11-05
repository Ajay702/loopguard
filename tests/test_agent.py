from __future__ import annotations

import asyncio
import json

import pytest

from loopguard import Agent, Budget, ScriptedModel, tool
from loopguard.models import ModelResponse, text, tool_call, turn
from loopguard.models.scripted import latest_tool_results
from loopguard.safety import Policy, ask, deny
from loopguard.safety.injection import UNTRUSTED_CONTENT_POLICY
from loopguard.tools import ToolError


@tool(idempotent=True)
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
async def slow(seconds: float) -> str:
    """Sleep then return."""
    await asyncio.sleep(seconds)
    return "slept"


@tool
async def broken() -> str:
    """Always fails with a business error."""
    raise ToolError("nope")


def agent(store, script, **kw):
    return Agent(model=ScriptedModel(script), tools=[add, slow, broken], system="You add numbers.",
                 store=store, fsync=False, **kw)


async def test_parallel_calls_run_concurrently_and_return_in_one_message(store):
    a = agent(store, [
        turn(tool_call("slow", {"seconds": 0.2}), tool_call("slow", {"seconds": 0.2}),
             tool_call("add", {"a": 1, "b": 2})),
        turn(text("done")),
    ])
    import time
    t0 = time.monotonic()
    r = await a.run("go")
    assert time.monotonic() - t0 < 0.35
    results_msg = r.messages[2]
    assert results_msg["role"] == "user"
    ids = [b["id"] for b in r.messages[1]["content"] if b["type"] == "tool_use"]
    assert [b["tool_use_id"] for b in results_msg["content"]] == ids
    assert "3" in results_msg["content"][2]["content"]


async def test_assistant_content_passed_back_unchanged(store):
    model = ScriptedModel([turn(tool_call("add", {"a": 1, "b": 1})), turn(text("2"))])
    a = Agent(model=model, tools=[add], system="s", store=store, fsync=False)
    await a.run("1+1")
    second = model.requests[1].messages
    assert second[1]["role"] == "assistant"
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "scripted-sig-0"}
    # history is append-only: request N is a strict prefix of request N+1
    assert model.requests[0].messages == second[: len(model.requests[0].messages)]


async def test_request_shape(store):
    model = ScriptedModel([turn(text("hi"))])
    await Agent(model=model, tools=[add], system="Base prompt.", store=store, fsync=False).run("x")
    req = model.requests[0]
    assert req.system.startswith("Base prompt.") and UNTRUSTED_CONTENT_POLICY in req.system
    assert req.tools[0]["strict"] is True
    assert req.tools[0]["input_schema"]["additionalProperties"] is False


async def test_failures_return_is_error_results(store):
    a = agent(store, [
        turn(tool_call("broken"), tool_call("nope_tool"), tool_call("add", {"a": "x", "b": 1})),
        lambda req: turn(text(json.dumps([b.get("is_error", False) for b in latest_tool_results(req)]))),
    ])
    r = await a.run("go")
    assert r.status == "completed" and r.final_text == "[true, true, true]"
    by_id = {x["tool_use_id"]: x["status"] for x in r.actions}  # completion order may vary
    assert [by_id[b["tool_use_id"]] for b in r.messages[2]["content"]] == ["error", "unknown_tool", "invalid_input"]


async def test_policy_deny_and_ask(store):
    a = agent(store, [turn(tool_call("add", {"a": 1, "b": 99}), tool_call("slow", {"seconds": 0})),
                      turn(text("ok"))],
              policy=Policy([deny("add", when=lambda x: x["b"] > 50, reason="too big"),
                             ask("slow", reason="slow is gated")]),
              approver=lambda req: req.tool == "slow")
    r = await a.run("go")
    assert sorted(x["status"] for x in r.actions) == ["blocked", "ok"]
    assert "too big" in r.blocked[0]["reason"]


async def test_async_approver_and_no_approver_fails_closed(store):
    async def approver(req):
        return False

    for kw in ({"approver": approver}, {}):
        a = agent(store, [turn(tool_call("add", {"a": 1, "b": 1})), turn(text("ok"))],
                  policy=Policy([ask("add")]), **kw)
        r = await a.run("go")
        assert r.actions[0]["status"] == "blocked"


@pytest.mark.parametrize("stop,status", [("refusal", "refused"), ("max_tokens", "max_tokens")])
async def test_terminal_stop_reasons(store, stop, status):
    resp = ModelResponse(content=[text("partial")], stop_reason=stop,
                         stop_details={"category": "bio", "explanation": "x"} if stop == "refusal" else None)
    a = agent(store, [resp], budget=Budget(max_output_tokens=64_000, max_output_tokens_ceiling=64_000))
    r = await a.run("go")
    assert r.status == status and r.partial


async def test_max_tokens_escalation_bounded(store):
    calls = []

    def always_truncated(req):
        calls.append(req.max_tokens)
        return turn(text("cut"), stop_reason="max_tokens")

    a = agent(store, [always_truncated], budget=Budget(max_output_tokens=1000, max_output_tokens_ceiling=4000))
    r = await a.run("go")
    assert calls == [1000, 2000, 4000] and r.status == "max_tokens"


async def test_pause_turn_limit(store):
    a = agent(store, [turn(text("..."), stop_reason="pause_turn")] * 10,
              budget=Budget(max_pause_continuations=2))
    r = await a.run("go")
    assert r.status == "pause_limit"


async def test_budget_stop_returns_partial_result_and_skips_unpaid_side_effects(store):
    calls = []

    @tool
    async def side_effect() -> str:
        """Do something."""
        calls.append(1)
        return "done"

    a = Agent(model=ScriptedModel([turn(tool_call("side_effect"))] * 5), tools=[side_effect],
              system="s", store=store, fsync=False, budget=Budget(max_steps=2))
    r = await a.run("go")
    assert r.status == "budget_exceeded" and "partial result" in r.final_text
    assert r.steps == 2 and len(calls) == 1  # the 2nd turn's tool call is not executed


async def test_per_tool_timeout(store):
    slow_tool = slow
    import dataclasses
    fast_timeout = dataclasses.replace(slow_tool, timeout=0.05)
    a = Agent(model=ScriptedModel([turn(tool_call("slow", {"seconds": 5})), turn(text("ok"))]),
              tools=[fast_timeout], system="s", store=store, fsync=False)
    r = await a.run("go")
    assert r.actions[0]["status"] == "timeout" and r.status == "completed"


async def test_run_id_collision_rejected(store):
    a = agent(store, [turn(text("hi"))])
    await a.run("x", run_id="fixed")
    with pytest.raises(ValueError):
        await a.run("x", run_id="fixed")
