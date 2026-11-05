from __future__ import annotations

import json
import os

import pytest

from loopguard import Agent, ScriptedModel
from loopguard.checkpoint import EventLog, RunState, SimulatedCrash, read_jsonl
from loopguard.evals.scenarios import refund_script
from loopguard.examples.support import (
    DEMO_TASK,
    SYSTEM_PROMPT,
    SupportBackend,
    build_policy,
    build_tools,
)


def test_event_log_appends_with_sequence_and_private_mode(tmp_path):
    p = tmp_path / "r" / "events.jsonl"
    log = EventLog(p, "r")
    log.append("run_started", task="t", system="s", model="m")
    log.append("x")
    log.close()
    events = list(read_jsonl(p))
    assert [e["seq"] for e in events] == [1, 2]
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"
    log2 = EventLog(p, "r")  # new session continues the sequence
    assert log2.append("y")["seq"] == 3 and log2.session == 1


def test_torn_final_line_is_ignored(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(json.dumps({"seq": 1}) + "\n" + '{"seq": 2, "type": "assis')
    assert [e["seq"] for e in read_jsonl(p)] == [1]


def test_crash_injection_kills_the_log(tmp_path):
    log = EventLog(tmp_path / "e.jsonl", "r", crash_after_event=1)
    with pytest.raises(SimulatedCrash):
        log.append("a")
    with pytest.raises(SimulatedCrash):
        log.append("b")
    assert len(list(read_jsonl(tmp_path / "e.jsonl"))) == 1


def _agent(store, backend, **kw):
    return Agent(model=ScriptedModel(refund_script()), tools=build_tools(backend), system=SYSTEM_PROMPT,
                 policy=build_policy(backend), store=store, fsync=False, **kw)


async def test_state_reconstruction_matches_live_history(store):
    r = await _agent(store, SupportBackend()).run(DEMO_TASK)
    st = RunState.from_events(r.run_id, store.load_events(r.run_id))
    assert st.messages == r.messages
    assert st.finished["status"] == "completed"
    assert st.steps == 4 and st.usage.total_tokens == r.total_tokens
    assert st.cost == pytest.approx(r.cost_usd)


async def _reference(store):
    return await _agent(store, SupportBackend()).run(DEMO_TASK, run_id="reference")


@pytest.mark.parametrize("mode", ["after", "before"])
async def test_resume_from_every_crash_point(store, mode):
    ref = await _reference(store)
    n = len(store.load_events(ref.run_id))
    for k in range(2, n):
        be = SupportBackend()
        kw = {f"crash_{mode}_event": k}
        with pytest.raises(SimulatedCrash):
            await _agent(store, be, **kw).run(DEMO_TASK, run_id=f"{mode}{k}")
        events = store.load_events(f"{mode}{k}")
        logged = {e["tool_use_id"] for e in events if e["type"] == "tool_result"}
        before = dict(be.calls)
        r = await _agent(store, be).resume(f"{mode}{k}")
        assert r.status == "completed", (mode, k)
        assert r.messages == ref.messages, (mode, k)
        assert len(be.refunds) == 1 and len(be.outbox) == 1, (mode, k)
        if "toolu_01_0" in logged:  # the refund result was durable -> never re-run
            assert be.calls["issue_refund"] == before.get("issue_refund", 0)


async def test_resume_of_finished_run_is_a_noop(store):
    be = SupportBackend()
    r = await _agent(store, be).run(DEMO_TASK)
    again = await _agent(store, be).resume(r.run_id)
    assert again.status == "completed" and len(be.refunds) == 1
    assert again.final_text == r.final_text


async def test_budget_stopped_run_can_resume_with_bigger_budget(store):
    from loopguard import Budget

    be = SupportBackend()
    r = await _agent(store, be, budget=Budget(max_steps=2)).run(DEMO_TASK)
    assert r.status == "budget_exceeded"
    r2 = await _agent(store, be, budget=Budget(max_steps=10)).resume(r.run_id)
    assert r2.status == "completed" and len(be.refunds) == 1 and r2.steps == 4


async def test_event_log_is_hash_chained_and_tamper_evident(store):
    from loopguard.checkpoint import verify_chain

    r = await _agent(store, SupportBackend()).run(DEMO_TASK)
    events = store.load_events(r.run_id)
    assert verify_chain(events) is None
    tampered = [dict(e) for e in events]
    tampered[3]["block"] = {"type": "tool_result", "tool_use_id": "x", "content": "edited"}
    assert verify_chain(tampered) == tampered[3]["seq"]
    assert verify_chain(events[:2] + events[3:]) == events[3]["seq"]  # deleted line
