from __future__ import annotations

import asyncio
import json

from loopguard.trace_html import render_html
from loopguard.tracing import Tracer, export_trace, load_trace


async def test_spans_nest_across_concurrent_tasks(tmp_path):
    tr = Tracer("run_x", sink=tmp_path / "spans.jsonl")
    async with tr.span("run", "run") as run:
        async with tr.span("step 1", "step") as step:
            async def tool(i):
                async with tr.span(f"tool {i}", "tool_call") as s:
                    await asyncio.sleep(0.01)
                    return s
            spans = await asyncio.gather(*(tool(i) for i in range(3)))
    assert all(s.parent_id == step.span_id for s in spans)
    assert step.parent_id == run.span_id and run.parent_id is None
    assert len((tmp_path / "spans.jsonl").read_text().splitlines()) == 5


async def test_sink_is_redacted_and_errors_recorded(tmp_path):
    tr = Tracer("run_x", sink=tmp_path / "spans.jsonl")
    try:
        async with tr.span("tool", "tool_call", input={"to": "dana@gmail.com"}):
            raise RuntimeError("boom for 4242 4242 4242 4242")
    except RuntimeError:
        pass
    rec = json.loads((tmp_path / "spans.jsonl").read_text())
    assert rec["status"] == "error"
    assert rec["attributes"]["input"]["to"] == "[EMAIL]"
    assert "4242 4242" not in json.dumps(rec)


async def test_export_summary_and_html(tmp_path):
    tr = Tracer("run_x", sink=tmp_path / "spans.jsonl")
    async with tr.span("run", "run"):
        async with tr.span("model_call", "model_call") as m:
            m.set(total_tokens=100, cost_usd=0.01)
        async with tr.span("tool t", "tool_call") as t:
            t.status = "blocked"
            t.set(injection_flagged=True, payload="</script><script>alert(1)</script>")
    doc = load_trace(tmp_path / "spans.jsonl", "run_x", [{"type": "run_finished", "status": "completed"}])
    assert doc["summary"]["tokens"] == 100 and doc["summary"]["blocked_actions"] == 1
    assert doc["summary"]["injections_flagged"] == 1 and doc["summary"]["status"] == "completed"
    html = render_html(doc)
    assert "<script>alert(1)" not in html  # embedded JSON cannot break out of its script tag
    assert "http://" not in html and "https://" not in html  # fully self-contained
    j, h = export_trace(tmp_path, "run_x")
    assert j.exists() and h.read_text().startswith("<!doctype html>")
