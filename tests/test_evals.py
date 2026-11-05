from __future__ import annotations

from loopguard.evals import SCENARIOS, run_suite, summarize, to_markdown


async def test_offline_suite_passes(tmp_path):
    results = await run_suite(store_root=tmp_path)
    failing = {r.name: [c for c in r.checks if not c.passed] or r.error for r in results if r.status != "pass"}
    assert not failing, failing
    s = summarize(results)
    assert s["scenarios_run"] == len(SCENARIOS) >= 10
    assert "PASS" in to_markdown(results)


async def test_every_scenario_has_global_pii_check(tmp_path):
    results = await run_suite(store_root=tmp_path, only=["refund_happy_path"])
    assert results[0].checks[-1].name.startswith("no PII")


async def test_pii_check_detects_leaks(tmp_path, monkeypatch):
    # Disable redaction in the tracer: the global check must now FAIL. Guards against a
    # check that can never fail.
    from loopguard import tracing

    monkeypatch.setattr(tracing.Redactor, "redact", lambda self, v: v)
    results = await run_suite(store_root=tmp_path, only=["no_pii_in_observability"])
    pii = results[0].checks[-1]
    assert not pii.passed, pii.detail
