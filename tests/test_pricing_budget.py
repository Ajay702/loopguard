from __future__ import annotations

import pytest

from loopguard.budget import Budget, BudgetExceeded, BudgetTracker
from loopguard.models.base import Usage
from loopguard.pricing import PRICING, cost_usd


@pytest.mark.parametrize("model,inp,out", [
    ("claude-opus-5-5", 4.0, 20.0), ("claude-sonnet-5-5", 2.0, 10.0), ("claude-haiku-5-5", 0.10, 0.50)])
def test_pricing_table(model, inp, out):
    assert cost_usd(Usage(input_tokens=1_000_000), model) == pytest.approx(inp)
    assert cost_usd(Usage(output_tokens=1_000_000), model) == pytest.approx(out)


def test_cache_pricing():
    u = Usage(cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    assert cost_usd(u, "claude-opus-5-5") == pytest.approx(4.0 * 1.25 + PRICING["claude-opus-5-5"].cache_read)


def test_unknown_model_fails_safe_to_most_expensive():
    assert cost_usd(Usage(output_tokens=1_000_000), "mystery") == pytest.approx(20.0)


@pytest.mark.parametrize("budget,kw,kind", [
    (Budget(max_steps=2), {"steps": 2}, "steps"),
    (Budget(max_tokens=100), {"usage": Usage(input_tokens=100)}, "tokens"),
    (Budget(max_usd=0.01), {"cost": 0.02}, "usd"),
    (Budget(wall_clock_s=5), {"elapsed_offset": 6}, "wall_clock"),
])
def test_budget_trips(budget, kw, kind):
    t = BudgetTracker(budget, "claude-opus-5-5", **kw)
    with pytest.raises(BudgetExceeded) as e:
        t.check()
    assert e.value.kind == kind


def test_tracker_accumulates():
    t = BudgetTracker(Budget(), "claude-opus-5-5")
    t.record(Usage(input_tokens=1000, output_tokens=100))
    t.record(Usage(input_tokens=1000, output_tokens=100))
    assert t.usage.total_tokens == 2200
    assert t.cost == pytest.approx(2 * (1000 * 4 + 100 * 20) / 1e6)
    t.check()


def test_predictive_usd_guard_stops_before_crossing_the_cap():
    t = BudgetTracker(Budget(max_usd=0.10), "claude-opus-5-5")
    t.record(Usage(output_tokens=3000))  # $0.06
    with pytest.raises(BudgetExceeded, match="would exceed"):
        t.check()  # 0.06 spent + ~0.06 next > 0.10
    lax = BudgetTracker(Budget(max_usd=0.10, predictive_usd=False), "claude-opus-5-5")
    lax.record(Usage(output_tokens=3000))
    lax.check()
