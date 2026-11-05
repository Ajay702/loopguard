"""Budgets: hard limits on steps, tokens, dollars and wall-clock time.

Budgets are checked *before* every model call (so we never start a call we already know
we cannot afford) and *after* every response (so a trip is reported with the step that
caused it). When a budget trips, the agent stops gracefully and returns a partial result -
it does not raise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .models.base import Usage
from .pricing import cost_usd


@dataclass(frozen=True)
class Budget:
    max_steps: int | None = 20
    max_tokens: int | None = 500_000
    max_usd: float | None = 2.00
    wall_clock_s: float | None = 300.0
    tool_timeout_s: float = 30.0  # default per-tool timeout when a tool declares none
    max_output_tokens: int = 16_000  # per-response max_tokens (non-streaming safe default)
    max_output_tokens_ceiling: int = 64_000  # cap when escalating after a max_tokens stop
    max_pause_continuations: int = 5
    # Predictive USD guard: refuse to start a step if spent + (cost of the previous step)
    # would cross max_usd. Contexts grow monotonically, so the last step is a conservative
    # lower bound on the next one - this keeps spend under the cap instead of one step over.
    predictive_usd: bool = True


class BudgetExceeded(Exception):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind} budget exceeded: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class BudgetTracker:
    budget: Budget
    model: str
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    elapsed_offset: float = 0.0  # wall-clock already spent in earlier (pre-crash) sessions
    last_step_cost: float = 0.0

    def record(self, usage: Usage, model: str | None = None) -> float:
        c = cost_usd(usage, model or self.model)
        self.usage = self.usage + usage
        self.cost += c
        self.last_step_cost = c
        return c

    @property
    def elapsed(self) -> float:
        return self.elapsed_offset + (time.monotonic() - self.started_at)

    def remaining_wall_clock(self) -> float | None:
        if self.budget.wall_clock_s is None:
            return None
        return max(0.0, self.budget.wall_clock_s - self.elapsed)

    def check(self) -> None:
        b = self.budget
        if b.max_steps is not None and self.steps >= b.max_steps:
            raise BudgetExceeded("steps", f"{self.steps}/{b.max_steps} steps used")
        if b.max_tokens is not None and self.usage.total_tokens >= b.max_tokens:
            raise BudgetExceeded("tokens", f"{self.usage.total_tokens}/{b.max_tokens} tokens")
        if b.max_usd is not None and self.cost >= b.max_usd:
            raise BudgetExceeded("usd", f"${self.cost:.4f}/${b.max_usd:.4f}")
        if (b.max_usd is not None and b.predictive_usd
                and self.cost + self.last_step_cost > b.max_usd):
            raise BudgetExceeded(
                "usd", f"${self.cost:.4f} spent; next step (~${self.last_step_cost:.4f}) "
                       f"would exceed ${b.max_usd:.4f}")
        if b.wall_clock_s is not None and self.elapsed >= b.wall_clock_s:
            raise BudgetExceeded("wall_clock", f"{self.elapsed:.1f}s/{b.wall_clock_s:.1f}s")

    def snapshot(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "usage": self.usage.to_dict(),
            "total_tokens": self.usage.total_tokens,
            "cost_usd": round(self.cost, 6),
            "elapsed_s": round(self.elapsed, 3),
        }
