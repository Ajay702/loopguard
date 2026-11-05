"""Token pricing (USD per million tokens) and cost accounting.

Cache writes (5-minute TTL) are billed at 1.25x the input rate; cache reads at the
model's cache-read rate. Keep this table in sync with the published pricing page.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models.base import Usage


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICING: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01),
}


def price_for(model: str) -> Price:
    if model in PRICING:
        return PRICING[model]
    # Unknown model: fail safe by assuming the most expensive known rate so budgets
    # trip early rather than late.
    return max(PRICING.values(), key=lambda p: p.output)


def cost_usd(usage: Usage, model: str) -> float:
    p = price_for(model)
    return (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + usage.cache_creation_input_tokens * p.cache_write
        + usage.cache_read_input_tokens * p.cache_read
    ) / 1_000_000
