"""Reproducible benchmarks for loopguard (offline: ScriptedModel, no API key).

    uv run python benchmarks/bench.py

1. Overhead   - wall time loopguard adds per agent step on top of a bare loop
                (model call + tool call), with and without fsync'd checkpoints.
2. Durability - a 20-tool-call run that charges a card on every step is crashed
                at every event boundary, then resumed. Count duplicate charges.
3. Budgets    - a runaway agent with a USD cap. How close to the cap does it stop,
                and does it ever go over?
"""
from __future__ import annotations

import asyncio
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from loopguard import Agent, Budget, RunStore, ScriptedModel, SimulatedCrash, ToolContext, tool
from loopguard.models.scripted import text, tool_call, turn

STEPS = 20


def script(n: int = STEPS):
    return [turn(text(f"step {i}"), tool_call("charge", {"order_id": f"A{i:04d}", "cents": 100 + i}))
            for i in range(n)] + [turn(text("All orders charged."))]


class Ledger:
    """A payment API that deduplicates on idempotency key, like Stripe does."""
    def __init__(self):
        self.executions = 0
        self.charges: dict[str, int] = {}

    def tool(self):
        @tool(idempotent=False, timeout=5)
        async def charge(order_id: str, cents: int, ctx: ToolContext) -> dict:
            """Charge an order.

            Args:
                order_id: Order id.
                cents: Amount in cents.
            """
            self.executions += 1
            self.charges.setdefault(ctx.idempotency_key, cents)
            return {"ok": True, "order_id": order_id}
        return charge


def agent(store, ledger, fsync=True, budget=None, **kw):
    return Agent(model=ScriptedModel(kw.pop("script", script())), tools=[ledger.tool()],
                 system="You charge orders.", store=store, fsync=fsync,
                 budget=budget or Budget(max_steps=200, max_usd=100.0), **kw)


async def bare_loop(n: int = STEPS):
    """The minimum any agent loop does: call the model, run the tool, append history."""
    from loopguard.models.base import ModelRequest
    model, ledger = ScriptedModel(script(n), with_thinking=False), Ledger()
    msgs = [{"role": "user", "content": "charge"}]
    while True:
        r = await model.create(ModelRequest("sys", [], msgs, 1024))
        msgs.append({"role": "assistant", "content": r.content})
        calls = [b for b in r.content if b["type"] == "tool_use"]
        if not calls:
            return
        res = []
        for b in calls:
            ledger.executions += 1
            res.append({"type": "tool_result", "tool_use_id": b["id"], "content": '{"ok": true}'})
        msgs.append({"role": "user", "content": res})


async def timed(coro_fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        await coro_fn()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


async def bench_overhead(root: Path, reps=30):
    bare = await timed(bare_loop, reps)
    res = {}
    for fs in (False, True):
        async def one(fs=fs):
            await agent(RunStore(root / f"ov{fs}"), Ledger(), fsync=fs).run("Charge all orders.")
        res[fs] = await timed(one, reps)
    steps = STEPS + 1
    print(f"== overhead: {STEPS}-tool-call run, median of {reps} ==")
    print(f"  bare loop                 {bare * 1e3:7.2f} ms  ({bare / steps * 1e3:.2f} ms/step)")
    for fs, label in ((False, "loopguard, fsync off"), (True, "loopguard, fsync on")):
        add = (res[fs] - bare) / steps * 1e3
        print(f"  {label:<25} {res[fs] * 1e3:7.2f} ms  (+{add:.2f} ms/step added)")


async def bench_durability(root: Path):
    store = RunStore(root / "dur")
    ref = await agent(store, Ledger(), fsync=False).run("Charge all orders.")
    n_events = len(store.load_events(ref.run_id))
    points = dup_exec = dup_charge = wrong = 0
    resume_t = []
    for mode in ("after", "before"):
        for k in range(1, n_events + 1):
            led = Ledger()
            rid = f"{mode}{k}"
            try:
                await agent(store, led, fsync=False, **{f"crash_{mode}_event": k}).run("Charge all orders.", run_id=rid)
                continue  # run finished before reaching k
            except SimulatedCrash:
                pass
            if not store.exists(rid):
                continue
            points += 1
            t = time.perf_counter()
            r = await agent(store, led, fsync=False).resume(rid)
            resume_t.append(time.perf_counter() - t)
            dup_exec += led.executions - STEPS
            if len(led.charges) != STEPS:
                dup_charge += 1
            if r.status != "completed":
                wrong += 1
    print(f"== durability: crash at every event boundary of a {STEPS}-charge run ({n_events} events) ==")
    print(f"  crash points resumed      {points}")
    print(f"  runs not completed        {wrong}")
    print(f"  runs with a double charge {dup_charge}   (ledger dedupes on idempotency key)")
    print(f"  tool re-executions        {dup_exec}   (in-flight calls retried with the same key)")
    print(f"  resume time               p50 {statistics.median(resume_t) * 1e3:.1f} ms, "
          f"max {max(resume_t) * 1e3:.1f} ms")


async def bench_budget(root: Path):
    print("== budgets: runaway agent (500 scripted tool calls) under a USD cap ==")
    print(f"  {'cap':>7} {'spent':>9} {'% of cap':>9} {'steps':>6}  over cap?")
    for cap in (0.05, 0.25, 1.00, 5.00):
        r = await agent(RunStore(root / "bud"), Ledger(), fsync=False, script=script(500),
                        budget=Budget(max_usd=cap, max_steps=10_000, max_tokens=None,
                                      wall_clock_s=600)).run("Charge forever.")
        print(f"  ${cap:>6.2f} ${r.cost_usd:>8.4f} {r.cost_usd / cap:>9.1%} {r.steps:>6}  "
              f"{'YES' if r.cost_usd > cap else 'no'} ({r.status})")


async def main():
    print(f"python {platform.python_version()} on {platform.machine()} {platform.system()}\n")
    root = Path(tempfile.mkdtemp(prefix="lgbench-"))
    try:
        await bench_overhead(root)
        print()
        await bench_durability(root)
        print()
        await bench_budget(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
