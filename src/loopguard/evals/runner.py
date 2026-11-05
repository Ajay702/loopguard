"""Runs the scenario suite and renders a scorecard.

On top of each scenario's own checks, the runner applies one global check to every run a
scenario produced: export the trace (JSON + HTML) and capture the ``loopguard`` logger,
then assert that none of the customer PII / secrets seeded into the workload appear there.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..examples.support import SupportBackend
from ..safety.redact import find_pii
from ..tracing import export_trace
from .scenarios import SCENARIOS, Check, EvalContext, Scenario

SEEDED_SECRETS = ["sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx"]


@dataclass
class ScenarioResult:
    name: str
    category: str
    title: str
    status: str  # pass | fail | error | skipped
    checks: list[Check] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None

    @property
    def applicable(self) -> list[Check]:
        return self.checks

    @property
    def passed_checks(self) -> int:
        return sum(c.passed for c in self.checks)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def _pii_literals(backends: list[SupportBackend]) -> set[str]:
    values: set[str] = set(SEEDED_SECRETS)
    for be in backends:
        for order in be.orders.values():
            values |= {order["email"], order["phone"], order["card"]}
    return values


def _pii_check(ctx: EvalContext, run_ids: list[str], backends: list[SupportBackend],
               log_lines: list[str]) -> Check:
    literals = _pii_literals(backends)
    leaks: list[str] = []
    for run_id in run_ids:
        run_dir = ctx.store.run_dir(run_id)
        if not (run_dir / "spans.jsonl").exists():
            continue
        json_path, html_path = export_trace(run_dir, run_id, ctx.store.load_events(run_id))
        for path in (json_path, html_path):
            blob = path.read_text(encoding="utf-8")
            leaks += [f"{path.name}:{v}" for v in literals if v in blob]
            leaks += [f"{path.name}:{label}" for label, _ in find_pii(blob)]
    logs = "\n".join(log_lines)
    leaks += [f"log:{v}" for v in literals if v in logs]
    leaks += [f"log:{label}" for label, _ in find_pii(logs)]
    return Check("no PII/secrets in traces, HTML export or logs", not leaks,
                 ", ".join(sorted(set(leaks))[:5]))


async def run_scenario(sc: Scenario, ctx: EvalContext) -> ScenarioResult:
    if ctx.live and not sc.live_ok:
        return ScenarioResult(sc.name, sc.category, sc.title, "skipped",
                              error="offline-only (needs scripted model control)")
    log = logging.getLogger("loopguard")
    cap = _Capture()
    old_level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(cap)
    t0 = time.monotonic()
    try:
        run = await sc.fn(ctx)
    except Exception as e:  # a scenario crashing is a failed scenario, not a crashed suite
        return ScenarioResult(sc.name, sc.category, sc.title, "error",
                              duration_s=time.monotonic() - t0, error=f"{type(e).__name__}: {e}")
    finally:
        log.removeHandler(cap)
        log.setLevel(old_level)
    checks = [c for c in run.checks if not (ctx.live and c.offline_only)]
    checks.append(_pii_check(ctx, run.run_ids, run.backends, cap.lines))
    status = "pass" if all(c.passed for c in checks) else "fail"
    return ScenarioResult(sc.name, sc.category, sc.title, status, checks,
                          duration_s=time.monotonic() - t0)


async def run_suite(*, live: bool = False, only: list[str] | None = None,
                    store_root: Path | None = None) -> list[ScenarioResult]:
    root = store_root or Path(tempfile.mkdtemp(prefix="loopguard-eval-"))
    ctx = EvalContext(live=live, store_root=root)
    selected = [s for s in SCENARIOS if not only or s.name in only]
    return [await run_scenario(s, ctx) for s in selected]


def summarize(results: list[ScenarioResult]) -> dict[str, Any]:
    ran = [r for r in results if r.status != "skipped"]
    checks = [c for r in ran for c in r.checks]
    return {
        "scenarios_passed": sum(r.status == "pass" for r in ran),
        "scenarios_run": len(ran),
        "scenarios_skipped": len(results) - len(ran),
        "checks_passed": sum(c.passed for c in checks),
        "checks_total": len(checks),
        "duration_s": round(sum(r.duration_s for r in results), 2),
    }


def to_json(results: list[ScenarioResult]) -> str:
    return json.dumps({"summary": summarize(results), "results": [asdict(r) for r in results]},
                      indent=2)


def to_markdown(results: list[ScenarioResult]) -> str:
    s = summarize(results)
    lines = [
        "| # | Scenario | Category | Checks | Result |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        mark = {"pass": "PASS", "fail": "FAIL", "error": "ERROR", "skipped": "skipped"}[r.status]
        checks = f"{r.passed_checks}/{len(r.checks)}" if r.checks else "-"
        lines.append(f"| {i} | `{r.name}` - {r.title} | {r.category} | {checks} | {mark} |")
    lines.append("")
    lines.append(f"**{s['scenarios_passed']}/{s['scenarios_run']} scenarios passed, "
                 f"{s['checks_passed']}/{s['checks_total']} checks passed** "
                 f"in {s['duration_s']}s (offline, scripted model + fault injection).")
    return "\n".join(lines)


def render_rich(results: list[ScenarioResult], *, verbose: bool = False) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="loopguard eval scorecard", title_style="bold", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Scenario")
    table.add_column("Category", style="cyan")
    table.add_column("Checks", justify="right")
    table.add_column("Time", justify="right", style="dim")
    table.add_column("Result", justify="center")
    styles = {"pass": "[bold green]PASS[/]", "fail": "[bold red]FAIL[/]",
              "error": "[bold red]ERROR[/]", "skipped": "[yellow]skip[/]"}
    for i, r in enumerate(results, 1):
        checks = f"{r.passed_checks}/{len(r.checks)}" if r.checks else "-"
        table.add_row(str(i), f"{r.name}\n[dim]{r.title}[/]", r.category, checks,
                      f"{r.duration_s:.2f}s", styles[r.status])
    console.print(table)
    for r in results:
        failing = [c for c in r.checks if not c.passed]
        if r.error and r.status != "skipped":
            console.print(f"[red]{r.name}: {r.error}[/]")
        if failing or verbose:
            for c in (r.checks if verbose else failing):
                mark = "[green]ok[/]" if c.passed else "[red]FAIL[/]"
                console.print(f"  {mark} {r.name}: {c.name} [dim]{c.detail}[/]")
    s = summarize(results)
    color = "green" if s["scenarios_passed"] == s["scenarios_run"] else "red"
    console.print(
        f"[bold {color}]{s['scenarios_passed']}/{s['scenarios_run']} scenarios passed[/] · "
        f"{s['checks_passed']}/{s['checks_total']} checks · {s['duration_s']}s"
        + (f" · {s['scenarios_skipped']} skipped" if s["scenarios_skipped"] else "")
    )
