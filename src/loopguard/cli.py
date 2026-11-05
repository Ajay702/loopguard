"""loopguard command line: demo, eval, trace, resume, runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .agent import new_run_id
from .checkpoint import DEFAULT_RUNS_DIR, RunState, RunStore, SimulatedCrash, verify_chain
from .models.claude import credentials_available

console = Console()


class DemoPrinter:
    """Turns runtime events into a readable, narrated console stream."""

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        handler = getattr(self, f"on_{kind}", None)
        if handler:
            handler(data)

    def on_run_started(self, d: dict[str, Any]) -> None:
        console.print(Panel(d["task"], title="[bold]Customer message[/]", border_style="blue"))

    def on_run_resumed(self, d: dict[str, Any]) -> None:
        console.print(Panel(
            f"Rebuilt conversation from the event log. Reusing {len(d['reused_results'])} "
            f"completed tool result(s); {len(d['pending_tool_calls'])} call(s) still pending.",
            title="[bold]Resumed after crash[/]", border_style="magenta"))

    def on_assistant_turn(self, d: dict[str, Any]) -> None:
        resp = d["response"]
        console.rule(f"[bold]Step {d['step']}[/]  [dim]stop_reason={resp['stop_reason']} · "
                     f"${d['cost_usd']:.4f}[/]", style="dim")
        said = "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text").strip()
        if said:
            console.print(f"[bold cyan]agent[/] {said}")
        for b in resp["content"]:
            if b["type"] == "tool_use":
                args = json.dumps(b["input"])
                args = args if len(args) <= 90 else args[:87] + "..."
                console.print(f"  [dim]calls[/] [bold]{b['name']}[/]([dim]{escape(args)}[/])")

    def on_retry(self, d: dict[str, Any]) -> None:
        what = "model API" if d["target"] == "model" else f"tool {d['target']}"
        code = f" {d['status_code']}" if d.get("status_code") else ""
        console.print(f"  [yellow]\\[retry][/] {what} failed with {d['error']}{code}; "
                      f"backing off {d['delay_s']:.2f}s (attempt {d['attempt']})")

    def on_tool(self, d: dict[str, Any]) -> None:
        status = d["status"]
        if status == "ok":
            console.print(f"  [green]\\[ok][/] {d['name']}")
        elif status == "blocked":
            console.print(f"  [bold red]\\[BLOCKED][/] {d['name']} -> {escape(d['reason'])}")
        else:
            console.print(f"  [red]\\[{status}][/] {d['name']}: {escape(d['reason'])}")
        if d.get("flagged"):
            console.print(f"  [bold red]\\[injection detected][/] {d['name']} output matched "
                          f"{', '.join(d['findings'])} -> quarantined; values in it are now "
                          f"tainted for sensitive tools")

    def on_approval(self, d: dict[str, Any]) -> None:
        verdict = "approved" if d["approved"] else "denied"
        console.print(f"  [magenta]\\[approval][/] {d['tool']}: {verdict} ({d['note']})")


def _summary(result: Any, store: RunStore) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    color = "green" if result.status == "completed" else "yellow"
    table.add_row("status", f"[bold {color}]{result.status}[/] {result.detail or ''}")
    table.add_row("steps", str(result.steps))
    table.add_row("tokens", f"{result.total_tokens:,}")
    table.add_row("cost", f"${result.cost_usd:.4f}")
    table.add_row("model API retries", str(result.retries))
    table.add_row("injections flagged", str(result.injections_flagged))
    table.add_row("actions blocked", "; ".join(f"{b['tool']}: {b['reason']}" for b in result.blocked) or "0")
    console.print(Panel(table, title=f"[bold]run {result.run_id}[/]", border_style=color))
    if result.final_text:
        console.print(Panel(result.final_text, title="[bold]Final answer[/]", border_style="cyan"))
    console.print(f"[dim]Event log:[/] {store.events_path(result.run_id)}")
    console.print(f"[dim]View the trace:[/] loopguard trace {result.run_id}")


def _store(args: argparse.Namespace) -> RunStore:
    return RunStore(Path(args.runs_dir))


def _require_live() -> None:
    if not credentials_available():
        console.print("[red]--live needs Claude API credentials (ANTHROPIC_API_KEY, "
                      "ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile).[/]")
        sys.exit(2)


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import build_demo_agent
    from .examples.support import DEMO_TASK

    if args.live:
        _require_live()
    store = _store(args)
    console.print(Panel(
        "Offline demo: a support agent with real guardrails, driven by a scripted model "
        "that gets [bold]fooled[/] by a poisoned help-center article. Injected faults: a "
        "429 from the model API and a flaky order lookup."
        if not args.live else "Live demo against Claude (claude-opus-5-5).",
        title="[bold]loopguard demo[/]", border_style="white"))
    agent, backend = build_demo_agent(store=store, live=args.live, on_event=DemoPrinter(),
                                      crash_after_event=args.crash_after)
    run_id = new_run_id()
    try:
        result = asyncio.run(agent.run(DEMO_TASK, run_id=run_id, meta={"live": args.live}))
    except SimulatedCrash as e:
        console.print(f"\n[bold red]Simulated crash:[/] {e}")
        console.print(f"Resume with: [bold]loopguard resume {run_id}[/]")
        return 1
    console.print()
    _summary(result, store)
    console.print(f"[dim]Side effects:[/] refunds={backend.refunds} emails_to="
                  f"{[m['to'] for m in backend.outbox]}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from .demo import DEMO_AGENT, build_demo_agent

    store = _store(args)
    if not store.exists(args.run_id):
        console.print(f"[red]No run {args.run_id} in {store.root}[/]")
        return 2
    st = RunState.from_events(args.run_id, store.load_events(args.run_id))
    if st.agent != DEMO_AGENT:
        console.print(f"[red]Run was created by agent {st.agent!r}; the CLI can only resume "
                      f"{DEMO_AGENT!r} runs. Use Agent.resume() from your own code.[/]")
        return 2
    live = bool(st.meta.get("live"))
    if live:
        _require_live()
    agent, _ = build_demo_agent(store=store, live=live, on_event=DemoPrinter(), inject_faults=False)
    result = asyncio.run(agent.resume(args.run_id))
    console.print()
    _summary(result, store)
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .tracing import export_trace

    store = _store(args)
    if not store.exists(args.run_id):
        console.print(f"[red]No run {args.run_id} in {store.root}[/]")
        return 2
    json_path, html_path = export_trace(store.run_dir(args.run_id), args.run_id,
                                        store.load_events(args.run_id))
    doc = json.loads(json_path.read_text())
    s = doc["summary"]
    console.print(f"[bold]{args.run_id}[/] · {s['status']} · {s['model_calls']} model calls · "
                  f"{s['tool_calls']} tool calls · {s['tokens']:,} tokens · ${s['cost_usd']:.4f} · "
                  f"{s['blocked_actions']} blocked · {s['injections_flagged']} injections · "
                  f"{s['retries']} retries")
    console.print(f"JSON: {json_path}\nHTML: {html_path}  [dim](single self-contained file)[/]")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = _store(args)
    table = Table(title=f"runs in {store.root}")
    for col in ("run id", "agent", "status", "steps", "cost", "log integrity"):
        table.add_column(col)
    for run_id in store.list_runs():
        events = store.load_events(run_id)
        try:
            st = RunState.from_events(run_id, events)
        except ValueError:
            continue
        status = st.finished["status"] if st.finished else "[yellow]incomplete (resumable)[/]"
        broken = verify_chain(events)
        integrity = "[green]hash chain ok[/]" if broken is None else f"[red]tampered at seq {broken}[/]"
        table.add_row(run_id, st.agent, status, str(st.steps), f"${st.cost:.4f}", integrity)
    console.print(table)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals import render_rich, run_suite, summarize, to_json, to_markdown

    if args.live:
        _require_live()
    results = asyncio.run(run_suite(live=args.live, only=args.only or None))
    render_rich(results, verbose=args.verbose)
    if args.json:
        Path(args.json).write_text(to_json(results))
    if args.markdown:
        print(to_markdown(results))
    s = summarize(results)
    return 0 if s["scenarios_passed"] == s["scenarios_run"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loopguard",
                                     description="Production runtime for tool-using AI agents.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR),
                        help="where event logs and traces are stored (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the support-agent demo (offline by default)")
    p.add_argument("--live", action="store_true", help="use Claude instead of the scripted model")
    p.add_argument("--crash-after", type=int, metavar="N",
                   help="simulate a process crash after the Nth event-log write")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("eval", help="run the scenario eval suite and print a scorecard")
    p.add_argument("--live", action="store_true", help="run against Claude (costs money)")
    p.add_argument("--only", nargs="*", help="scenario names to run")
    p.add_argument("--json", metavar="PATH", help="write full results as JSON")
    p.add_argument("--markdown", action="store_true", help="also print a markdown scorecard")
    p.add_argument("-v", "--verbose", action="store_true", help="show every check")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("trace", help="export a run's trace as JSON + static HTML viewer")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser("resume", help="resume a crashed or budget-stopped run")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("runs", help="list recorded runs")
    p.set_defaults(fn=cmd_runs)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
