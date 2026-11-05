"""Tracing: nested spans (run -> step -> model_call / tool_call) with tokens, cost, latency.

Spans propagate through ``contextvars``, so concurrently executing tool calls (each in its
own asyncio task) correctly parent to the step that launched them. Every finished span is
redacted and appended to ``spans.jsonl`` immediately, so a crashed run still leaves a
trace; resumed sessions append to the same file and appear in the same trace.

The data model mirrors OpenTelemetry (trace id, span id, parent id, attributes, events,
status) so an OTLP exporter is a thin adapter - see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import contextvars
import json
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .checkpoint import read_jsonl
from .safety.redact import Redactor

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "loopguard_span", default=None
)

MAX_ATTR_CHARS = 2000


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_ATTR_CHARS:
        return value[:MAX_ATTR_CHARS] + f"... [{len(value) - MAX_ATTR_CHARS} chars truncated]"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    return value


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start: float
    end: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append({"name": name, "ts": time.time(), **attrs})

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 2)


class Tracer:
    def __init__(self, trace_id: str, sink: Path | None = None,
                 redactor: Redactor | None = None) -> None:
        self.trace_id = trace_id
        self.sink = sink
        self.redactor = redactor or Redactor()
        self.spans: list[Span] = []

    @asynccontextmanager
    async def span(self, name: str, kind: str, **attrs: Any) -> AsyncIterator[Span]:
        parent = _current_span.get()
        span = Span(
            trace_id=self.trace_id,
            span_id=secrets.token_hex(8),
            parent_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start=time.time(),
            attributes=dict(attrs),
        )
        token = _current_span.set(span)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.set(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _current_span.reset(token)
            span.end = time.time()
            self._finish(span)

    def _finish(self, span: Span) -> None:
        self.spans.append(span)
        if self.sink is None:
            return
        record = asdict(span)
        record["duration_ms"] = span.duration_ms
        record = self.redactor.redact(_truncate(record))
        self.sink.parent.mkdir(parents=True, exist_ok=True)
        with self.sink.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def load_trace(spans_path: Path, run_id: str, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the exported trace document (already-redacted spans + run summary)."""
    spans = list(read_jsonl(spans_path))
    spans.sort(key=lambda s: s["start"])
    model_spans = [s for s in spans if s["kind"] == "model_call"]
    tool_spans = [s for s in spans if s["kind"] == "tool_call"]
    tokens = sum(int(s["attributes"].get("total_tokens", 0)) for s in model_spans)
    cost = sum(float(s["attributes"].get("cost_usd", 0.0)) for s in model_spans)
    finished = next((e for e in reversed(events or []) if e["type"] == "run_finished"), None)
    t0 = spans[0]["start"] if spans else 0.0
    t1 = max((s["end"] or s["start"] for s in spans), default=t0)
    summary = {
        "run_id": run_id,
        "status": finished["status"] if finished else "incomplete",
        "duration_ms": round((t1 - t0) * 1000, 2),
        "model_calls": len(model_spans),
        "tool_calls": len(tool_spans),
        "tokens": tokens,
        "cost_usd": round(cost, 6),
        "blocked_actions": sum(1 for s in tool_spans if s["status"] == "blocked"),
        "injections_flagged": sum(1 for s in tool_spans if s["attributes"].get("injection_flagged")),
        "errors": sum(1 for s in spans if s["status"] == "error"),
        "retries": sum(1 for s in spans for e in s.get("events", []) if e["name"] == "retry"),
    }
    return {"trace_id": run_id, "summary": summary, "spans": spans}


def export_trace(run_dir: Path, run_id: str, events: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    from .trace_html import render_html

    doc = load_trace(run_dir / "spans.jsonl", run_id, events)
    json_path = run_dir / "trace.json"
    html_path = run_dir / "trace.html"
    json_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    html_path.write_text(render_html(doc), encoding="utf-8")
    os.chmod(json_path, 0o644)
    return json_path, html_path
