"""Durable, event-sourced checkpoints.

Every state transition of a run is appended to ``<runs>/<run_id>/events.jsonl`` *before*
the runtime acts on it. The conversation is never stored as a mutable blob; it is a pure
function of the event log. ``RunState.from_events`` folds the log back into:

* the exact message history (assistant turns stored byte-for-byte, so thinking blocks and
  their signatures are replayed unchanged),
* which tool calls of the last assistant turn already completed (their results are reused,
  never re-executed) and which are still pending,
* budget counters (tokens, cost, steps, wall-clock) and taint state.

Tamper evidence: every event carries ``hash = sha256(prev_hash + canonical event)``, so
editing, deleting or reordering any line breaks the chain (``verify_chain``).

Durability: each append is flushed and ``fsync``-ed. A torn final line (crash mid-write)
is detected and ignored on load. Files are created ``0600`` - the event log is the run's
source of truth and therefore holds raw tool data; human-facing artefacts (traces, logs,
exports) are the redacted views.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models.base import Message, ModelResponse, Usage

DEFAULT_RUNS_DIR = Path(os.environ.get("LOOPGUARD_RUNS_DIR", ".loopguard/runs"))


class SimulatedCrash(BaseException):
    """Raised by fault injection to emulate the process being killed. Deliberately a
    ``BaseException`` so ordinary ``except Exception`` handlers cannot swallow it."""


class RunStore:
    def __init__(self, root: str | Path = DEFAULT_RUNS_DIR) -> None:
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.root / run_id

    def events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def spans_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "spans.jsonl"

    def exists(self, run_id: str) -> bool:
        path = self.events_path(run_id)
        return path.exists() and path.stat().st_size > 0

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        runs = [p for p in self.root.iterdir() if (p / "events.jsonl").exists()]
        return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime)]

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        return list(read_jsonl(self.events_path(run_id)))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Torn write from a crash: only the final line can be partial; stop here.
                return


GENESIS = "0" * 64


def event_hash(prev_hash: str, event: dict[str, Any]) -> str:
    body = {k: v for k, v in event.items() if k != "hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()


def verify_chain(events: list[dict[str, Any]]) -> int | None:
    """Return the seq of the first event whose hash doesn't match, or None if intact."""
    prev = GENESIS
    for ev in events:
        if ev.get("hash") != event_hash(prev, ev):
            return int(ev.get("seq", -1))
        prev = ev["hash"]
    return None


def _open_private(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    return os.fdopen(fd, "a", encoding="utf-8")


class EventLog:
    """Append-only JSONL log with optional crash injection for testing."""

    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        fsync: bool = True,
        crash_after_event: int | None = None,
        crash_before_event: int | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.fsync = fsync
        self.crash_after_event = crash_after_event
        self.crash_before_event = crash_before_event
        existing = list(read_jsonl(path))
        self.seq = existing[-1]["seq"] if existing else 0
        self.prev_hash = existing[-1].get("hash", GENESIS) if existing else GENESIS
        self.session = (max((e.get("session", 0) for e in existing), default=-1) + 1)
        self._written_this_session = 0
        self._dead = False
        self._fh = _open_private(path)

    def append(self, type: str, **data: Any) -> dict[str, Any]:
        if self._dead:
            raise SimulatedCrash("process is dead")
        n = self._written_this_session + 1
        if self.crash_before_event is not None and n == self.crash_before_event:
            self._dead = True
            raise SimulatedCrash(f"crash before event #{n} ({type})")
        self.seq += 1
        event = {"seq": self.seq, "ts": time.time(), "session": self.session,
                 "run_id": self.run_id, "type": type, **data}
        event = json.loads(json.dumps(event, default=str))  # canonical JSON types
        event["hash"] = event_hash(self.prev_hash, event)
        self.prev_hash = event["hash"]
        self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())
        self._written_this_session = n
        if self.crash_after_event is not None and n == self.crash_after_event:
            self._dead = True
            raise SimulatedCrash(f"crash after event #{n} ({type})")
        return event

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


@dataclass
class RunState:
    run_id: str
    task: str = ""
    system: str = ""
    model: str = ""
    agent: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    pending_tool_uses: list[dict[str, Any]] = field(default_factory=list)
    completed_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals: dict[str, bool] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    steps: int = 0
    elapsed: float = 0.0
    pause_continuations: int = 0
    finished: dict[str, Any] | None = None
    trusted_texts: list[str] = field(default_factory=list)
    tainted_texts: list[tuple[str, str]] = field(default_factory=list)
    tool_stats: dict[str, int] = field(default_factory=dict)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    injections: int = 0
    last_response: ModelResponse | None = None

    @classmethod
    def from_events(cls, run_id: str, events: list[dict[str, Any]]) -> RunState:
        if not events or events[0]["type"] != "run_started":
            raise ValueError(f"run {run_id}: event log is empty or corrupt")
        st = cls(run_id=run_id)
        start = events[0]
        st.task, st.system, st.model = start["task"], start["system"], start["model"]
        st.agent, st.meta = start.get("agent", ""), start.get("meta", {})
        st.messages = [{"role": "user", "content": st.task}]
        st.trusted_texts = [st.task, st.system]

        sessions: dict[int, tuple[float, float]] = {}
        for ev in events:
            s = ev.get("session", 0)
            lo, hi = sessions.get(s, (ev["ts"], ev["ts"]))
            sessions[s] = (min(lo, ev["ts"]), max(hi, ev["ts"]))
            t = ev["type"]
            if t == "assistant_turn":
                resp = ModelResponse.from_dict(ev["response"])
                st.last_response = resp
                st.messages.append({"role": "assistant", "content": resp.content})
                st.usage = st.usage + resp.usage
                st.cost += ev.get("cost_usd", 0.0)
                st.steps = max(st.steps, ev["step"])
                st.pending_tool_uses = resp.tool_uses if resp.stop_reason == "tool_use" else []
                st.completed_results = {}
                st.pause_continuations = (
                    st.pause_continuations + 1 if resp.stop_reason == "pause_turn" else 0
                )
            elif t == "discarded_response":
                st.usage = st.usage + Usage.from_dict(ev["usage"])
                st.cost += ev.get("cost_usd", 0.0)
                st.steps = max(st.steps, ev["step"])
            elif t == "tool_result":
                st.completed_results[ev["tool_use_id"]] = ev["block"]
                st.tool_stats[ev["status"]] = st.tool_stats.get(ev["status"], 0) + 1
                st.actions.append({"tool": ev["name"], "status": ev["status"],
                                   "tool_use_id": ev["tool_use_id"]})
                st.injections += 1 if ev.get("flagged") else 0
                if ev["status"] == "blocked":
                    st.blocked.append({"tool": ev["name"], "reason": ev.get("reason", "")})
                if ev.get("raw_output") is not None:
                    if ev.get("flagged"):
                        st.tainted_texts.append((ev["raw_output"], f"tool:{ev['name']}"))
                    else:
                        st.trusted_texts.append(ev["raw_output"])
            elif t == "model_retry":
                st.retries += 1
            elif t == "approval":
                st.approvals[ev["tool_use_id"]] = bool(ev["approved"])
            elif t == "tool_results_committed":
                st.messages.append(ev["message"])
                st.pending_tool_uses = []
                st.completed_results = {}
            elif t == "run_finished":
                st.finished = ev
            elif t == "run_resumed":
                st.finished = None
        st.elapsed = sum(hi - lo for lo, hi in sessions.values())
        return st
