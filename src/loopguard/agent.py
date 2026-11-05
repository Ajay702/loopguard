"""The agent runtime: a manual tool-use loop over the Messages API with every guard layer.

One step of the loop:

    budget check -> model call (retries/backoff) -> append assistant turn (unchanged)
      -> stop_reason dispatch
         tool_use : for each tool_use block, concurrently:
                      validate input -> taint check -> policy (allow/ask/deny)
                      -> circuit breaker -> execute (timeout, idempotent retries)
                      -> injection scan -> quarantine envelope
                    then ONE user message with every tool_result (errors: is_error=true)
         end_turn : done
         pause_turn : append and continue (bounded)
         max_tokens : discard + retry with a larger max_tokens, then give up gracefully
         refusal  : stop, surface stop_details

Every state transition is written to the event log first, which is what makes
``resume(run_id)`` possible after a crash.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
from pydantic import ValidationError

from .budget import Budget, BudgetExceeded, BudgetTracker
from .checkpoint import EventLog, RunState, RunStore
from .models.base import ContentBlock, Message, ModelBackend, ModelRequest, ModelResponse
from .pricing import cost_usd
from .resilience import (
    CircuitBreaker,
    RetryPolicy,
    is_retryable_api_error,
    is_retryable_tool_error,
    retry_async,
)
from .safety.injection import (
    UNTRUSTED_CONTENT_POLICY,
    HeuristicScanner,
    ScanResult,
    TaintTracker,
    wrap_untrusted,
)
from .safety.policy import ApprovalRequest, Approver, Decision, Policy
from .safety.redact import RedactingFilter
from .tools import (
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    TransientToolError,
    format_tool_output,
)
from .tracing import Tracer

logger = logging.getLogger("loopguard")
logger.addFilter(RedactingFilter())
logger.addHandler(logging.NullHandler())  # library: no output unless the app configures logging

EventHook = Callable[[str, dict[str, Any]], None]


class Scanner:  # structural type for injection scanners
    async def scan(self, text: str, source: str = "") -> ScanResult:  # pragma: no cover
        raise NotImplementedError


@dataclass
class RunResult:
    run_id: str
    status: str  # completed | budget_exceeded | refused | max_tokens | pause_limit | error
    final_text: str
    detail: str | None = None
    steps: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    actions: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    injections_flagged: int = 0
    retries: int = 0
    messages: list[Message] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    @property
    def partial(self) -> bool:
        return not self.ok


def new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"


async def _gather_or_cancel(coros: Iterable[Coroutine[Any, Any, Any]]) -> list[Any]:
    """gather() that cancels siblings if one fails hard (e.g. simulated process death)."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass
class _Session:
    run_id: str
    log: EventLog
    tracer: Tracer
    tracker: BudgetTracker
    messages: list[Message]
    taint: TaintTracker
    pending: list[ContentBlock] = field(default_factory=list)
    completed: dict[str, ContentBlock] = field(default_factory=dict)
    approvals: dict[str, bool] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    injections: int = 0
    retries: int = 0
    pause_continuations: int = 0
    last_response: ModelResponse | None = None


class Agent:
    def __init__(
        self,
        *,
        model: ModelBackend,
        tools: Iterable[Tool] = (),
        system: str,
        name: str = "agent",
        policy: Policy | None = None,
        approver: Approver | None = None,
        budget: Budget | None = None,
        retry: RetryPolicy | None = None,
        tool_retry: RetryPolicy | None = None,
        scanner: Scanner | None = None,
        quarantine: bool = True,
        taint_tracking: bool = True,
        store: RunStore | None = None,
        breaker_threshold: int = 3,
        breaker_reset_s: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_event: EventHook | None = None,
        fsync: bool = True,
        crash_after_event: int | None = None,
        crash_before_event: int | None = None,
    ) -> None:
        self.model = model
        self.tools = ToolRegistry(tools)
        self.system = system.rstrip() + ("\n\n" + UNTRUSTED_CONTENT_POLICY if quarantine else "")
        self.name = name
        self.policy = policy or Policy()
        self.approver = approver
        self.budget = budget or Budget()
        self.retry = retry or RetryPolicy()
        self.tool_retry = tool_retry or RetryPolicy(max_attempts=3, base_delay=0.2, max_delay=2.0)
        self.scanner: Scanner = scanner or HeuristicScanner()  # type: ignore[assignment]
        self.quarantine = quarantine
        self.taint_tracking = taint_tracking
        self.store = store or RunStore()
        self.breakers: dict[str, CircuitBreaker] = {
            t.name: CircuitBreaker(breaker_threshold, breaker_reset_s) for t in self.tools
        }
        self.sleep = sleep
        self.on_event = on_event
        self.fsync = fsync
        self.crash_after_event = crash_after_event
        self.crash_before_event = crash_before_event
        self._tool_defs = self.tools.definitions()

    # ------------------------------------------------------------------ public API

    async def run(self, task: str, *, run_id: str | None = None,
                  meta: dict[str, Any] | None = None) -> RunResult:
        run_id = run_id or new_run_id()
        if self.store.exists(run_id):
            raise ValueError(f"run {run_id} already exists; use resume()")
        sess = self._new_session(run_id, [{"role": "user", "content": task}],
                                 BudgetTracker(self.budget, self.model.model))
        sess.taint.add_trusted(task)
        sess.taint.add_trusted(self.system)
        try:
            self._emit(sess, "run_started", task=task, system=self.system,
                       model=self.model.model, agent=self.name, meta=meta or {},
                       tools=[t.name for t in self.tools], budget=self.budget.__dict__)
        except BaseException:
            sess.log.close()
            raise
        return await self._drive(sess, "run", task=task)

    async def resume(self, run_id: str, *, budget: Budget | None = None) -> RunResult:
        """Continue a crashed or budget-stopped run from its event log.

        Completed tool calls are never re-executed: their logged results are reused.
        In-flight calls (started, result not logged) are re-executed with the same
        idempotency key, so idempotent downstreams deduplicate them."""
        st = RunState.from_events(run_id, self.store.load_events(run_id))
        if st.finished and st.finished["status"] in ("completed", "refused"):
            return self._result_from_state(st)
        if budget is not None:
            self.budget = budget
        tracker = BudgetTracker(self.budget, self.model.model, steps=st.steps, usage=st.usage,
                                cost=st.cost, elapsed_offset=st.elapsed)
        sess = self._new_session(run_id, st.messages, tracker)
        sess.pending = st.pending_tool_uses
        sess.completed = st.completed_results
        sess.approvals = st.approvals
        sess.actions = list(st.actions)
        sess.blocked = list(st.blocked)
        sess.injections, sess.retries = st.injections, st.retries
        sess.pause_continuations = st.pause_continuations
        sess.last_response = st.last_response
        for text in st.trusted_texts:
            sess.taint.add_trusted(text)
        for text, source in st.tainted_texts:
            sess.taint.add_tainted(text, source)
        if st.model and st.model != self.model.model:
            logger.warning("resuming %s on %s (started on %s): prior thinking blocks will be "
                           "ignored by the new model", run_id, self.model.model, st.model)
        try:
            self._emit(sess, "run_resumed", pending_tool_calls=[b["id"] for b in sess.pending],
                       reused_results=list(sess.completed))
        except BaseException:
            sess.log.close()
            raise
        return await self._drive(sess, "resume", resumed_from_step=st.steps)

    def _terminal_status(self, sess: _Session) -> tuple[str, str | None] | None:
        """If the history already ends in a terminal assistant turn (crash between the final
        model response and ``run_finished``), the run is over - don't call the model again."""
        resp = sess.last_response
        if resp is None or sess.pending or sess.messages[-1]["role"] != "assistant":
            return None
        if resp.stop_reason in ("end_turn", "stop_sequence"):
            return "completed", None
        if resp.stop_reason == "refusal":
            d = resp.stop_details or {}
            return "refused", f"category={d.get('category')}: {d.get('explanation') or 'model declined'}"
        if resp.stop_reason == "max_tokens":
            return "max_tokens", "response truncated"
        return None

    # ------------------------------------------------------------------ internals

    def _new_session(self, run_id: str, messages: list[Message], tracker: BudgetTracker) -> _Session:
        log = EventLog(self.store.events_path(run_id), run_id, fsync=self.fsync,
                       crash_after_event=self.crash_after_event,
                       crash_before_event=self.crash_before_event)
        tracer = Tracer(run_id, sink=self.store.spans_path(run_id))
        return _Session(run_id, log, tracer, tracker, messages, TaintTracker())

    def _emit(self, sess: _Session, type: str, **data: Any) -> dict[str, Any]:
        event = sess.log.append(type, **data)
        if self.on_event:
            self.on_event(type, event)
        return event

    def _notice(self, kind: str, **data: Any) -> None:
        if self.on_event:
            self.on_event(kind, data)

    async def _drive(self, sess: _Session, span_name: str, **attrs: Any) -> RunResult:
        try:
            async with sess.tracer.span(span_name, "run", agent=self.name,
                                        model=self.model.model, **attrs) as span:
                result = await self._loop(sess)
                span.set(status=result.status, steps=result.steps,
                         total_tokens=result.total_tokens, cost_usd=round(result.cost_usd, 6))
                if result.status != "completed":
                    span.status = "error" if result.status == "error" else "ok"
                return result
        finally:
            sess.log.close()

    async def _loop(self, sess: _Session) -> RunResult:
        b = self.budget
        max_tokens = b.max_output_tokens
        terminal = self._terminal_status(sess)
        if terminal:
            return self._finish(sess, terminal[0], detail=terminal[1])
        while True:
            if sess.pending:  # resumed with tool calls from the last turn not yet answered
                # Token/cost budgets are not re-checked here: the model already paid for these
                # calls and answering them costs nothing. The next model call is gated below.
                remaining = sess.tracker.remaining_wall_clock()
                if remaining is not None and remaining <= 0:
                    return self._finish(sess, "budget_exceeded", detail="wall_clock budget exceeded")
                try:
                    await self._execute_tools(sess)
                except BudgetExceeded as e:
                    return self._finish(sess, "budget_exceeded", detail=str(e))
                continue

            try:
                sess.tracker.check()
            except BudgetExceeded as e:
                return self._finish(sess, "budget_exceeded", detail=str(e))

            step = sess.tracker.steps + 1
            sess.tracker.steps = step
            async with sess.tracer.span(f"step {step}", "step", step=step) as step_span:
                try:
                    resp = await self._call_model(sess, max_tokens)
                except BudgetExceeded as e:
                    return self._finish(sess, "budget_exceeded", detail=str(e))
                except anthropic.APIError as e:
                    step_span.status = "error"
                    return self._finish(sess, "error", detail=f"{type(e).__name__}: {e}")

                cost = sess.tracker.record(resp.usage, resp.model or self.model.model)

                if resp.stop_reason == "max_tokens" and max_tokens < b.max_output_tokens_ceiling:
                    # A truncated turn is discarded (never appended - history stays clean)
                    # and re-requested with more room.
                    new_max = min(b.max_output_tokens_ceiling, max_tokens * 2)
                    self._emit(sess, "discarded_response", step=step, reason="max_tokens",
                               usage=resp.usage.to_dict(), cost_usd=cost,
                               old_max_tokens=max_tokens, new_max_tokens=new_max)
                    step_span.add_event("truncated_retry", new_max_tokens=new_max)
                    max_tokens = new_max
                    continue

                self._emit(sess, "assistant_turn", step=step, response=resp.to_dict(),
                           cost_usd=cost)
                # Append the assistant content exactly as returned (thinking blocks and
                # signatures included). History is append-only from here on.
                sess.messages.append({"role": "assistant", "content": resp.content})
                sess.last_response = resp
                max_tokens = b.max_output_tokens
                step_span.set(stop_reason=resp.stop_reason, tool_calls=len(resp.tool_uses))

                sr = resp.stop_reason
                if sr in ("end_turn", "stop_sequence"):
                    return self._finish(sess, "completed")
                if sr == "tool_use":
                    sess.pending = resp.tool_uses
                    sess.pause_continuations = 0
                    try:
                        sess.tracker.check()
                    except BudgetExceeded as e:
                        # Don't execute side effects whose results the model will never see.
                        return self._finish(sess, "budget_exceeded", detail=str(e))
                    try:
                        await self._execute_tools(sess)
                    except BudgetExceeded as e:
                        return self._finish(sess, "budget_exceeded", detail=str(e))
                    continue
                if sr == "pause_turn":
                    sess.pause_continuations += 1
                    if sess.pause_continuations > b.max_pause_continuations:
                        return self._finish(sess, "pause_limit",
                                            detail=f"exceeded {b.max_pause_continuations} continuations")
                    continue
                if sr == "max_tokens":
                    return self._finish(sess, "max_tokens",
                                        detail=f"response truncated at {max_tokens} tokens")
                if sr == "refusal":
                    details = resp.stop_details or {}
                    return self._finish(sess, "refused",
                                        detail=f"category={details.get('category')}: "
                                               f"{details.get('explanation') or 'model declined'}")
                return self._finish(sess, "error", detail=f"unknown stop_reason {sr!r}")

    async def _call_model(self, sess: _Session, max_tokens: int) -> ModelResponse:
        request = ModelRequest(self.system, self._tool_defs, list(sess.messages), max_tokens)
        async with sess.tracer.span("model_call", "model_call", model=self.model.model,
                                    max_tokens=max_tokens,
                                    history_messages=len(sess.messages)) as span:
            t0 = time.monotonic()

            def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
                sess.retries += 1
                status = getattr(exc, "status_code", None)
                span.add_event("retry", attempt=attempt, error=type(exc).__name__,
                               status_code=status, delay_s=round(delay, 3))
                sess.log.append("model_retry", attempt=attempt, error=type(exc).__name__,
                                status_code=status, delay_s=round(delay, 3))
                logger.warning("model call failed (%s), retry %d in %.2fs",
                               type(exc).__name__, attempt, delay)
                self._notice("retry", target="model", attempt=attempt,
                             error=type(exc).__name__, status_code=status, delay_s=delay)

            async def attempt(n: int) -> ModelResponse:
                span.set(attempts=n)
                return await self.model.create(request)

            remaining = sess.tracker.remaining_wall_clock()
            deadline = time.monotonic() + remaining if remaining is not None else None
            call = retry_async(attempt, policy=self.retry, retryable=is_retryable_api_error,
                               on_retry=on_retry, sleep=self.sleep, deadline=deadline)
            try:
                resp = await (asyncio.wait_for(call, remaining) if remaining is not None else call)
            except asyncio.TimeoutError as e:
                raise BudgetExceeded("wall_clock", "deadline hit during model call") from e
            except anthropic.APIError as e:
                span.set(error_type=type(e).__name__,
                         retryable=is_retryable_api_error(e))
                raise
            span.set(
                stop_reason=resp.stop_reason,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cache_read_input_tokens=resp.usage.cache_read_input_tokens,
                cache_creation_input_tokens=resp.usage.cache_creation_input_tokens,
                total_tokens=resp.usage.total_tokens,
                cost_usd=round(cost_usd(resp.usage, resp.model or self.model.model), 6),
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                request_id=resp.request_id,
                text=resp.text,
                tool_calls=[{"name": b["name"], "input": b["input"]} for b in resp.tool_uses],
            )
            return resp

    async def _execute_tools(self, sess: _Session) -> None:
        uses = list(sess.pending)
        step = sess.tracker.steps

        async def one(block: ContentBlock) -> ContentBlock:
            if block["id"] in sess.completed:  # finished before a crash: reuse, don't re-run
                return sess.completed[block["id"]]
            outcome = await self._run_tool(sess, block)
            self._emit(sess, "tool_result", step=step, tool_use_id=block["id"],
                       name=block["name"], status=outcome.status, reason=outcome.reason,
                       block=outcome.block, raw_output=outcome.raw_output,
                       flagged=outcome.flagged, duration_ms=outcome.duration_ms)
            sess.completed[block["id"]] = outcome.block
            sess.actions.append({"tool": block["name"], "status": outcome.status,
                                 "tool_use_id": block["id"]})
            if outcome.status == "blocked":
                sess.blocked.append({"tool": block["name"], "reason": outcome.reason})
            return outcome.block

        remaining = sess.tracker.remaining_wall_clock()
        gathered = _gather_or_cancel(one(b) for b in uses)
        try:
            results = await (asyncio.wait_for(gathered, remaining) if remaining is not None
                             else gathered)
        except asyncio.TimeoutError as e:
            raise BudgetExceeded("wall_clock", "deadline hit during tool execution") from e

        # All results for one assistant turn go back in ONE user message, in call order.
        message = {"role": "user", "content": results}
        self._emit(sess, "tool_results_committed", step=step, message=message)
        sess.messages.append(message)
        sess.pending = []
        sess.completed = {}

    async def _run_tool(self, sess: _Session, block: ContentBlock) -> _ToolOutcome:
        name, tid, raw_input = block["name"], block["id"], block.get("input")
        t0 = time.monotonic()
        async with sess.tracer.span(f"tool {name}", "tool_call", tool=name, tool_use_id=tid,
                                    input=raw_input) as span:
            outcome = await self._run_tool_inner(sess, name, tid, raw_input, span)
            outcome.duration_ms = round((time.monotonic() - t0) * 1000, 2)
            span.set(outcome=outcome.status, reason=outcome.reason,
                     injection_flagged=outcome.flagged,
                     injection_score=outcome.scan.score if outcome.scan else None,
                     injection_findings=[f.name for f in outcome.scan.findings] if outcome.scan else [],
                     output=outcome.raw_output if outcome.raw_output is not None else outcome.reason)
            if outcome.status == "blocked":
                span.status = "blocked"
            elif outcome.status != "ok":
                span.status = "error"
            logger.info("tool %s(%s) -> %s %s", name, raw_input, outcome.status, outcome.reason)
            self._notice("tool", name=name, input=raw_input, status=outcome.status,
                         reason=outcome.reason, flagged=outcome.flagged,
                         findings=[f.name for f in outcome.scan.findings] if outcome.scan else [])
            return outcome

    async def _run_tool_inner(self, sess: _Session, name: str, tid: str, raw_input: Any,
                              span: Any) -> _ToolOutcome:
        t = self.tools.get(name)
        if t is None:
            return _ToolOutcome.error(tid, "unknown_tool", f"no tool named {name!r}")

        try:
            args = t.validate(raw_input)
        except ValidationError as e:
            errs = "; ".join(f"{'.'.join(map(str, er['loc'])) or 'input'}: {er['msg']}"
                             for er in e.errors())
            return _ToolOutcome.error(tid, "invalid_input",
                                      f"invalid input for {name}: {errs}. Fix the arguments and retry.")
        args_dict = args.model_dump(mode="json")

        # 1) static policy: hard denies first
        decision = self.policy.evaluate(name, args_dict)
        span.set(policy_decision=decision.decision.value, policy_rule=decision.rule)
        if decision.decision is Decision.DENY:
            return _ToolOutcome.blocked(tid, f"policy denied: {decision.reason}")

        # 2) taint: was this sensitive action steered by quarantined content? Checked before
        #    asking a human, so nobody is asked to rubber-stamp an injected action.
        if self.taint_tracking and t.sensitive:
            hit = sess.taint.check(args_dict)
            if hit:
                indicator, source = hit
                span.set(taint_indicator=indicator, taint_source=source)
                return _ToolOutcome.blocked(
                    tid, f"taint: argument value '{indicator}' originates from untrusted "
                         f"content ({source}) flagged as prompt injection")

        # 3) human approval for gated actions
        if decision.decision is Decision.ASK:
            approved = sess.approvals.get(tid)
            if approved is None:
                approved = await self._ask(sess, t, tid, args_dict, decision.reason)
            if not approved:
                return _ToolOutcome.blocked(tid, f"human approval denied: {decision.reason}")

        # 4) circuit breaker
        breaker = self.breakers[name]
        if not breaker.allow():
            return _ToolOutcome.error(tid, "circuit_open",
                                      f"{name} is temporarily unavailable (circuit open after "
                                      f"repeated failures). Do not retry it now.")

        # 5) execute with timeout and (idempotent-only) retries
        timeout = t.timeout or self.budget.tool_timeout_s
        policy = RetryPolicy(max_attempts=1 + t.retries, base_delay=self.tool_retry.base_delay,
                             max_delay=self.tool_retry.max_delay, jitter=self.tool_retry.jitter)

        def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
            span.add_event("retry", attempt=attempt, error=type(exc).__name__,
                           delay_s=round(delay, 3))
            self._notice("retry", target=name, attempt=attempt, error=type(exc).__name__,
                         delay_s=delay)

        async def attempt(n: int) -> Any:
            return await asyncio.wait_for(t.invoke(args, ToolContext(sess.run_id, tid, n)), timeout)

        try:
            output = await retry_async(attempt, policy=policy, retryable=is_retryable_tool_error,
                                       on_retry=on_retry, sleep=self.sleep)
        except asyncio.TimeoutError:
            breaker.record_failure()
            return _ToolOutcome.error(tid, "timeout", f"{name} timed out after {timeout}s")
        except TransientToolError as e:
            breaker.record_failure()
            return _ToolOutcome.error(tid, "error", f"{name} failed (transient): {e}")
        except ToolError as e:  # business error: the tool is healthy, the request wasn't
            breaker.record_success()
            return _ToolOutcome.error(tid, "error", f"{name} failed: {e}")
        except Exception as e:
            breaker.record_failure()
            return _ToolOutcome.error(tid, "error", f"{name} raised {type(e).__name__}: {e}")
        breaker.record_success()

        # 6) scan + quarantine
        raw = format_tool_output(output)
        scan = await self.scanner.scan(raw, source=f"tool:{name}")
        if scan.flagged:
            sess.injections += 1
            sess.taint.add_tainted(raw, f"tool:{name}")
            self._notice("injection", tool=name, score=scan.score,
                         findings=[f.name for f in scan.findings])
        else:
            sess.taint.add_trusted(raw)
        content = (wrap_untrusted(raw, source=f"tool:{name}", call_id=tid, scan=scan)
                   if self.quarantine else raw)
        return _ToolOutcome(
            block={"type": "tool_result", "tool_use_id": tid, "content": content},
            status="ok", raw_output=raw, flagged=scan.flagged, scan=scan)

    async def _ask(self, sess: _Session, t: Tool, tid: str, args: dict[str, Any],
                   reason: str) -> bool:
        if self.approver is None:
            approved = False
            note = "no approver configured"
        else:
            req = ApprovalRequest(sess.run_id, t.name, args, reason, tid)
            answer = self.approver(req)
            approved = bool(await answer) if inspect.isawaitable(answer) else bool(answer)
            note = "human decision"
        sess.approvals[tid] = approved
        self._emit(sess, "approval", tool_use_id=tid, tool=t.name, approved=approved,
                   reason=reason, note=note)
        return approved

    def _finish(self, sess: _Session, status: str, *, detail: str | None = None) -> RunResult:
        last_text = sess.last_response.text if sess.last_response else ""
        if status != "completed":
            done = [a for a in sess.actions if a["status"] == "ok"]
            summary = ", ".join(a["tool"] for a in done) or "none"
            last_text = (f"[partial result - run stopped: {status}"
                         f"{' (' + detail + ')' if detail else ''}]\n"
                         f"Completed tool calls: {summary}\n" + (last_text or ""))
        snap = sess.tracker.snapshot()
        self._emit(sess, "run_finished", status=status, detail=detail, final_text=last_text,
                   stats=snap, blocked=sess.blocked, injections_flagged=sess.injections)
        return RunResult(
            run_id=sess.run_id, status=status, final_text=last_text, detail=detail,
            steps=sess.tracker.steps, total_tokens=sess.tracker.usage.total_tokens,
            cost_usd=sess.tracker.cost, elapsed_s=sess.tracker.elapsed,
            actions=list(sess.actions), blocked=list(sess.blocked),
            injections_flagged=sess.injections, retries=sess.retries,
            messages=sess.messages)

    def _result_from_state(self, st: RunState) -> RunResult:
        fin = st.finished or {}
        return RunResult(
            run_id=st.run_id, status=fin.get("status", "unknown"),
            final_text=fin.get("final_text", ""), detail=fin.get("detail"), steps=st.steps,
            total_tokens=st.usage.total_tokens, cost_usd=st.cost, elapsed_s=st.elapsed,
            actions=st.actions, blocked=st.blocked, injections_flagged=st.injections,
            retries=st.retries, messages=st.messages)


@dataclass
class _ToolOutcome:
    block: ContentBlock
    status: str  # ok | error | invalid_input | timeout | blocked | circuit_open | unknown_tool
    reason: str = ""
    raw_output: str | None = None
    flagged: bool = False
    scan: ScanResult | None = None
    duration_ms: float = 0.0

    @classmethod
    def error(cls, tid: str, status: str, message: str) -> _ToolOutcome:
        return cls({"type": "tool_result", "tool_use_id": tid, "content": f"Error: {message}",
                    "is_error": True}, status, message)

    @classmethod
    def blocked(cls, tid: str, reason: str) -> _ToolOutcome:
        msg = (f"Blocked by loopguard: {reason}. This action was not performed. Do not retry "
               f"it; tell the user what was blocked and why.")
        return cls({"type": "tool_result", "tool_use_id": tid, "content": msg, "is_error": True},
                   "blocked", reason)
