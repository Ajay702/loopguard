# loopguard architecture

This document explains how loopguard is built and why. It covers the trade-offs, the threat model, and what changes at scale.

## 1. The shape of the system

```
                 ┌──────────────────────── Agent (agent.py) ─────────────────────────┐
 task ──► run()  │  budget check ─► model call ─► append turn ─► stop_reason switch  │ ──► RunResult
                 │                   │  retries     (unchanged)    │                    │
                 │                   ▼                             ▼ tool_use           │
                 │            ModelBackend           for each call, concurrently:      │
                 │       (Claude | Scripted |        validate ─► policy deny ─► taint   │
                 │        FaultyModel wrapper)       ─► ask human ─► circuit breaker    │
                 │                                   ─► execute (timeout, retries)      │
                 │                                   ─► injection scan ─► quarantine    │
                 │                                   ═► ONE user message, all results   │
                 └──────────┬───────────────────────────────┬────────────────────────┘
                            │ every state transition        │ every span
                            ▼                               ▼
                  events.jsonl (raw, 0600)          spans.jsonl (redacted)
                  source of truth: resume()         trace.json / trace.html
```

| Module | Responsibility |
|---|---|
| `agent.py` | The manual tool-use loop, stop-reason handling, and composition of all guard layers |
| `models/claude.py` | Messages API backend built on the official `anthropic` SDK |
| `models/scripted.py` | Offline backend that replays scripted turns through the same interface |
| `tools.py` | `@tool` decorator: type hints → pydantic model → strict JSON schema |
| `budget.py`, `pricing.py` | Step, token, USD and wall-clock budgets, plus cost accounting |
| `resilience.py` | Typed retry classification, backoff with jitter, circuit breaker |
| `checkpoint.py` | Append-only event log; `RunState.from_events` folds the log back into state |
| `safety/policy.py` | allow / ask / deny rules with argument-level predicates |
| `safety/injection.py` | Scanner (heuristics + optional LLM), quarantine envelope, taint tracking |
| `safety/redact.py` | PII and secret detectors (Luhn-validated cards); logging filter |
| `tracing.py`, `trace_html.py` | Nested spans, JSON export, single-file HTML viewer |
| `faults.py` | Fault injection: real SDK exceptions, malformed tool input, hanging and flaky tools |
| `evals/` | Scenario suite that asserts outcomes, not implementation details |

## 2. Why a manual loop (and not the SDK tool runner)

The SDK's tool runner is the right default for most agents. loopguard owns the loop for these reasons:

- **Every transition must be durable before it takes effect.** The runtime writes the assistant turn to the log before it runs any tools, and writes each tool result before it commits the results message. This ordering is what makes resume correct, and it needs control over every boundary.
- **Retries belong to the runtime, not the transport.** The SDK client runs with `max_retries=0`. Every attempt then appears as a span event, counts against the wall-clock budget, and is visible to the eval suite. Hidden SDK retries would make cost and latency accounting wrong.
- **Policy, taint and approval run between the model's decision and the side effect.** A runner can host these hooks. A loop makes the ordering explicit, so you can audit it in one function (`_run_tool_inner`).

## 3. Request shape for claude-opus-5-5

| Decision | Reason |
|---|---|
| `output_config={"effort": "medium"}`, always sent | Effort is the only depth control on this model. Its default is `medium`, but sending it explicitly keeps behaviour stable across model upgrades |
| No `thinking`, no `budget_tokens`, no `temperature`, no prefill | Thinking cannot be disabled on Opus 5.5; budgets, sampling parameters and prefill all return 400 |
| `tool_choice: auto`, never `any` / `tool` | Forced tool choice returns 400 on this model. `strict: true` keeps the schema-valid-arguments guarantee |
| `strict: true` plus `additionalProperties: false` on every object | The API guarantees arguments match the schema. loopguard still re-validates with pydantic, because the fault layer (or a proxy) can corrupt input |
| Explicit `cache_control` on the system block, plus top-level automatic caching | Render order is tools → system → messages, so one breakpoint on the system prompt caches the tools too. Automatic caching covers the growing conversation tail |
| Tools sorted by name, system prompt frozen per run | Any byte change in the prefix invalidates the cache. The effective system prompt is stored in `run_started`, so a resumed run sends identical bytes |
| Server-side refusal fallbacks (`fallbacks: "default"`, beta) on by default | A decline from a safety classifier is retried on a fallback model inside the same call. If the whole chain declines, loopguard surfaces `stop_reason="refusal"` and its category. Set `ClaudeModel(refusal_fallback=False)` to opt out |

### Stop reasons

| `stop_reason` | Behaviour |
|---|---|
| `end_turn` / `stop_sequence` | Finish as `completed` |
| `tool_use` | Run every call concurrently, then return all results in **one** user message (in call order). Failures become `is_error: true` results. They are never dropped |
| `pause_turn` | Append the turn unchanged and call again, up to `max_pause_continuations` |
| `max_tokens` | The truncated turn is **discarded, never appended**, and the request is retried with double `max_tokens` (up to a ceiling). If the turn is still truncated, the run stops gracefully. A truncated `tool_use` input is never executed |
| `refusal` | Stop. Run no tools from that turn. Surface `stop_details.category` |

## 4. Event-sourced checkpoints

**Why event sourcing, not snapshots.** A snapshot written after each step tells you where the run was. An event log tells you how it got there. The log is the audit trail, the input to the trace exporter, and the resume state, all at once. It also makes the hardest question easy: *which tool calls finished before the crash?* A finished call has a `tool_result` event. An unfinished call does not.

The event types are `run_started`, `assistant_turn`, `discarded_response`, `model_retry`, `tool_result`, `approval`, `tool_results_committed`, `run_finished` and `run_resumed`.

**Write ordering.**

1. The model response arrives. The runtime appends `assistant_turn` to the log, then to the in-memory history.
2. The runtime runs each tool and appends its `tool_result` as soon as that tool finishes, so parallel tools are checkpointed independently.
3. When every result exists, the runtime appends `tool_results_committed` with the exact user message.

**Resume.** `RunState.from_events` replays the log and does the following:

- It rebuilds the message history byte-for-byte.
- It reuses logged tool results.
- It re-runs only calls that have no `tool_result` event.
- It restores budget counters, approvals and taint state.
- If the log ends in a terminal assistant turn (the crash happened after the final answer but before `run_finished`), it finishes without calling the model again.

**Exactly-once effects from at-least-once execution.** One crash window cannot be closed by logging: after the tool's side effect but before its `tool_result` is durable. A resumed run re-runs that call. loopguard therefore gives each tool `ToolContext.idempotency_key = f"{run_id}:{tool_use_id}"`. The key is stable across retries and across crash and resume, because the `tool_use_id` lives in the log. The downstream system deduplicates on it and returns the original response, Stripe-style. The `crash_resume_exactly_once` eval crashes the run before and after **every** event boundary (24 points) and asserts three things:

- the resumed history is identical to an uncrashed run;
- no logged call is ever re-run;
- exactly one refund and one email happen.

**Durability and integrity.** Each append is `flush` plus `fsync`. Torn final lines are detected and ignored. Files are created with mode `0600`. Every event carries `hash = sha256(prev_hash + canonical_event)`, so editing, deleting or reordering any line breaks the chain. `verify_chain()` reports the first bad `seq`, and `loopguard runs` shows the integrity of each run. This is tamper-*evident*, not tamper-*proof*: an attacker with write access can rewrite the whole chain. Anchoring the head hash in an external, append-only store closes that gap.

## 5. Why history is append-only

loopguard never edits, reorders or deletes an earlier turn. Assistant content blocks, including `thinking` blocks and their signatures, are passed back exactly as received. There are three reasons:

1. **Correctness on current models.** Opus 5.5 checks that replayed thinking blocks belong to an unedited conversation ("preserved thinking"). If you edit history, later thinking blocks become invalid, or the request is rejected on newer accounts.
2. **Caching.** The prompt cache matches on prefixes. Appending keeps request N as a strict prefix of request N+1, which a test asserts. One edit anywhere re-bills everything after it.
3. **Auditability and resume.** If history is a pure fold over an append-only log, "what did the model see at step 3?" has exactly one answer.

The consequences show up in the design. A truncated turn is discarded before it is appended, and is never appended and then removed. Security warnings are added inside the tool result as it is created, and never patched in later. A budget stop leaves pending `tool_use` blocks in place, and a later `resume()` answers them.

## 6. Prompt-injection threat model

**Assets.** Money (`issue_refund`), customer data (`send_email` as an exfiltration channel), and irreversible actions (`delete_*`).

**Attacker.** Anyone who can put text where a tool will read it: knowledge-base articles, web pages, emails, tickets, product reviews, or database fields. The attacker cannot change the system prompt, the user's message, or the code.

**Assumption.** The model *will* sometimes follow injected instructions. The demo and the evals use a scripted model that always falls for the attack, on purpose. Any defence that only works when the model resists is not a defence.

| Layer | Stops | Doesn't stop |
|---|---|---|
| Heuristic scanner | Common phrasings ("ignore previous instructions"), role spoofing, envelope-breaking tags, hidden Unicode, exfiltration phrasing. Fast, free and deterministic | Paraphrased, translated, encoded or subtle attacks. One test payload is deliberately missed by the heuristics |
| LLM classifier (optional, grey zone only) | Subtler attacks that the heuristics only partly match | Adversarial text that targets the classifier itself (mitigated by putting the reviewed text in an envelope). It also adds cost and latency |
| Quarantine envelope + system policy | Makes provenance explicit to the model, and stops payloads from closing or forging the envelope | A model that ignores the instruction anyway |
| Static policy (deny / ask) | Any action outside its bounds, whether injected or not (for example, email to external domains) | Harmful actions that fall *within* policy bounds |
| **Taint tracking** | Sensitive calls whose arguments use values (emails, URLs, IDs, amounts) that appear **only** in flagged content. The refund-hijack eval shows a $450 refund to another customer that the static policy would allow, blocked by taint | Content that was never flagged (taint needs detection first). Values that also appear in trusted context. Attacks that steer *which* trusted value is used |
| Human approval | Anything gated as `ask`. Taint is checked *before* asking, so nobody is asked to approve an injected action | Rubber-stamping by a careless approver |

**Known limits, stated plainly.**

- Detection is probabilistic. Taint is only as good as detection.
- Read-only exfiltration through the model's final answer is out of scope. The final answer goes to the user, who is the principal.
- Indicator extraction is lexical. An attacker who gets the model to compute a value (for example "refund the order total") produces no tainted token.
- A sound defence for high-stakes tools is a capability design that does not depend on detection at all. An example is the dual-LLM or CaMeL-style pattern, where a quarantined model reads untrusted data and the privileged model only sees symbolic references. That is on the roadmap.

## 7. PII: the redacted view and the raw store

Traces, HTML exports and logs are **redacted at write time**. Each span is redacted before it touches disk, and a `logging.Filter` sits on the `loopguard` logger. Detectors cover emails, phone numbers, card numbers (Luhn-validated, so order numbers survive), SSNs, and API keys of several shapes.

The **event log is not redacted**, by design. It is the state store that resume replays to the model. Redacting it would change the model's history after the fact, which breaks preserved thinking and changes what a resumed agent knows. So the log is treated like a database row: mode `0600`, excluded by `.gitignore`, and never exported. The `no_pii_in_observability` eval asserts both halves:

- traces, the HTML export and the logs contain no seeded PII or secrets;
- the event log is `0600` and keeps exact data.

A test also disables the redactor to prove that the PII check can fail.

**At scale:** encrypt the event log at rest with a per-tenant key (KMS), or tokenise PII before it reaches the model, using a vault that tools de-tokenise at execution time.

## 8. Testing without an API

`ScriptedModel` implements the same `ModelBackend` protocol as `ClaudeModel`. It picks the turn to play from **the number of assistant turns in the request history**, not from an internal counter. This makes scripts replay-stable: after a crash, `resume()` rebuilds the history and the script continues from exactly the right turn, just as a real model would respond to the restored context. A script step can be a callable, so a turn can react to tool results.

`FaultyModel` and `faulty_tool` wrap the real components. They raise the SDK's own exception classes (`RateLimitError` with `retry-after`, `InternalServerError`, `OverloadedError`, `APIConnectionError`, `APITimeoutError` and `BadRequestError`), so the retry classifier is tested against the exceptions production raises.

The live backend is covered by tests that use a fake client. Those tests assert the request parameters: effort, no `thinking` or `temperature`, cache breakpoints, `tool_choice=auto`, and the beta fallback path. They also assert that response blocks are mapped without changes. No test touches the network.

## 9. What I'd add at scale

| Need | Approach |
|---|---|
| Many concurrent runs, horizontal scale | Move the event log into Postgres (one row per event, `(run_id, seq)` primary key, optimistic concurrency on `seq`) or into an append-only stream (Kafka or Redpanda). Workers pull runs from a durable queue (SQS, Redis Streams) with a lease. A crashed worker's lease expires, and another worker calls `resume(run_id)`. Nothing else changes, because resume is already the recovery path |
| Exactly-once at the queue boundary | Use idempotency keys at every side-effecting tool (already present) plus a transactional outbox for emails and webhooks |
| Observability stack | An OTLP exporter for `Tracer`. The span model already mirrors OpenTelemetry (trace id, span id, parent, attributes, events, status). Map attributes to the GenAI semantic conventions (`gen_ai.usage.input_tokens` and so on) |
| Budgets across runs | Per-tenant and per-day spend limits in a shared store (Redis token bucket), checked alongside per-run budgets |
| Rate limits | A client-side concurrency limiter per model and tier, so 429s become queueing instead of retries |
| Durable approvals | `ask` becomes a durable "awaiting approval" state. The run parks and a webhook resumes it. That is the same resume path, with an `approval` event written by the approver |
| Policy | Load policies from versioned config (OPA or Cedar) so security teams can change them without a deploy. Log the policy version in `run_started` |
| Long runs | Server-side compaction or context editing. History must stay append-only from the API's point of view, so this means compaction blocks rather than client-side truncation |
| Evals | Run live evals nightly with statistical reporting (pass@k, confidence intervals), and grow the adversarial corpus for injection (paraphrase, translation, encoding) |
