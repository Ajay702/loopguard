"""Tool declaration: a decorator that turns a typed Python function into a strict Claude tool.

    @tool(idempotent=True, timeout=5)
    async def lookup_order(order_id: str) -> dict:
        '''Look up an order. Call this before refunding or emailing about an order.

        Args:
            order_id: The order identifier, e.g. "A1001".
        '''

The JSON schema is derived from the type hints via a generated pydantic model, then made
*strict*: ``additionalProperties: false`` on every object, titles removed, and the tool
definition carries ``strict: true`` so the API guarantees schema-valid arguments. The same
pydantic model validates arguments again at execution time - defence in depth, because the
fault-injection layer (and, in principle, a proxy) can still hand us malformed input.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, get_args, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


class ToolError(Exception):
    """A tool failure whose message should be shown to the model (not retried)."""


class TransientToolError(ToolError):
    """A failure that may succeed on retry (timeouts, flaky upstreams). Retried only for
    tools declared ``idempotent=True``."""


@dataclass(frozen=True)
class ToolContext:
    """Injected into tools that declare a parameter annotated with ``ToolContext``."""

    run_id: str
    tool_use_id: str
    attempt: int = 1

    @property
    def idempotency_key(self) -> str:
        # Stable across retries *and* across crash/resume: the tool_use id is persisted in
        # the event log, so a resumed run re-executing an in-flight call reuses the key and
        # the downstream system can deduplicate. This is what turns at-least-once execution
        # into exactly-once effects.
        return f"{self.run_id}:{self.tool_use_id}"


_ARGS_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters):\s*$", re.MULTILINE)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    doc = inspect.cleandoc(doc or "")
    match = _ARGS_SECTION.search(doc)
    if not match:
        return doc.strip(), {}
    summary = doc[: match.start()].strip()
    params: dict[str, str] = {}
    current: str | None = None
    for line in doc[match.end():].splitlines():
        if not line.strip():
            continue
        m = re.match(r"^\s{2,}(\w+)\s*(\([^)]*\))?:\s*(.*)$", line)
        if m:
            current = m.group(1)
            params[current] = m.group(3).strip()
        elif re.match(r"^\S", line):  # next section (Returns:, Raises: ...)
            break
        elif current:
            params[current] += " " + line.strip()
    return summary, params


def make_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively harden a pydantic JSON schema for strict tool use."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("properties", {})
                node["additionalProperties"] = False
                node.setdefault("required", [])
            return node
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    input_model: type[BaseModel]
    idempotent: bool = False
    timeout: float | None = None
    retries: int = 0
    sensitive: bool = False
    context_param: str | None = None
    model_param: str | None = None  # set when fn takes a single pydantic model
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def input_schema(self) -> dict[str, Any]:
        return make_strict(self.input_model.model_json_schema())

    def definition(self) -> dict[str, Any]:
        """The tool definition sent to the Messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }

    def validate(self, raw: Any) -> BaseModel:
        if not isinstance(raw, dict):
            raise ValidationError.from_exception_data(
                self.input_model.__name__,
                [{"type": "dict_type", "loc": (), "input": raw}],
            )
        return self.input_model.model_validate(raw)

    async def invoke(self, args: BaseModel, ctx: ToolContext) -> Any:
        if self.model_param:
            kwargs: dict[str, Any] = {self.model_param: args}
        else:
            kwargs = {name: getattr(args, name) for name in type(args).model_fields}
        if self.context_param:
            kwargs[self.context_param] = ctx
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        # Sync tools run in a worker thread so they cannot block the event loop and so
        # per-tool timeouts still fire. (A thread cannot be killed - a timed-out sync tool
        # keeps running in the background; prefer async tools for anything slow.)
        return await asyncio.to_thread(self.fn, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # still callable as a function
        return self.fn(*args, **kwargs)


def _build_tool(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    idempotent: bool,
    timeout: float | None,
    retries: int,
    sensitive: bool,
    tags: Iterable[str],
) -> Tool:
    summary, param_docs = _parse_docstring(fn.__doc__ or "")
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)

    context_param: str | None = None
    model_param: str | None = None
    fields: dict[str, Any] = {}
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    for p in params:
        ann = hints.get(p.name, Any)
        if ann is ToolContext or ToolContext in get_args(ann):
            context_param = p.name
            continue
        fields[p.name] = (ann, p)

    if len(fields) == 1:
        (only_name, (ann, _p)), = fields.items()
        if inspect.isclass(ann) and issubclass(ann, BaseModel):
            model_param = only_name
            input_model = ann
    if model_param is None:
        model_fields: dict[str, Any] = {}
        for pname, (ann, p) in fields.items():
            default = ... if p.default is inspect.Parameter.empty else p.default
            model_fields[pname] = (ann, Field(default, description=param_docs.get(pname)))
        input_model = create_model(  # type: ignore[call-overload]
            f"{fn.__name__}_input",
            __config__=ConfigDict(extra="forbid"),
            **model_fields,
        )

    return Tool(
        name=name or fn.__name__,
        description=(description or summary or fn.__name__).strip(),
        fn=fn,
        input_model=input_model,
        idempotent=idempotent,
        timeout=timeout,
        retries=retries if idempotent else 0,
        sensitive=sensitive,
        context_param=context_param,
        model_param=model_param,
        tags=frozenset(tags),
    )


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    idempotent: bool = False,
    timeout: float | None = None,
    retries: int = 2,
    sensitive: bool = False,
    tags: Iterable[str] = (),
) -> Any:
    """Declare a tool. Usable bare (``@tool``) or with options (``@tool(idempotent=True)``).

    ``idempotent``: safe to retry automatically on transient failure (reads, keyed writes).
    ``retries``: extra attempts for idempotent tools (ignored for non-idempotent tools).
    ``sensitive``: the tool has external side effects (money, email, deletion). Sensitive
    tools are what the prompt-injection taint check protects.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return _build_tool(
            f,
            name=name,
            description=description,
            idempotent=idempotent,
            timeout=timeout,
            retries=retries,
            sensitive=sensitive,
            tags=tags,
        )

    return wrap(fn) if fn is not None else wrap


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        # Deterministic order: tool definitions are part of the cached prompt prefix, and a
        # reordering would silently invalidate the cache.
        return [self._tools[n].definition() for n in sorted(self._tools)]


def format_tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=str, sort_keys=True)
