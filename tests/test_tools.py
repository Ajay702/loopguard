from __future__ import annotations

import threading
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from loopguard.tools import ToolContext, ToolRegistry, make_strict, tool


class Address(BaseModel):
    street: str
    city: str


@tool(idempotent=True, timeout=3)
async def ship(order_id: str, address: Address, speed: Literal["standard", "express"] = "standard",
               ctx: ToolContext | None = None) -> str:
    """Ship an order. Call this after payment clears.

    Args:
        order_id: The order to ship.
        address: Destination address.
        speed: Shipping speed.
    """
    return f"{order_id}->{address.city} ({speed})"


def test_schema_is_strict_and_documented():
    d = ship.definition()
    assert d["strict"] is True
    assert d["description"] == "Ship an order. Call this after payment clears."
    schema = d["input_schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["order_id", "address"]
    assert schema["properties"]["order_id"]["description"] == "The order to ship."
    assert schema["properties"]["speed"]["enum"] == ["standard", "express"]
    assert "ctx" not in schema["properties"]  # injected, not model-visible
    nested = schema["$defs"]["Address"]
    assert nested["additionalProperties"] is False
    assert "title" not in nested and "title" not in schema


def test_make_strict_is_recursive():
    out = make_strict({"type": "object", "title": "x", "properties": {
        "a": {"type": "array", "items": {"type": "object", "properties": {"b": {"type": "string", "default": "q"}}}}}})
    item = out["properties"]["a"]["items"]
    assert item["additionalProperties"] is False and item["required"] == []
    assert "default" not in item["properties"]["b"] and "title" not in out


def test_validation_rejects_extra_and_wrong_types():
    with pytest.raises(ValidationError):
        ship.validate({"order_id": 1, "address": {"street": "s", "city": "c"}})
    with pytest.raises(ValidationError):
        ship.validate({"order_id": "A", "address": {"street": "s", "city": "c"}, "extra": 1})
    with pytest.raises(ValidationError):
        ship.validate("not a dict")


async def test_invoke_passes_context_and_nested_model():
    args = ship.validate({"order_id": "A1", "address": {"street": "1 Main", "city": "Oslo"}})
    assert await ship.invoke(args, ToolContext("run", "toolu_1")) == "A1->Oslo (standard)"


async def test_sync_tools_run_off_the_event_loop():
    seen = {}

    @tool
    def where() -> str:
        """Report thread."""
        seen["thread"] = threading.current_thread() is threading.main_thread()
        return "ok"

    assert await where.invoke(where.validate({}), ToolContext("r", "t")) == "ok"
    assert seen["thread"] is False


def test_single_pydantic_param_is_used_as_schema():
    @tool
    def create(addr: Address) -> str:
        """Create an address."""
        return addr.city

    assert set(create.input_schema["properties"]) == {"street", "city"}


def test_retries_only_for_idempotent_tools():
    @tool(retries=5)
    def write() -> None:
        """Write."""

    assert write.retries == 0 and ship.retries == 2


def test_registry_definitions_are_sorted_and_unique():
    @tool
    def b_tool() -> None:
        """B."""

    @tool
    def a_tool() -> None:
        """A."""

    reg = ToolRegistry([b_tool, a_tool])
    assert [d["name"] for d in reg.definitions()] == ["a_tool", "b_tool"]
    with pytest.raises(ValueError):
        reg.add(a_tool)


def test_idempotency_key_is_stable():
    assert ToolContext("run_1", "toolu_9", attempt=3).idempotency_key == "run_1:toolu_9"
