"""ClaudeModel request/response mapping, verified against a fake client (no network)."""

from __future__ import annotations

from types import SimpleNamespace

from anthropic.types import Message

from loopguard.models import ClaudeModel, ModelRequest


def sdk_message(**over):
    data = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "EqQBCkYIBxgCKkA"},
            {"type": "text", "text": "Looking that up."},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup_order", "input": {"order_id": "A1001"}},
        ],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 2048},
    }
    data.update(over)
    return Message.model_validate(data)


class FakeClient:
    def __init__(self, msg):
        self.calls = []

        async def create(**kw):
            self.calls.append(kw)
            return msg

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


TOOLS = [{"name": "lookup_order", "description": "d", "strict": True,
          "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}]


async def test_request_params_follow_model_rules():
    client = FakeClient(sdk_message())
    model = ClaudeModel(client=client, refusal_fallback=False)
    await model.create(ModelRequest("SYSTEM", TOOLS, [{"role": "user", "content": "hi"}], 16000))
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["tool_choice"] == {"type": "auto"}
    assert kw["system"] == [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}]
    assert kw["cache_control"] == {"type": "ephemeral"}
    for forbidden in ("thinking", "temperature", "top_p", "top_k", "betas"):
        assert forbidden not in kw
    assert kw["messages"][-1]["role"] == "user"  # never an assistant prefill


async def test_refusal_fallback_uses_beta_default_mode():
    client = FakeClient(sdk_message())
    await ClaudeModel(client=client).create(ModelRequest("S", [], [{"role": "user", "content": "x"}], 100))
    kw = client.calls[0]
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tools" not in kw and "tool_choice" not in kw


async def test_response_mapping_preserves_blocks():
    resp = await ClaudeModel(client=FakeClient(sdk_message()), refusal_fallback=False).create(
        ModelRequest("S", TOOLS, [{"role": "user", "content": "x"}], 100))
    assert resp.stop_reason == "tool_use"
    assert resp.content[0] == {"type": "thinking", "thinking": "", "signature": "EqQBCkYIBxgCKkA"}
    assert resp.tool_uses[0]["input"] == {"order_id": "A1001"}
    assert resp.usage.cache_read_input_tokens == 2048 and resp.usage.input_tokens == 120


async def test_refusal_stop_details_mapped():
    msg = sdk_message(content=[], stop_reason="refusal",
                      stop_details={"type": "refusal", "category": "cyber", "explanation": "no"})
    resp = await ClaudeModel(client=FakeClient(msg), refusal_fallback=False).create(
        ModelRequest("S", [], [{"role": "user", "content": "x"}], 100))
    assert resp.stop_reason == "refusal" and resp.stop_details["category"] == "cyber"
