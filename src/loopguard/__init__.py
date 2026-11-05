"""loopguard - a production runtime for tool-using AI agents.

Budgets, resilience, durable checkpoints, guardrails and tracing around a manual
Claude Messages API tool-use loop.
"""

from .agent import Agent, RunResult
from .budget import Budget, BudgetExceeded
from .checkpoint import RunState, RunStore, SimulatedCrash
from .models import ClaudeModel, ModelRequest, ModelResponse, ScriptedModel, Usage
from .resilience import CircuitBreaker, RetryPolicy
from .safety import Decision, Policy, allow, ask, deny
from .tools import Tool, ToolContext, ToolError, TransientToolError, tool
from .tracing import Tracer

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Budget",
    "BudgetExceeded",
    "CircuitBreaker",
    "ClaudeModel",
    "Decision",
    "ModelRequest",
    "ModelResponse",
    "Policy",
    "RetryPolicy",
    "RunResult",
    "RunState",
    "RunStore",
    "ScriptedModel",
    "SimulatedCrash",
    "Tool",
    "ToolContext",
    "ToolError",
    "Tracer",
    "TransientToolError",
    "Usage",
    "allow",
    "ask",
    "deny",
    "tool",
]
