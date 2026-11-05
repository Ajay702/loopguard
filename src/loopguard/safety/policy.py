"""Permission policy engine: allow / ask / deny per tool, with argument-level rules.

    policy = Policy(
        rules=[
            deny("send_email", when=arg_domain_not_in("to", {"acme.com"}),
                 reason="email to external domains is not allowed"),
            ask("issue_refund", when=arg_gt("amount", 500), reason="large refund"),
            ask("delete_*", reason="destructive action"),
        ],
        default=Decision.ALLOW,
    )

Semantics:
* Tool patterns are shell-style globs (``delete_*``).
* Every matching rule is considered and the **most restrictive** decision wins
  (deny > ask > allow). Order-independent evaluation means a later, broader ``allow`` can
  never accidentally override an earlier ``deny`` - a common bug in first-match engines.
* A predicate that raises is treated as a match (fail closed).
* ``ask`` needs a human approval callback. With no approver configured, ``ask`` becomes
  ``deny`` - an unattended agent never silently performs a gated action.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

Predicate = Callable[[dict[str, Any]], bool]


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

    @property
    def severity(self) -> int:
        return {"allow": 0, "ask": 1, "deny": 2}[self.value]


@dataclass(frozen=True)
class Rule:
    tool: str
    decision: Decision
    when: Predicate | None = None
    reason: str = ""
    name: str = ""

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not fnmatch.fnmatchcase(tool_name, self.tool):
            return False
        if self.when is None:
            return True
        try:
            return bool(self.when(args))
        except Exception:
            return True  # fail closed


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str
    rule: str | None = None


class Policy:
    def __init__(self, rules: Iterable[Rule] = (), default: Decision = Decision.ALLOW) -> None:
        self.rules = list(rules)
        self.default = default

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> PolicyResult:
        matched = [r for r in self.rules if r.matches(tool_name, args)]
        if not matched:
            return PolicyResult(self.default, "default policy", None)
        worst = max(matched, key=lambda r: r.decision.severity)
        return PolicyResult(worst.decision, worst.reason or worst.decision.value,
                            worst.name or f"{worst.decision.value}:{worst.tool}")


def allow(tool: str, *, when: Predicate | None = None, reason: str = "", name: str = "") -> Rule:
    return Rule(tool, Decision.ALLOW, when, reason, name)


def ask(tool: str, *, when: Predicate | None = None, reason: str = "", name: str = "") -> Rule:
    return Rule(tool, Decision.ASK, when, reason, name)


def deny(tool: str, *, when: Predicate | None = None, reason: str = "", name: str = "") -> Rule:
    return Rule(tool, Decision.DENY, when, reason, name)


# --- argument predicates -------------------------------------------------------------

def _get(args: dict[str, Any], field: str) -> Any:
    value: Any = args
    for part in field.split("."):
        value = value[part] if isinstance(value, dict) else getattr(value, part)
    return value


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def arg_domain_not_in(field: str, domains: Iterable[str]) -> Predicate:
    """True if any address in ``field`` is outside ``domains`` (subdomains allowed)."""
    allowed = {d.lower().lstrip("@") for d in domains}

    def pred(args: dict[str, Any]) -> bool:
        for addr in _as_list(_get(args, field)):
            domain = str(addr).rsplit("@", 1)[-1].strip().lower().rstrip(".>")
            if not any(domain == d or domain.endswith("." + d) for d in allowed):
                return True
        return False

    return pred


def arg_not_in(field: str, values_fn: Callable[[], Iterable[Any]] | Iterable[Any]) -> Predicate:
    """True if the value is not in an allow-list (which may be computed lazily)."""

    def pred(args: dict[str, Any]) -> bool:
        allowed = set(values_fn() if callable(values_fn) else values_fn)
        return any(str(v).lower() not in {str(a).lower() for a in allowed}
                   for v in _as_list(_get(args, field)))

    return pred


def arg_gt(field: str, threshold: float) -> Predicate:
    return lambda args: float(_get(args, field)) > threshold


def arg_matches(field: str, pattern: str) -> Predicate:
    rx = re.compile(pattern)
    return lambda args: any(rx.search(str(v)) for v in _as_list(_get(args, field)))


# --- human approval --------------------------------------------------------------------

@dataclass(frozen=True)
class ApprovalRequest:
    run_id: str
    tool: str
    args: dict[str, Any]
    reason: str
    tool_use_id: str


Approver = Callable[[ApprovalRequest], "Awaitable[bool] | bool"]


def auto_approver(decision: bool) -> Approver:
    """Fixed-answer approver for tests and unattended runs."""
    return lambda _req: decision
