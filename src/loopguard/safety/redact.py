"""PII and secret redaction for logs, traces and exports.

Detectors:
* email addresses
* phone numbers (international ``+CC ...`` and North-American formats)
* payment card numbers - 13-19 digits (spaces/dashes allowed) that pass the Luhn check,
  so order numbers and timestamps are not falsely redacted
* API keys / secrets - Anthropic (``sk-ant-...``), OpenAI-style ``sk-...``, AWS access
  key ids, GitHub tokens, Slack tokens, bearer tokens, and ``key=value`` style secrets
* US social security numbers

Redaction replaces the match with a typed placeholder such as ``[EMAIL]`` so traces stay
readable ("the agent emailed [EMAIL]") without leaking the value.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def luhn_valid(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class Detector:
    label: str
    pattern: re.Pattern[str]
    validate: Callable[[str], bool] | None = None
    group: int = 0


DETECTORS: tuple[Detector, ...] = (
    Detector("API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{10,}")),
    Detector("API_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}")),
    Detector("API_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    Detector("API_KEY", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    Detector("API_KEY", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    Detector("API_KEY", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-._~+/]{20,}=*)"), group=1),
    Detector(
        "API_KEY",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*[\"']?([^\s\"',;\[]{8,})"
        ),
        group=1,
    ),
    Detector("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    Detector(
        "CARD",
        re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
        validate=luhn_valid,
    ),
    Detector("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    Detector(
        "PHONE",
        re.compile(
            r"(?<![\w+])(?:\+\d{1,3}[ .\-]?)?(?:\(\d{3}\)|\d{3})[ .\-]\d{3}[ .\-]\d{4}(?!\d)"
            r"|(?<![\w+])\+\d{1,3}[ .\-]?\d{2,4}[ .\-]?\d{3,4}[ .\-]?\d{3,4}(?!\d)"
        ),
    ),
)


@dataclass
class Redactor:
    detectors: tuple[Detector, ...] = DETECTORS
    counts: dict[str, int] = field(default_factory=dict)

    def redact_text(self, text: str) -> str:
        for det in self.detectors:
            def _sub(m: re.Match[str], det: Detector = det) -> str:
                value = m.group(det.group)
                if det.validate and not det.validate(value):
                    return m.group(0)
                self.counts[det.label] = self.counts.get(det.label, 0) + 1
                if det.group:
                    start, end = m.span(det.group)
                    s0 = m.start(0)
                    whole = m.group(0)
                    return whole[: start - s0] + f"[{det.label}]" + whole[end - s0 :]
                return f"[{det.label}]"

            text = det.pattern.sub(_sub, text)
        return text

    def redact(self, value: Any) -> Any:
        """Recursively redact strings inside dicts/lists/tuples. Keys are left alone."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {k: self.redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.redact(v) for v in value)
        return value


_default = Redactor()


def redact(value: Any) -> Any:
    return _default.redact(value)


def find_pii(text: str) -> list[tuple[str, str]]:
    """Return (label, match) pairs - used by tests and the no-PII-in-logs eval."""
    found: list[tuple[str, str]] = []
    for det in DETECTORS:
        for m in det.pattern.finditer(text):
            value = m.group(det.group)
            if det.validate and not det.validate(value):
                continue
            found.append((det.label, value))
    return found


class RedactingFilter(logging.Filter):
    """``logging`` filter that redacts the formatted message and args of every record."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        super().__init__()
        self.redactor = redactor or Redactor()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = self.redactor.redact_text(message)
        record.args = None
        return True
