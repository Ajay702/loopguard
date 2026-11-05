from __future__ import annotations

import logging

import pytest

from loopguard.safety.redact import RedactingFilter, Redactor, find_pii, luhn_valid, redact


@pytest.mark.parametrize("text,label", [
    ("contact dana.whitfield@gmail.com now", "EMAIL"),
    ("call +1 415-555-0142", "PHONE"),
    ("call (212) 555-0199", "PHONE"),
    ("uk +44 20 7946 0958", "PHONE"),
    ("card 4242 4242 4242 4242", "CARD"),
    ("card 5555-5555-5555-4444", "CARD"),
    ("key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx", "API_KEY"),
    ("AKIAABCDEFGHIJKLMNOP", "API_KEY"),
    ("ghp_" + "a" * 36, "API_KEY"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456", "API_KEY"),
    ("password=hunter2hunter2", "API_KEY"),
    ("ssn 123-45-6789", "SSN"),
])
def test_detects_and_redacts(text, label):
    out = redact(text)
    assert f"[{label}]" in out, out
    assert not find_pii(out), find_pii(out)


def test_luhn_prevents_false_positives():
    assert luhn_valid("4242424242424242")
    assert not luhn_valid("4242424242424241")
    # order numbers / timestamps that look like cards but fail Luhn are left alone
    assert redact("order 1234 5678 9012 3456") == "order 1234 5678 9012 3456"


def test_keeps_context_around_keyed_secrets():
    assert redact("api_key: abcdefgh12345678") == "api_key: [API_KEY]"


def test_recursive_structures_and_counts():
    r = Redactor()
    out = r.redact({"to": ["a@b.co", "x"], "meta": ({"phone": "+1 415-555-0142"},), "n": 3})
    assert out == {"to": ["[EMAIL]", "x"], "meta": ({"phone": "[PHONE]"},), "n": 3}
    assert r.counts == {"EMAIL": 1, "PHONE": 1}


def test_benign_text_untouched():
    s = "Refund of $89.99 for order A1001 on 2026-11-05 at 14:03, 3-5 business days."
    assert redact(s) == s


def test_logging_filter_redacts_formatted_message(caplog):
    log = logging.getLogger("test.redact")
    log.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger="test.redact"):
        log.info("emailing %s with card %s", "dana@x.com", "4242424242424242")
    assert caplog.records[0].getMessage() == "emailing [EMAIL] with card [CARD]"
