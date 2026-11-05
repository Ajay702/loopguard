from __future__ import annotations

from loopguard.safety.policy import (
    Decision,
    Policy,
    allow,
    arg_domain_not_in,
    arg_gt,
    arg_matches,
    arg_not_in,
    ask,
    deny,
)


def make() -> Policy:
    return Policy([
        allow("*"),
        deny("send_email", when=arg_domain_not_in("to", {"acme.com"}), reason="external"),
        ask("issue_refund", when=arg_gt("amount", 500), reason="large"),
        ask("delete_*", reason="destructive"),
    ])


def test_most_restrictive_wins_regardless_of_order():
    p = make()
    assert p.evaluate("send_email", {"to": "bob@evil.com"}).decision is Decision.DENY
    assert p.evaluate("send_email", {"to": "ann@acme.com"}).decision is Decision.ALLOW
    assert p.evaluate("send_email", {"to": "ann@eu.acme.com"}).decision is Decision.ALLOW
    assert p.evaluate("send_email", {"to": "ann@notacme.com"}).decision is Decision.DENY


def test_argument_rules_and_globs():
    p = make()
    assert p.evaluate("issue_refund", {"amount": 501}).decision is Decision.ASK
    assert p.evaluate("issue_refund", {"amount": 20}).decision is Decision.ALLOW
    r = p.evaluate("delete_user", {})
    assert r.decision is Decision.ASK and r.reason == "destructive"


def test_predicate_errors_fail_closed():
    p = Policy([deny("pay", when=arg_gt("amount", 10))])
    assert p.evaluate("pay", {"wrong_field": 1}).decision is Decision.DENY


def test_default_decision():
    assert Policy(default=Decision.DENY).evaluate("x", {}).decision is Decision.DENY


def test_other_predicates():
    assert arg_matches("path", r"^/etc/")({"path": "/etc/passwd"})
    assert arg_not_in("to", lambda: {"A@x.com"})({"to": "b@x.com"})
    assert not arg_not_in("to", ["A@x.com"])({"to": "a@x.com"})
    assert arg_domain_not_in("to", ["acme.com"])({"to": ["a@acme.com", "b@gmail.com"]})
