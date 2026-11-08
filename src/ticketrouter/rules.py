"""Tier 1: keyword and regex rules with a confidence score.

Each intent has weighted patterns. The score for an intent is the sum of the weights that match.
Confidence comes from the winning score and its margin over the runner-up, so a ticket that hits
two intents equally gets a low score and moves on to the next tier.

Safety patterns are separate. They do not try to classify, they decide whether a ticket must go to
a human queue right away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .taxonomy import BY_NAME, PRIORITIES

P = re.compile


def _p(pattern: str) -> re.Pattern:
    return P(pattern, re.I)


RULES: dict[str, list[tuple[re.Pattern, float]]] = {
    "billing_dispute": [
        (_p(r"\bdispute\b"), 2.0),
        (_p(r"charge(d)? (is )?(wrong|twice)|charged twice"), 2.0),
        (_p(r"\boverage fee\b"), 1.5),
        (_p(r"\bcontract says\b"), 1.5),
    ],
    "refund_request": [(_p(r"\brefund"), 2.5), (_p(r"money back"), 1.5), (_p(r"\breembolso\b"), 2.0)],
    "invoice_copy": [
        (_p(r"\b(copy of|download|resend).{0,20}invoice"), 2.5),
        (_p(r"\bVAT\b"), 1.5),
        (_p(r"past invoices"), 2.0),
    ],
    "payment_failed": [
        (_p(r"payment (failed|did not go through)"), 2.5),
        (_p(r"\bdeclined\b"), 2.0),
        (_p(r"past due"), 1.5),
    ],
    "plan_upgrade": [(_p(r"\bupgrade\b"), 2.0), (_p(r"\bquote\b"), 1.5), (_p(r"more seats"), 1.5)],
    "cancellation": [(_p(r"\bcancel"), 2.5), (_p(r"auto renewal"), 1.0), (_p(r"close our account"), 2.0)],
    "password_reset": [
        (_p(r"\b(password|pw)\b.{0,15}reset|reset.{0,15}\b(password|pw)\b"), 3.0),
        (_p(r"forgot my password"), 3.0),
        (_p(r"\bcontrasena\b"), 2.0),
    ],
    "account_locked": [
        (_p(r"\blocked (out|after)\b|account is locked"), 2.5),
        (_p(r"\bsuspended\b"), 1.5),
        (_p(r"too many login attempts"), 2.0),
    ],
    "sso_setup": [(_p(r"\bSSO\b"), 2.5), (_p(r"\bSAML\b"), 2.5), (_p(r"single sign-on"), 2.5)],
    "user_provisioning": [(_p(r"\bSCIM\b"), 3.0), (_p(r"provision"), 2.0), (_p(r"in bulk"), 1.0)],
    "api_rate_limit": [(_p(r"\b429\b"), 3.0), (_p(r"rate limit"), 3.0), (_p(r"throttl"), 2.0), (_p(r"\bquota\b"), 1.5)],
    "api_key_issue": [(_p(r"api (key|token)"), 2.5), (_p(r"\b401\b"), 2.0), (_p(r"\brotate\b"), 1.0)],
    "webhook_failure": [(_p(r"\bwebhook"), 3.0)],
    "bug_dashboards": [(_p(r"\bdashboard"), 1.5), (_p(r"\b(widget|tiles?|charts?)\b"), 1.0)],
    "bug_integrations": [(_p(r"\b(connector|integration)\b"), 2.0), (_p(r"\bsync(ing)?\b"), 1.0)],
    "bug_mobile": [(_p(r"\b(mobile app|ios app|android app|mobile)\b"), 2.5), (_p(r"\bcrash"), 1.0)],
    "bug_reporting": [(_p(r"scheduled reports?|report (builder|export)"), 2.5), (_p(r"\bPDF\b"), 1.0)],
    "performance_slow": [(_p(r"\bslow\b|sluggish"), 2.5), (_p(r"time(s)? out|timeout"), 1.0)],
    "feature_request": [(_p(r"feature request|suggestion"), 3.0), (_p(r"would be great|can you add"), 2.0)],
    "data_export": [(_p(r"\bexport"), 2.0), (_p(r"\bCSV\b"), 1.5)],
    "data_deletion": [
        (_p(r"\bGDPR\b|RGPD"), 3.0),
        (_p(r"\b(delete|erase)\b.{0,30}\bdata\b"), 2.0),
        (_p(r"data subject"), 2.5),
    ],
    "security_concern": [(_p(r"compromised|breach|phishing|leaked"), 3.0), (_p(r"suspicious login"), 3.0)],
    "outage_report": [
        (_p(r"\boutage\b|\bis down\b|production is down"), 3.0),
        (_p(r"\b50[03]\b"), 1.5),
        (_p(r"service unavailable|caido"), 2.5),
    ],
    "how_to_question": [(_p(r"^(quick question, )?(how do i|how can i|where do i|what is the)\b"), 1.5)],
    "onboarding_help": [(_p(r"\bonboarding\b|training materials"), 2.5)],
}

# Hard safety rules: any match sends the ticket to a human queue, no model involved.
SAFETY_RULES: dict[str, list[re.Pattern]] = {
    "security_concern": [
        _p(r"\b(compromised|breach|hacked|phishing|leaked|unauthori[sz]ed access)\b"),
        _p(r"suspicious login"),
        _p(r"unknown admin|logins? from .{0,30}(do not|don't) recogni[sz]e"),
        _p(r"\bcomprometida\b"),
    ],
    "outage_report": [
        _p(r"\boutage\b"),
        _p(r"\b(is|are) down\b|production is down|service unavailable"),
        _p(r"\b50[03]\b.{0,40}\b(every|all)\b|\b(every|all)\b.{0,40}\b50[03]\b"),
        _p(r"\bcaido\b"),
    ],
}

URGENT_WORDS = _p(r"\b(urgent|asap|production|blocking|unacceptable|fix this now|immediately)\b")


@dataclass
class RuleResult:
    intent: str | None
    confidence: float
    priority: str
    matched: list[str]
    safety: str | None = None


def safety_check(text: str) -> str | None:
    for intent, patterns in SAFETY_RULES.items():
        if any(p.search(text) for p in patterns):
            return intent
    return None


def rule_priority(text: str, intent: str | None) -> str:

    base = BY_NAME[intent].priority if intent in BY_NAME else "normal"
    if base != "urgent" and URGENT_WORDS.search(text):
        return PRIORITIES[max(0, PRIORITIES.index(base) - 1)]
    return base


def classify(text: str) -> RuleResult:
    scores: dict[str, float] = {}
    matched: list[str] = []
    for intent, patterns in RULES.items():
        s = 0.0
        for pat, w in patterns:
            if pat.search(text):
                s += w
                matched.append(f"{intent}:{pat.pattern}")
        if s:
            scores[intent] = s
    safety = safety_check(text)
    if not scores:
        return RuleResult(None, 0.0, rule_priority(text, None), [], safety)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, top_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (top_s - second_s) / top_s
    strength = min(1.0, top_s / 4.0)
    confidence = round(0.35 + 0.6 * strength * margin, 4)
    return RuleResult(top, confidence, rule_priority(text, top), matched, safety)
