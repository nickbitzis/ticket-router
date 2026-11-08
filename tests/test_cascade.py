import json

import numpy as np
import pytest

from ticketrouter.cascade import OFF, TIERS, Policy, Router, simulate, single_tier, summarize
from ticketrouter.classical import Prediction
from ticketrouter.llm import LLMRefusal, LLMResult, StructuredOutputError


class StubClassical:
    def __init__(self, intent="refund_request", conf=0.5):
        self.intent, self.conf, self.calls = intent, conf, 0

    def predict(self, text):
        self.calls += 1
        return Prediction(self.intent, self.conf, "normal", {self.intent: self.conf})


class StubLLM:
    def __init__(self, model, intent="billing_dispute", conf=0.9, error=None):
        self.model, self.intent, self.conf, self.error, self.calls = model, intent, conf, error, 0

    def classify(self, text, gold=None, hint=None):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResult(self.intent, "high", "billing", self.conf, "stub", self.model, 400, 60, 100.0, True)


def make(policy=None, cls_conf=0.5, haiku_conf=0.9, haiku_error=None, opus_error=None, opus_intent="billing_dispute"):
    c = StubClassical(conf=cls_conf)
    h = StubLLM("claude-haiku-5-5", conf=haiku_conf, error=haiku_error)
    o = StubLLM("claude-opus-5-5", intent=opus_intent, error=opus_error)
    return Router(c, h, o, policy or Policy(t_rules=0.9, t_classical=0.8, t_haiku=0.85)), c, h, o


def test_rules_tier_accepts_confident_match():
    r, c, h, o = make()
    d = r.route("I forgot my password and the reset email never arrives.")
    assert d.tier == "rules" and d.intent == "password_reset" and c.calls == 0 and d.cost == 0


def test_classical_tier_accepts_above_threshold():
    r, c, h, o = make(cls_conf=0.95)
    d = r.route("The money for those months, can it come back?")
    assert d.tier == "classical" and h.calls == 0 and d.cost < 1e-6


def test_escalates_to_haiku_when_classical_unsure():
    r, c, h, o = make(cls_conf=0.5, haiku_conf=0.9)
    d = r.route("The money for those months, can it come back?")
    assert d.tier == "haiku" and h.calls == 1 and o.calls == 0


def test_escalates_to_opus_when_haiku_unsure_and_cost_is_cumulative():
    r, c, h, o = make(cls_conf=0.5, haiku_conf=0.5)
    d = r.route("The money for those months, can it come back?")
    assert d.tier == "opus" and h.calls == 1 and o.calls == 1
    haiku_cost = (400 * 0.10 + 60 * 0.50) / 1e6
    opus_cost = (400 * 4 + 60 * 20) / 1e6
    assert d.cost == pytest.approx(haiku_cost + opus_cost + 6e-8)
    assert [p["tier"] for p in d.path] == ["rules", "classical", "haiku", "opus"]


def test_skip_haiku_goes_straight_to_opus():
    r, c, h, o = make(policy=Policy(0.9, 0.8, None), cls_conf=0.5)
    d = r.route("The money for those months, can it come back?")
    assert d.tier == "opus" and h.calls == 0


def test_rules_off_threshold_never_accepts():
    r, c, h, o = make(policy=Policy(OFF, 0.8, 0.85), cls_conf=0.95)
    d = r.route("I forgot my password and the reset email never arrives.")
    assert d.tier == "classical"


def test_haiku_bad_output_falls_through_to_opus():
    r, c, h, o = make(haiku_error=StructuredOutputError("bad"))
    d = r.route("The money for those months, can it come back?")
    assert d.tier == "opus" and "error" in d.path[2]


def test_opus_refusal_goes_to_human_triage():
    r, c, h, o = make(haiku_conf=0.1, opus_error=LLMRefusal("no"))
    d = r.route("The money for those months, can it come back?")
    assert d.tier == "human" and d.team == "human_triage"


def test_safety_rule_skips_every_model():
    r, c, h, o = make()
    d = r.route("Possible security breach: an unknown admin user was created in our workspace.")
    assert d.tier == "safety" and d.safety and d.team == "security_oncall" and d.priority == "urgent"
    assert c.calls == h.calls == o.calls == 0 and d.cost == 0


def test_model_predicting_outage_still_goes_to_human_queue():
    r, c, h, o = make(cls_conf=0.1, haiku_conf=0.1, opus_intent="outage_report")
    d = r.route("Nothing loads for anyone in the office")
    assert d.tier == "opus" and d.safety and d.team == "incident_response" and d.priority == "urgent"


def test_decision_serializes():
    r, *_ = make()
    assert json.loads(json.dumps(r.route("pw reset pls").to_dict()))["tier"] in TIERS


def test_simulate_matches_live_router(bench):
    pol = Policy(0.9, 0.8, 0.85)
    res = simulate(bench.table, pol)
    router = Router(bench.classical, bench.haiku, bench.opus, pol)
    for i, t in enumerate(bench.test[:120]):
        d = router.route(t.text, gold=t)
        assert TIERS[res.tier[i]] == d.tier and res.intent[i] == d.intent
        assert res.cost[i] == pytest.approx(d.cost)


def test_higher_threshold_sends_more_to_llm(bench):
    shares = [float((simulate(bench.table, Policy(OFF, t, 0.85)).tier >= 3).mean()) for t in (0.5, 0.7, 0.9, 0.99)]
    assert shares == sorted(shares) and shares[-1] > shares[0]


def test_safety_override_can_be_measured(bench):
    with_rules = simulate(bench.table, Policy(OFF, 0.8, 0.85), safety=True)
    without = simulate(bench.table, Policy(OFF, 0.8, 0.85), safety=False)
    assert (with_rules.tier == 0).sum() > 0 and (without.tier == 0).sum() == 0


def test_cost_per_tier_accounting(bench):
    res = simulate(bench.table, Policy(OFF, 0.8, 0.85))
    tab = bench.table
    at_opus = res.tier == 4
    assert np.allclose(res.cost[at_opus], tab.haiku_cost[at_opus] + tab.opus_cost[at_opus] + 6e-8)
    assert (res.cost[res.tier == 0] == 0).all()


def test_single_tier_baselines(bench):
    s = {k: summarize(bench.table, single_tier(bench.table, k)) for k in ("rules", "classical", "haiku", "opus")}
    assert s["opus"]["cost_per_1k"] > s["haiku"]["cost_per_1k"] * 20 > s["classical"]["cost_per_1k"]
    assert s["opus"]["tier_share"]["opus"] == 1.0
    assert s["rules"]["intent_accuracy"] < s["classical"]["intent_accuracy"]
