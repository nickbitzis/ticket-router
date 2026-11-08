import json

import pytest

from ticketrouter import data
from ticketrouter.llm import (
    HAIKU_SIM,
    OPUS_SIM,
    OUTPUT_SCHEMA,
    ClaudeTier,
    LLMRefusal,
    SimulatedLLM,
    StructuredOutputError,
    validate_output,
)
from ticketrouter.pricing import LOCAL_COST_PER_TICKET, call_cost

GOOD = {
    "intent": "refund_request",
    "priority": "normal",
    "team": "billing",
    "confidence": 0.82,
    "reason": "Customer asks for money back on removed seats.",
}


def test_validate_accepts_good_reply():
    assert validate_output(dict(GOOD))["intent"] == "refund_request"


@pytest.mark.parametrize(
    "patch",
    [
        {"intent": "make_coffee"},
        {"priority": "critical"},
        {"team": "marketing"},
        {"confidence": 1.4},
        {"confidence": True},
        {"confidence": "high"},
        {"reason": "  "},
    ],
)
def test_validate_rejects_bad_values(patch):
    with pytest.raises(StructuredOutputError):
        validate_output({**GOOD, **patch})


def test_validate_rejects_missing_and_extra_fields():
    with pytest.raises(StructuredOutputError):
        validate_output({k: v for k, v in GOOD.items() if k != "team"})
    with pytest.raises(StructuredOutputError):
        validate_output({**GOOD, "sentiment": "angry"})
    with pytest.raises(StructuredOutputError):
        validate_output(["not", "an", "object"])


def test_claude_request_uses_structured_output(fake_client):
    client = fake_client(json.dumps(GOOD))
    tier = ClaudeTier("claude-haiku-5-5", client=client)
    res = tier.classify("Please refund the seats we removed")
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert "<ticket>" in call["messages"][0]["content"] and "refund_request" in call["system"]
    assert res.intent == "refund_request" and not res.simulated
    assert res.cost == pytest.approx(call_cost("claude-haiku-5-5", 420, 60))


def test_claude_invalid_json_raises(fake_client):
    tier = ClaudeTier("claude-opus-5-5", client=fake_client("not json"))
    with pytest.raises(StructuredOutputError):
        tier.classify("hello")


def test_claude_off_schema_raises(fake_client):
    tier = ClaudeTier("claude-opus-5-5", client=fake_client(json.dumps({**GOOD, "intent": "other"})))
    with pytest.raises(StructuredOutputError):
        tier.classify("hello")


def test_claude_refusal_raises(fake_client):
    tier = ClaudeTier("claude-opus-5-5", client=fake_client(("", "refusal")))
    with pytest.raises(LLMRefusal):
        tier.classify("hello")


def test_pricing_table():
    assert call_cost("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert call_cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert call_cost("claude-opus-5-5", 500, 60) / call_cost("claude-haiku-5-5", 500, 60) == pytest.approx(40.0)
    assert LOCAL_COST_PER_TICKET["rules"] == 0.0


def test_simulated_is_deterministic_and_labeled():
    t = data.generate(n=5, seed=1)[0]
    a = SimulatedLLM(HAIKU_SIM).classify(t.text, gold=t)
    b = SimulatedLLM(HAIKU_SIM).classify(t.text, gold=t)
    assert a == b and a.simulated and a.input_tokens > 300 and 45 <= a.output_tokens <= 90


def test_simulated_accuracy_follows_config():
    ts = data.generate(n=3000, seed=4)
    for cfg in (HAIKU_SIM, OPUS_SIM):
        sim = SimulatedLLM(cfg)
        acc = sum(sim.classify(t.text, gold=t).intent == t.true_intent for t in ts) / len(ts)
        mean_diff = sum(t.difficulty for t in ts) / len(ts)
        expected = cfg.acc_easy - (cfg.acc_easy - cfg.acc_hard) * mean_diff
        assert abs(acc - expected) < 0.025


def test_simulated_free_text_returns_hint():
    r = SimulatedLLM(OPUS_SIM).classify("some new text", gold=None, hint="sso_setup")
    assert r.intent == "sso_setup" and r.team == "identity"
