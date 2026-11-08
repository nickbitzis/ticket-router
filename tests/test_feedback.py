from ticketrouter.cascade import Policy
from ticketrouter.feedback import FeedbackStore, retrain, simulate_weeks, training_set


def test_store_roundtrip_and_latest_wins(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db")
    s.add("T1", "text one", "refund_request", "billing_dispute", "high")
    s.add("T1", "text one", "refund_request", "payment_failed", "high")
    s.add("T2", "text two", None, "sso_setup", "normal", agent="ana")
    assert s.count() == 3
    latest = {r["ticket_id"]: r["intent"] for r in s.latest_per_ticket()}
    assert latest == {"T1": "payment_failed", "T2": "sso_setup"}
    s.mark_used()
    assert all(r["used_in_training"] == 1 for r in s.all())


def test_new_intent_waits_for_enough_examples():
    corr = [{"text": f"pulse forecast flat {i}", "intent": "pulse_forecasting", "priority": "normal"} for i in range(2)]
    texts, intents, _, pending = training_set(["a b c"] * 3, ["how_to_question"] * 3, ["low"] * 3, corr)
    assert pending == {"pulse_forecasting": 2} and "pulse_forecasting" not in intents


def test_new_intent_enters_after_corrections(bench):
    b = bench
    corr = [
        {"text": t, "intent": "pulse_forecasting", "priority": "normal"}
        for t in (
            "Pulse forecasts show flat lines for every metric",
            "Pulse anomaly alerts fire every few minutes",
            "Pulse forecasting model training is stuck in queued state",
            "Pulse confidence bands look too wide",
            "How do we pick the seasonality setting in Pulse",
        )
    ] * 2
    model, pending = retrain(
        [t.text for t in b.train], [t.intent for t in b.train], [t.priority for t in b.train], corr
    )
    assert pending == {} and "pulse_forecasting" in model.classes
    assert model.predict("Our Pulse forecasts are flat lines again").intent == "pulse_forecasting"


def test_corrections_improve_the_classical_tier(bench):
    rows = simulate_weeks(
        bench.train,
        bench.test[:600],
        bench.pool,
        bench.haiku,
        bench.opus,
        Policy(1.01, 0.8, 0.85),
        weeks=3,
        per_week=400,
        seed_size=300,
    )
    assert rows[0]["labeled_examples"] == 300 and rows[-1]["labeled_examples"] > 300
    assert rows[-1]["classical_accuracy"] > rows[0]["classical_accuracy"]
