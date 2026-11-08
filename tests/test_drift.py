from ticketrouter import drift
from ticketrouter.taxonomy import DRIFT_INTENT


def test_alarm_rule_on_synthetic_series():
    rows = [{"week": w, "n": 500, "low_conf_share": s} for w, s in enumerate([0.1, 0.11, 0.09, 0.1, 0.1, 0.18])]
    p0, _, out = drift.alarms(rows, baseline_weeks=4, z=3.0)
    assert abs(p0 - 0.1) < 1e-9
    assert [r["alarm"] for r in out] == [False] * 5 + [True]


def test_baseline_weeks_never_alarm():
    rows = [{"week": w, "n": 100, "low_conf_share": s} for w, s in enumerate([0.05, 0.4, 0.05, 0.05])]
    _, _, out = drift.alarms(rows, baseline_weeks=4)
    assert not any(r["alarm"] for r in out)


def test_detects_launch_without_false_alarms(drift_setup):
    model, stream, train = drift_setup
    res = drift.detect(model, stream, [t.text for t in train], threshold=0.5)
    assert res["first_alarm_week"] is not None and 7 <= res["first_alarm_week"] <= 8
    assert not any(r["alarm"] for r in res["weeks"] if r["week"] < 7)


def test_surfaced_cluster_is_the_new_intent(drift_setup):
    model, stream, train = drift_setup
    res = drift.detect(model, stream, [t.text for t in train], threshold=0.5)
    assert res["surfaced"], "expected at least one unfamiliar cluster"
    top = res["surfaced"][0]
    assert top["hidden_majority_intent"] == DRIFT_INTENT.name and top["hidden_purity"] >= 0.7
    assert top["novelty"] >= 0.4 and top["ticket_ids"]


def test_cluster_novelty_scores_unseen_terms_higher():
    train = ["refund the invoice charge"] * 50 + ["reset my password link"] * 50
    texts = (
        ["pulse forecast seasonality bands"] * 10
        + ["pulse anomaly forecast alerts"] * 10
        + ["refund invoice charge please"] * 10
        + ["password reset link please"] * 10
    )
    clusters = drift.cluster_unknowns(texts, train, k=3, seed=1)
    assert any("pulse" in c["top_terms"] for c in clusters[:1])
    assert clusters[0]["novelty"] > clusters[-1]["novelty"]
