import numpy as np

from ticketrouter.classical import ClassicalTier, reliability_bins
from ticketrouter.taxonomy import INTENT_NAMES


def test_trains_on_all_intents(bench):
    assert set(bench.classical.classes) == set(INTENT_NAMES)
    assert bench.classical.n_train == len(bench.train)


def test_probabilities_are_valid(bench):
    P = bench.classical.predict_proba([t.text for t in bench.test[:50]])
    assert P.shape == (50, 25)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6) and (P >= 0).all()


def test_accuracy_is_reasonable(bench):
    tab = bench.table
    assert (tab.cls_intent == tab.gold_intent).mean() > 0.75


def test_calibration_error_is_small(bench):
    tab = bench.table
    assert reliability_bins(tab.cls_conf, tab.cls_intent == tab.gold_intent)["ece"] < 0.06


def test_confident_predictions_are_more_accurate(bench):
    tab = bench.table
    ok = tab.cls_intent == tab.gold_intent
    hi, lo = tab.cls_conf >= 0.8, tab.cls_conf < 0.5
    assert ok[hi].mean() > ok[lo].mean() + 0.2


def test_reliability_bins_math():
    conf = np.array([0.95, 0.95, 0.95, 0.95, 0.15, 0.15])
    correct = np.array([1, 1, 1, 0, 0, 0])
    out = reliability_bins(conf, correct, n_bins=10)
    top = out["bins"][9]
    assert top["count"] == 4 and top["accuracy"] == 0.75
    assert abs(out["ece"] - (4 / 6 * 0.2 + 2 / 6 * 0.15)) < 1e-4


def test_small_classes_still_train():
    texts = [
        "refund please",
        "refund my money",
        "refund the seats",
        "reset my password",
        "password reset link",
        "forgot password",
        "cancel plan",
        "cancel subscription",
    ]
    intents = ["refund_request"] * 3 + ["password_reset"] * 3 + ["cancellation"] * 2
    m = ClassicalTier().fit(texts * 2, intents * 2, ["normal"] * 16)
    assert m.predict("I need a refund").intent == "refund_request"
