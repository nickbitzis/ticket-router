import random

from ticketrouter import data
from ticketrouter.taxonomy import DRIFT_INTENT, INTENT_NAMES, SAFETY_INTENTS


def test_generation_is_seeded():
    a = data.generate(n=300, seed=1)
    b = data.generate(n=300, seed=1)
    c = data.generate(n=300, seed=2)
    assert [t.text for t in a] == [t.text for t in b]
    assert [t.text for t in a] != [t.text for t in c]


def test_dataset_shape_and_variety():
    ts = data.generate(n=4000, seed=7)
    assert len(ts) == 4000
    assert {t.intent for t in ts} <= set(INTENT_NAMES)
    assert len({t.true_intent for t in ts}) == 25
    share = lambda f: sum(map(f, ts)) / len(ts)  # noqa: E731
    assert 0.02 < share(lambda t: t.language == "es") < 0.1
    assert 0.08 < share(lambda t: t.secondary is not None) < 0.3
    assert 0.02 < share(lambda t: t.intent != t.true_intent) < 0.07  # label noise near 4 percent
    assert min(len(t.text) for t in ts) < 40 < max(len(t.text) for t in ts) / 5


def test_safety_secondary_becomes_primary_label():
    ts = data.generate(n=4000, seed=7)
    for t in ts:
        if t.secondary in SAFETY_INTENTS:
            assert t.true_intent in SAFETY_INTENTS


def test_typos_change_text_but_keep_length_close():
    rng = random.Random(0)
    s = "Our payment failed and the account shows a past due banner"
    t = data.add_typos(s, rate=1.0, rng=rng)
    assert t != s and abs(len(t) - len(s)) <= len(s.split())


def test_partition_is_disjoint_and_sized():
    ts = data.generate(n=1000, seed=7)
    train, test, pool = data.partition(ts)
    assert (len(train), len(test), len(pool)) == (200, 300, 500)
    ids = [t.id for t in train + test + pool]
    assert len(ids) == len(set(ids)) == 1000


def test_drift_stream_launches_new_intent_mid_stream():
    ts = data.generate(n=3000, seed=7)
    stream = data.drift_stream(ts, per_week=200, weeks=12, launch_week=7)
    by_week = {w: [t for t in stream if t.week == w] for w in range(12)}
    assert all(len(v) == 200 for v in by_week.values())
    assert not any(t.true_intent == DRIFT_INTENT.name for w in range(7) for t in by_week[w])
    late = sum(t.true_intent == DRIFT_INTENT.name for t in by_week[11]) / 200
    assert 0.15 <= late <= 0.25
