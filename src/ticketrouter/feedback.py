"""Agent corrections: stored in SQLite, used to retrain the classical tier.

`FeedbackStore` is what the API writes to when an agent fixes a routing decision in the UI.
`simulate_weeks` plays the same loop forward over the unlabeled pool: route a week of tickets,
let agents fix a share of the mistakes and confirm a share of the right answers, retrain, repeat.
"""

from __future__ import annotations

import random
import sqlite3
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .cascade import Policy, build_table, simulate, summarize
from .classical import ClassicalTier, reliability_bins

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    text TEXT NOT NULL,
    predicted_intent TEXT,
    intent TEXT NOT NULL,
    priority TEXT NOT NULL,
    agent TEXT NOT NULL,
    created_at TEXT NOT NULL,
    used_in_training INTEGER NOT NULL DEFAULT 0
);
"""


class FeedbackStore:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(
        self, ticket_id: str, text: str, predicted_intent: str | None, intent: str, priority: str, agent: str = "agent"
    ) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO corrections (ticket_id, text, predicted_intent, intent, priority, agent, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ticket_id,
                    text,
                    predicted_intent,
                    intent,
                    priority,
                    agent,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            self.db.commit()
            return int(cur.lastrowid)

    def all(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM corrections ORDER BY id")]

    def latest_per_ticket(self) -> list[dict]:
        """One row per ticket, the most recent correction wins."""
        rows: dict[str, dict] = {}
        for r in self.all():
            rows[r["ticket_id"]] = r
        return list(rows.values())

    def mark_used(self) -> None:
        with self.lock:
            self.db.execute("UPDATE corrections SET used_in_training = 1")
            self.db.commit()

    def count(self) -> int:
        with self.lock:
            return int(self.db.execute("SELECT COUNT(*) FROM corrections").fetchone()[0])


MIN_EXAMPLES = 3  # calibration needs a few examples per class


def training_set(
    base_texts: list[str], base_intents: list[str], base_priorities: list[str], corrections: list[dict]
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Original labels plus agent corrections. A correction to an intent the model has never seen
    (a new product, say) is how a new class enters the taxonomy, but only once it has
    MIN_EXAMPLES examples. Until then it waits and is reported as pending."""
    texts = list(base_texts) + [c["text"] for c in corrections]
    intents = list(base_intents) + [c["intent"] for c in corrections]
    prios = list(base_priorities) + [c["priority"] for c in corrections]
    counts = Counter(intents)
    pending = {k: v for k, v in counts.items() if v < MIN_EXAMPLES}
    keep = [i for i, x in enumerate(intents) if x not in pending]
    return [texts[i] for i in keep], [intents[i] for i in keep], [prios[i] for i in keep], pending


def retrain(
    base_texts: list[str], base_intents: list[str], base_priorities: list[str], corrections: list[dict]
) -> tuple[ClassicalTier, dict[str, int]]:
    texts, intents, prios, pending = training_set(base_texts, base_intents, base_priorities, corrections)
    return ClassicalTier().fit(texts, intents, prios), pending


def simulate_weeks(
    train: list,
    holdout: list,
    pool: list,
    haiku,
    opus,
    policy: Policy,
    weeks: int = 8,
    per_week: int = 1250,
    seed_size: int = 1000,
    correction_rate: float = 0.7,
    confirm_rate: float = 0.3,
    seed: int = 5,
) -> list[dict]:
    """Start from a small labeled seed and let a week of agent feedback flow back each week."""
    rng = random.Random(seed)
    seed_set = train[:seed_size]
    labeled = [{"text": t.text, "intent": t.intent, "priority": t.priority} for t in seed_set]
    rows = []
    classical, _ = retrain([], [], [], labeled)
    for week in range(weeks + 1):
        tab = build_table(holdout, classical, haiku, opus)
        res = simulate(tab, policy)
        s = summarize(tab, res)
        correct = tab.cls_intent == tab.gold_intent
        rows.append(
            {
                "week": week,
                "labeled_examples": len(labeled),
                "classical_accuracy": round(float(correct.mean()), 4),
                "classical_ece": reliability_bins(tab.cls_conf, correct)["ece"],
                "cascade_accuracy": s["intent_accuracy"],
                "cascade_cost_per_1k": s["cost_per_1k"],
                "llm_share": round(s["tier_share"]["haiku"] + s["tier_share"]["opus"], 4),
            }
        )
        if week == weeks:
            break
        batch = pool[week * per_week : (week + 1) * per_week]
        if not batch:
            break
        btab = build_table(batch, classical, haiku, opus)
        bres = simulate(btab, policy)
        wrong = bres.intent != btab.gold_intent
        for t, w in zip(batch, np.asarray(wrong), strict=True):
            if (w and rng.random() < correction_rate) or (not w and rng.random() < confirm_rate):
                labeled.append({"text": t.text, "intent": t.intent, "priority": t.priority})
        classical, _ = retrain([], [], [], labeled)
    return rows
