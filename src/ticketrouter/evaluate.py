"""End-to-end evaluation: train tiers, compare single-tier policies with the cascade sweep, pick an
operating point, and break down errors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from . import data
from .cascade import (
    TIERS,
    Policy,
    TierTable,
    build_table,
    choose_operating_point,
    pareto_frontier,
    simulate,
    single_tier,
    summarize,
    sweep,
)
from .classical import ClassicalTier, reliability_bins
from .llm import HAIKU_SIM, OPUS_SIM, SimulatedLLM
from .taxonomy import INTENT_NAMES, SAFETY_INTENTS

DEFAULT_TOLERANCE = 0.01  # accuracy points below LLM-for-everything we accept
DEFAULT_BUDGET = 0.50  # USD per 1,000 tickets


@dataclass
class Bench:
    tickets: list
    train: list
    test: list
    pool: list
    classical: ClassicalTier
    haiku: SimulatedLLM
    opus: SimulatedLLM
    table: TierTable


def train_classical(train: list, **kw) -> ClassicalTier:
    return ClassicalTier(**kw).fit([t.text for t in train], [t.intent for t in train], [t.priority for t in train])


def build_bench(n: int = 20_000, seed: int = 7, haiku=None, opus=None) -> Bench:
    tickets = data.generate(n=n, seed=seed)
    train, test, pool = data.partition(tickets, seed=seed)
    classical = train_classical(train)
    haiku = haiku or SimulatedLLM(HAIKU_SIM)
    opus = opus or SimulatedLLM(OPUS_SIM)
    return Bench(tickets, train, test, pool, classical, haiku, opus, build_table(test, classical, haiku, opus))


def top_confusions(gold: np.ndarray, pred: np.ndarray, texts: list[str], k: int = 8) -> list[dict]:
    pairs = Counter((g, p) for g, p in zip(gold, pred, strict=True) if g != p)
    out = []
    for (g, p), c in pairs.most_common(k):
        idx = [i for i in range(len(gold)) if gold[i] == g and pred[i] == p][:2]
        out.append({"gold": g, "predicted": p, "count": c, "examples": [texts[i][:220] for i in idx]})
    return out


def per_intent(gold: np.ndarray, pred: np.ndarray) -> list[dict]:
    pr, rc, f1, sup = precision_recall_fscore_support(
        gold, pred.astype(str), labels=list(INTENT_NAMES), zero_division=0
    )
    rows = [
        {
            "intent": name,
            "precision": round(float(pr[i]), 3),
            "recall": round(float(rc[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(sup[i]),
        }
        for i, name in enumerate(INTENT_NAMES)
    ]
    return sorted(rows, key=lambda r: r["f1"])


def report(bench: Bench, tolerance: float = DEFAULT_TOLERANCE, budget: float = DEFAULT_BUDGET) -> dict:
    tab = bench.table
    singles = {name: summarize(tab, single_tier(tab, name)) for name in ("rules", "classical", "haiku", "opus")}
    points = sweep(tab)
    frontier = pareto_frontier(points)
    llm_all = singles["opus"]["intent_accuracy"]
    target = round(llm_all - tolerance, 4)
    op = choose_operating_point(frontier, budget)
    policy = Policy(**{k: op["policy"][k] for k in ("t_rules", "t_classical", "t_haiku")})
    res = simulate(tab, policy)
    cascade = summarize(tab, res)

    per_tier = {}
    for i, t in enumerate(TIERS):
        m = res.tier == i
        if m.any():
            per_tier[t] = {
                "share": round(float(m.mean()), 4),
                "accuracy": round(float((res.intent[m] == tab.gold_intent[m]).mean()), 4),
            }

    gold_safe = np.isin(tab.gold_intent, list(SAFETY_INTENTS))
    in_human = np.isin(res.team, ["security_oncall", "incident_response"])
    safety = {
        "safety_tickets": int(gold_safe.sum()),
        "routed_to_human_queue": round(float(in_human[gold_safe].mean()), 4) if gold_safe.any() else None,
        "caught_by_hard_rule": round(float((res.tier[gold_safe] == 0).mean()), 4) if gold_safe.any() else None,
        "hard_rule_latency_ms": 0.2,
        "false_alarm_share": round(float((in_human & ~gold_safe).mean()), 4),
    }
    cls_correct = tab.cls_intent == tab.gold_intent
    texts = [t.text for t in bench.test]
    return {
        "dataset": {
            "tickets": len(bench.tickets),
            "train": len(bench.train),
            "test": len(bench.test),
            "pool": len(bench.pool),
            "intents": len(INTENT_NAMES),
            "label_noise": 0.04,
            "spanish_share": round(sum(t.language == "es" for t in bench.tickets) / len(bench.tickets), 4),
            "multi_intent_share": round(sum(t.secondary is not None for t in bench.tickets) / len(bench.tickets), 4),
        },
        "llm_tiers_simulated": bench.haiku.__class__.__name__ == "SimulatedLLM",
        "single_tier": singles,
        "cascade": cascade | {"policy": policy.to_dict()},
        "operating_point": op | {"target_accuracy": target, "budget_per_1k": budget, "tolerance": tolerance},
        "frontier": frontier,
        "sweep_points": points,
        "per_tier": per_tier,
        "safety": safety,
        "calibration": reliability_bins(tab.cls_conf, cls_correct),
        "per_intent": per_intent(tab.gold_intent, res.intent),
        "top_confusions": top_confusions(tab.gold_intent, res.intent, texts),
    }


def gate(rep: dict) -> tuple[bool, list[str]]:
    """CI gate: the chosen operating point must stay within tolerance of LLM-for-everything accuracy
    and under the cost budget."""
    op = rep["operating_point"]
    llm_all = rep["single_tier"]["opus"]["intent_accuracy"]
    msgs = []
    ok = True
    if op["intent_accuracy"] < llm_all - op["tolerance"]:
        ok = False
        msgs.append(
            f"FAIL accuracy {op['intent_accuracy']:.4f} < LLM-for-everything {llm_all:.4f} "
            f"minus tolerance {op['tolerance']}"
        )
    else:
        msgs.append(f"ok   accuracy {op['intent_accuracy']:.4f} >= {llm_all - op['tolerance']:.4f}")
    if op["cost_per_1k"] > op["budget_per_1k"]:
        ok = False
        msgs.append(f"FAIL cost ${op['cost_per_1k']:.4f} per 1k > budget ${op['budget_per_1k']:.2f}")
    else:
        msgs.append(f"ok   cost ${op['cost_per_1k']:.4f} per 1k <= budget ${op['budget_per_1k']:.2f}")
    return ok, msgs


def markdown_table(rep: dict) -> str:
    rows = [
        ("Rules only", rep["single_tier"]["rules"]),
        ("Classical only", rep["single_tier"]["classical"]),
        ("Haiku for everything (simulated)", rep["single_tier"]["haiku"]),
        ("Opus for everything (simulated)", rep["single_tier"]["opus"]),
        ("Cascade, chosen point", rep["cascade"]),
    ]
    lines = [
        "| Policy | Intent acc | Macro F1 | Priority acc | Team acc | Cost per 1k | LLM share |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rows:
        llm = r["tier_share"].get("haiku", 0) + r["tier_share"].get("opus", 0)
        lines.append(
            f"| {name} | {r['intent_accuracy']:.1%} | {r['intent_macro_f1']:.3f} | "
            f"{r['priority_accuracy']:.1%} | {r['team_accuracy']:.1%} | ${r['cost_per_1k']:.3f} | {llm:.0%} |"
        )
    return "\n".join(lines)
