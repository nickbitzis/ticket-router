"""The cascade: cheap tiers first, escalate only when unsure, hard safety rules before any model.

Order for one ticket:
  0. Safety rules. A security or outage pattern sends the ticket to a human queue right away.
  1. Rules. Accept if rule confidence >= t_rules.
  2. Classical model. Accept if calibrated confidence >= t_classical.
  3. Haiku. Accept if its confidence >= t_haiku (skip this step when t_haiku is None).
  4. Opus. Always accepted, unless it refuses or replies off-schema, then a human triages.
After any tier: if the final intent is a safety intent, the ticket still goes to the human queue.

Cost is cumulative. A ticket that ends at Opus also paid for the Haiku call before it.

`route` handles one live ticket. `simulate` runs the same policy over a precomputed table of tier
outputs with numpy, which is what makes a sweep over hundreds of threshold settings take seconds.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import rules
from .classical import ClassicalTier
from .llm import UNKNOWN, LLMClassifier, LLMRefusal, StructuredOutputError
from .pricing import LOCAL_COST_PER_TICKET
from .taxonomy import HUMAN_QUEUES, SAFETY_INTENTS, team_for

TIERS = ("safety", "rules", "classical", "haiku", "opus", "human")
OFF = 1.01  # a threshold no confidence can reach, which turns a tier off
LOCAL_LATENCY_MS = {"rules": 0.2, "classical": 2.0}
HUMAN_TRIAGE = "human_triage"


@dataclass(frozen=True)
class Policy:
    t_rules: float = 0.9
    t_classical: float = 0.8
    t_haiku: float | None = 0.85
    name: str = ""

    def label(self) -> str:
        h = "skip" if self.t_haiku is None else f"{self.t_haiku:.2f}"
        r = "off" if self.t_rules >= OFF else f"{self.t_rules:.2f}"
        return self.name or f"rules {r}, classical {self.t_classical:.2f}, haiku {h}"

    def to_dict(self) -> dict:
        return asdict(self) | {"label": self.label()}


@dataclass
class Decision:
    tier: str
    intent: str
    priority: str
    team: str
    confidence: float
    cost: float
    latency_ms: float
    reason: str
    safety: bool = False
    simulated_llm: bool = False
    path: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self) | {"cost": round(self.cost, 8)}


def _finalize(d: Decision) -> Decision:
    if d.intent in SAFETY_INTENTS:
        d.safety = True
        d.team = HUMAN_QUEUES[d.intent]
        d.priority = "urgent"
    return d


class Router:
    def __init__(self, classical: ClassicalTier, haiku: LLMClassifier, opus: LLMClassifier, policy: Policy):
        self.classical = classical
        self.haiku = haiku
        self.opus = opus
        self.policy = policy

    def route(self, text: str, gold=None) -> Decision:
        start = time.perf_counter()
        p = self.policy
        path: list[dict] = []
        cost = 0.0
        llm_latency = 0.0

        def done(d: Decision) -> Decision:
            d.path = path
            d.cost = cost
            d.latency_ms = round((time.perf_counter() - start) * 1000 + llm_latency, 2)
            return _finalize(d)

        rr = rules.classify(text)
        if rr.safety:
            path.append({"tier": "safety", "intent": rr.safety, "confidence": 1.0, "cost": 0.0})
            return done(
                Decision(
                    "safety",
                    rr.safety,
                    "urgent",
                    HUMAN_QUEUES[rr.safety],
                    1.0,
                    0,
                    0,
                    f"hard rule: {rr.safety.replace('_', ' ')} pattern, sent to a human queue",
                )
            )
        path.append({"tier": "rules", "intent": rr.intent, "confidence": rr.confidence, "cost": 0.0})
        if rr.intent and rr.confidence >= p.t_rules:
            return done(
                Decision(
                    "rules",
                    rr.intent,
                    rr.priority,
                    team_for(rr.intent),
                    rr.confidence,
                    0,
                    0,
                    f"keyword rules matched {len(rr.matched)} pattern(s)",
                )
            )

        cp = self.classical.predict(text)
        cost += LOCAL_COST_PER_TICKET["classical"]
        path.append(
            {
                "tier": "classical",
                "intent": cp.intent,
                "confidence": round(cp.confidence, 4),
                "cost": LOCAL_COST_PER_TICKET["classical"],
            }
        )
        if cp.confidence >= p.t_classical:
            return done(
                Decision(
                    "classical",
                    cp.intent,
                    cp.priority,
                    team_for(cp.intent),
                    cp.confidence,
                    0,
                    0,
                    f"calibrated model, top alternatives {cp.probs}",
                )
            )

        hint = cp.intent
        for tier, model, threshold in (("haiku", self.haiku, p.t_haiku), ("opus", self.opus, 0.0)):
            if threshold is None:
                continue
            try:
                res = model.classify(text, gold=gold, hint=hint)
            except (LLMRefusal, StructuredOutputError) as exc:
                path.append({"tier": tier, "error": str(exc), "cost": 0.0})
                if tier == "opus":
                    return done(
                        Decision(
                            "human",
                            hint,
                            cp.priority,
                            HUMAN_TRIAGE,
                            0.0,
                            0,
                            0,
                            f"{tier} could not classify ({type(exc).__name__}), sent to triage",
                        )
                    )
                continue
            cost += res.cost
            llm_latency += res.latency_ms
            path.append(
                {
                    "tier": tier,
                    "intent": res.intent,
                    "confidence": res.confidence,
                    "cost": round(res.cost, 8),
                    "latency_ms": res.latency_ms,
                    "simulated": res.simulated,
                }
            )
            if tier == "opus" or res.confidence >= threshold:
                team = HUMAN_TRIAGE if res.intent == UNKNOWN else team_for(res.intent)
                return done(
                    Decision(
                        tier,
                        res.intent,
                        res.priority,
                        team,
                        res.confidence,
                        0,
                        0,
                        res.reason,
                        simulated_llm=res.simulated,
                    )
                )
            hint = res.intent
        raise AssertionError("unreachable: opus always decides")  # pragma: no cover


# Batch simulation over precomputed tier outputs


@dataclass
class TierTable:
    """Every tier's answer for every ticket in an evaluation set, computed once."""

    gold_intent: np.ndarray
    gold_priority: np.ndarray
    gold_team: np.ndarray
    safety_intent: np.ndarray  # "" when no safety rule fired
    rules_intent: np.ndarray
    rules_conf: np.ndarray
    rules_priority: np.ndarray
    cls_intent: np.ndarray
    cls_conf: np.ndarray
    cls_priority: np.ndarray
    haiku_intent: np.ndarray
    haiku_conf: np.ndarray
    haiku_priority: np.ndarray
    haiku_cost: np.ndarray
    haiku_latency: np.ndarray
    opus_intent: np.ndarray
    opus_conf: np.ndarray
    opus_priority: np.ndarray
    opus_cost: np.ndarray
    opus_latency: np.ndarray

    def __len__(self) -> int:
        return len(self.gold_intent)


def build_table(tickets, classical: ClassicalTier, haiku: LLMClassifier, opus: LLMClassifier) -> TierTable:
    texts = [t.text for t in tickets]
    rr = [rules.classify(x) for x in texts]
    ci, cc, cpr = classical.predict_many(texts)
    hr = [haiku.classify(t.text, gold=t, hint=ci[k]) for k, t in enumerate(tickets)]
    orr = [opus.classify(t.text, gold=t, hint=ci[k]) for k, t in enumerate(tickets)]
    a = np.asarray
    return TierTable(
        gold_intent=a([t.intent for t in tickets]),
        gold_priority=a([t.priority for t in tickets]),
        gold_team=a([t.team for t in tickets]),
        safety_intent=a([r.safety or "" for r in rr]),
        rules_intent=a([r.intent or "" for r in rr]),
        rules_conf=a([r.confidence for r in rr], dtype=float),
        rules_priority=a([r.priority for r in rr]),
        cls_intent=a(ci),
        cls_conf=a(cc, dtype=float),
        cls_priority=a(cpr),
        haiku_intent=a([r.intent for r in hr]),
        haiku_conf=a([r.confidence for r in hr], dtype=float),
        haiku_priority=a([r.priority for r in hr]),
        haiku_cost=a([r.cost for r in hr], dtype=float),
        haiku_latency=a([r.latency_ms for r in hr], dtype=float),
        opus_intent=a([r.intent for r in orr]),
        opus_conf=a([r.confidence for r in orr], dtype=float),
        opus_priority=a([r.priority for r in orr]),
        opus_cost=a([r.cost for r in orr], dtype=float),
        opus_latency=a([r.latency_ms for r in orr], dtype=float),
    )


@dataclass
class SimResult:
    intent: np.ndarray
    priority: np.ndarray
    team: np.ndarray
    tier: np.ndarray  # index into TIERS
    cost: np.ndarray
    latency: np.ndarray


_TEAM_OF = np.vectorize(team_for, otypes=[object])


def _teams(intents: np.ndarray) -> np.ndarray:
    teams = _TEAM_OF(intents).astype(object)
    for intent, queue in HUMAN_QUEUES.items():
        teams[intents == intent] = queue
    teams[intents == UNKNOWN] = HUMAN_TRIAGE
    return teams


def simulate(tab: TierTable, policy: Policy, safety: bool = True) -> SimResult:
    n = len(tab)
    safe = (tab.safety_intent != "") if safety else np.zeros(n, bool)
    at_rules = ~safe & (tab.rules_intent != "") & (tab.rules_conf >= policy.t_rules)
    left = ~safe & ~at_rules
    at_cls = left & (tab.cls_conf >= policy.t_classical)
    left &= ~at_cls
    if policy.t_haiku is None:
        at_haiku = np.zeros(n, bool)
        reached_haiku = np.zeros(n, bool)
    else:
        reached_haiku = left.copy()
        at_haiku = left & (tab.haiku_conf >= policy.t_haiku)
    at_opus = left & ~at_haiku

    tier = np.select([safe, at_rules, at_cls, at_haiku, at_opus], [0, 1, 2, 3, 4], default=4)
    intent = np.select(
        [safe, at_rules, at_cls, at_haiku],
        [tab.safety_intent, tab.rules_intent, tab.cls_intent, tab.haiku_intent],
        default=tab.opus_intent,
    ).astype(object)
    priority = np.select(
        [at_rules, at_cls, at_haiku, at_opus],
        [tab.rules_priority, tab.cls_priority, tab.haiku_priority, tab.opus_priority],
        default="urgent",
    ).astype(object)
    priority[np.isin(intent, list(SAFETY_INTENTS))] = "urgent"
    reached_cls = ~safe & ~at_rules
    cost = reached_cls * LOCAL_COST_PER_TICKET["classical"] + reached_haiku * tab.haiku_cost + at_opus * tab.opus_cost
    latency = (
        LOCAL_LATENCY_MS["rules"]
        + reached_cls * LOCAL_LATENCY_MS["classical"]
        + reached_haiku * tab.haiku_latency
        + at_opus * tab.opus_latency
    )
    return SimResult(intent, priority, _teams(intent), tier, cost, latency)


def single_tier(tab: TierTable, tier: str) -> SimResult:
    """Baselines with no cascade and no safety rules: one tier answers every ticket."""
    n = len(tab)
    zeros = np.zeros(n)
    if tier == "rules":
        intent = np.where(tab.rules_intent != "", tab.rules_intent, "how_to_question").astype(object)
        return SimResult(
            intent,
            tab.rules_priority.astype(object),
            _teams(intent),
            np.full(n, 1),
            zeros,
            zeros + LOCAL_LATENCY_MS["rules"],
        )
    if tier == "classical":
        return SimResult(
            tab.cls_intent.astype(object),
            tab.cls_priority.astype(object),
            _teams(tab.cls_intent),
            np.full(n, 2),
            zeros + LOCAL_COST_PER_TICKET["classical"],
            zeros + LOCAL_LATENCY_MS["classical"],
        )
    if tier == "haiku":
        return SimResult(
            tab.haiku_intent.astype(object),
            tab.haiku_priority.astype(object),
            _teams(tab.haiku_intent),
            np.full(n, 3),
            tab.haiku_cost,
            tab.haiku_latency,
        )
    if tier == "opus":
        return SimResult(
            tab.opus_intent.astype(object),
            tab.opus_priority.astype(object),
            _teams(tab.opus_intent),
            np.full(n, 4),
            tab.opus_cost,
            tab.opus_latency,
        )
    raise ValueError(tier)


def summarize(tab: TierTable, res: SimResult) -> dict:
    from sklearn.metrics import f1_score

    n = len(tab)
    shares = {t: round(float((res.tier == i).mean()), 4) for i, t in enumerate(TIERS) if t != "human"}
    return {
        "intent_accuracy": round(float((res.intent == tab.gold_intent).mean()), 4),
        "intent_macro_f1": round(
            float(f1_score(tab.gold_intent, res.intent.astype(str), average="macro", zero_division=0)), 4
        ),
        "priority_accuracy": round(float((res.priority == tab.gold_priority).mean()), 4),
        "team_accuracy": round(float((res.team == tab.gold_team).mean()), 4),
        "cost_per_1k": round(float(res.cost.sum() / n * 1000), 4),
        "latency_mean_ms": round(float(res.latency.mean()), 1),
        "latency_p95_ms": round(float(np.percentile(res.latency, 95)), 1),
        "tier_share": shares,
        "n": n,
    }


def default_grid() -> list[Policy]:
    rules_t = (0.8, 0.9, 0.95, OFF)
    cls_t = tuple(round(x, 2) for x in np.arange(0.4, 1.0, 0.05)) + (0.97, 0.99, OFF)
    haiku_t: tuple[float | None, ...] = (None, 0.0, 0.7, 0.8, 0.85, 0.9, 0.95, OFF)
    return [Policy(r, c, h) for r in rules_t for c in cls_t for h in haiku_t]


def sweep(tab: TierTable, grid: list[Policy] | None = None) -> list[dict]:
    out = []
    for pol in grid or default_grid():
        res = simulate(tab, pol)
        n = len(tab)
        out.append(
            {
                "policy": pol.to_dict(),
                "intent_accuracy": round(float((res.intent == tab.gold_intent).mean()), 4),
                "cost_per_1k": round(float(res.cost.sum() / n * 1000), 4),
                "llm_share": round(float((res.tier >= 3).mean()), 4),
            }
        )
    return out


def pareto_frontier(points: list[dict], x: str = "cost_per_1k", y: str = "intent_accuracy") -> list[dict]:
    """Points no other point beats on both cost (lower) and accuracy (higher). Sorted by cost."""
    ordered = sorted(points, key=lambda p: (p[x], -p[y]))
    front: list[dict] = []
    best = -np.inf
    for p in ordered:
        if p[y] > best:
            front.append(p)
            best = p[y]
    return front


def choose_operating_point(frontier: list[dict], budget_per_1k: float, slack: float = 0.0025) -> dict:
    """Best accuracy the budget can buy, then the cheapest frontier point within `slack` of it.

    The slack stops the choice from paying a lot more for a gain smaller than the noise between
    test sets. If nothing fits the budget, the cheapest point is returned, marked over budget.
    """
    within = [p for p in frontier if p["cost_per_1k"] <= budget_per_1k]
    if not within:
        return {**min(frontier, key=lambda p: p["cost_per_1k"]), "within_budget": False}
    best = max(p["intent_accuracy"] for p in within)
    near = [p for p in within if p["intent_accuracy"] >= best - slack]
    return {**min(near, key=lambda p: p["cost_per_1k"]), "within_budget": True}
