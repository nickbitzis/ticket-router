"""Drift monitor: notice when a new kind of ticket starts arriving and show it for labeling.

Two signals:
  1. The weekly share of tickets the classical tier is unsure about (calibrated confidence under 0.5,
     a fixed monitor setting separate from the routing threshold). A baseline comes from the first
     weeks; an alarm fires when a week sits more than `z` binomial standard errors above it.
  2. Clustering of the unsure tickets from the alarm onward. A cluster whose top terms are rare or
     absent in the training data is likely a new intent, not just hard tickets.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .classical import ClassicalTier


def weekly_low_confidence(conf: np.ndarray, weeks: np.ndarray, threshold: float) -> list[dict]:
    rows = []
    for w in sorted(set(weeks.tolist())):
        m = weeks == w
        rows.append(
            {"week": int(w), "n": int(m.sum()), "low_conf_share": round(float((conf[m] < threshold).mean()), 4)}
        )
    return rows


def alarms(rows: list[dict], baseline_weeks: int = 4, z: float = 3.0) -> tuple[float, float, list[dict]]:
    base = [r["low_conf_share"] for r in rows[:baseline_weeks]]
    p0 = float(np.mean(base))
    out = []
    for r in rows:
        se = float(np.sqrt(max(p0 * (1 - p0), 1e-6) / max(r["n"], 1)))
        limit = p0 + z * se
        out.append(
            {**r, "limit": round(limit, 4), "alarm": r["week"] >= baseline_weeks and r["low_conf_share"] > limit}
        )
    return p0, z, out


def cluster_unknowns(
    texts: list[str], train_texts: list[str], k: int = 6, seed: int = 7, rare_df: float = 0.002, top_n: int = 8
) -> list[dict]:
    """Cluster unsure tickets and score each cluster by how unfamiliar its top terms are."""
    if len(texts) < k * 3:
        return []
    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, ngram_range=(1, 1))
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)
    terms = np.array(vec.get_feature_names_out())
    cv = CountVectorizer(binary=True, vocabulary=terms.tolist())
    train_df = np.asarray(cv.fit_transform(train_texts).mean(axis=0)).ravel()
    clusters = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        top = np.argsort(-km.cluster_centers_[c])[:top_n]
        novelty = float((train_df[top] < rare_df).mean())
        clusters.append(
            {
                "cluster": c,
                "size": int(len(idx)),
                "top_terms": terms[top].tolist(),
                "novelty": round(novelty, 3),
                "score": round(novelty * len(idx), 2),
                "member_index": idx.tolist(),
                "examples": [texts[i][:200] for i in idx[:3]],
            }
        )
    return sorted(clusters, key=lambda c: -c["score"])


def detect(
    classical: ClassicalTier,
    stream: list,
    train_texts: list[str],
    threshold: float = 0.5,
    baseline_weeks: int = 4,
    z: float = 3.0,
    min_novelty: float = 0.4,
    min_size: int = 15,
    max_cluster_tickets: int = 800,
) -> dict:
    texts = [t.text for t in stream]
    _, conf, _ = classical.predict_many(texts)
    weeks = np.array([t.week for t in stream])
    p0, z, rows = alarms(weekly_low_confidence(conf, weeks, threshold), baseline_weeks, z)
    first = next((r["week"] for r in rows if r["alarm"]), None)
    surfaced: list[dict] = []
    clusters: list[dict] = []
    if first is not None:
        pick = [i for i in range(len(stream)) if weeks[i] >= first and conf[i] < threshold][:max_cluster_tickets]
        clusters = cluster_unknowns([texts[i] for i in pick], train_texts)
        for c in clusters:
            members = [stream[pick[j]] for j in c.pop("member_index")]
            c["ticket_ids"] = [m.id for m in members[-50:]]  # most recent first in the queue
            # Hidden labels, for evaluating the monitor only. The service never shows these as truth.
            counts = Counter(m.true_intent for m in members)
            label, n = counts.most_common(1)[0]
            c["hidden_majority_intent"] = label
            c["hidden_purity"] = round(n / len(members), 3)
        surfaced = [c for c in clusters if c["novelty"] >= min_novelty and c["size"] >= min_size]
    return {
        "threshold": threshold,
        "baseline_share": round(p0, 4),
        "z": z,
        "weeks": rows,
        "first_alarm_week": first,
        "clusters": clusters,
        "surfaced": surfaced,
    }
