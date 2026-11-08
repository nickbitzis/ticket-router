"""Tier 2: TF-IDF features and a linear model, with calibrated probabilities.

Word n-grams catch phrasing, character n-grams make it tolerant of typos and give it some grip on
Spanish. The intent model is a linear SVM (`linear`, the default, trains in about 12 seconds on
14,000 tickets) or logistic regression (`logreg`, slower here, same idea). Either way it is wrapped in
CalibratedClassifierCV with isotonic regression, so a reported 0.9 means about nine in ten of those
predictions are right. That is what lets the cascade use one threshold per tier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .taxonomy import PRIORITIES


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40_000)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60_000
                ),
            ),
        ]
    )


@dataclass
class Prediction:
    intent: str
    confidence: float
    priority: str
    probs: dict[str, float]


class ClassicalTier:
    def __init__(
        self, C: float = 0.5, calibration: str = "isotonic", cv: int = 3, model: str = "linear", seed: int = 7
    ):
        self.C = C
        self.model = model
        self.calibration = calibration
        self.cv = cv
        self.seed = seed
        self.features: FeatureUnion | None = None
        self.intent_model: CalibratedClassifierCV | None = None
        self.priority_model: LogisticRegression | None = None
        self.n_train = 0

    @property
    def classes(self) -> list[str]:
        assert self.intent_model is not None, "train first"
        return [str(c) for c in self.intent_model.classes_]

    def fit(self, texts: list[str], intents: list[str], priorities: list[str]) -> ClassicalTier:
        self.features = _features()
        X = self.features.fit_transform(texts)
        if self.model == "logreg":
            base = LogisticRegression(C=self.C * 8, max_iter=2000)
        else:
            base = LinearSVC(C=self.C, random_state=self.seed)
        # cv cannot exceed the smallest class count, which matters for small feedback seeds.
        counts = np.unique(np.asarray(intents), return_counts=True)[1]
        cv = int(max(2, min(self.cv, counts.min())))
        self.intent_model = CalibratedClassifierCV(base, method=self.calibration, cv=cv)
        self.intent_model.fit(X, intents)
        self.default_priority = max(set(priorities), key=list(priorities).count)
        self.priority_model = None
        if len(set(priorities)) > 1:
            self.priority_model = LogisticRegression(C=2.0, max_iter=500).fit(X, priorities)
        self.n_train = len(texts)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.features is not None and self.intent_model is not None, "train first"
        return self.intent_model.predict_proba(self.features.transform(texts))

    def predict_many(self, texts: list[str]) -> tuple[list[str], np.ndarray, list[str]]:
        """Intent, calibrated confidence, and priority for each text."""
        assert self.features is not None and self.intent_model is not None, "train first"
        X = self.features.transform(texts)
        P = self.intent_model.predict_proba(X)
        idx = P.argmax(axis=1)
        classes = self.intent_model.classes_
        intents = [str(classes[i]) for i in idx]
        conf = P[np.arange(len(texts)), idx]
        if self.priority_model is None:
            prios = [self.default_priority] * len(texts)
        else:
            prios = [str(p) for p in self.priority_model.predict(X)]
        return intents, conf, prios

    def predict(self, text: str) -> Prediction:
        P = self.predict_proba([text])[0]
        intents, conf, prios = self.predict_many([text])
        top = np.argsort(-P)[:3]
        return Prediction(
            intent=intents[0],
            confidence=float(conf[0]),
            priority=prios[0] if prios[0] in PRIORITIES else "normal",
            probs={self.classes[i]: round(float(P[i]), 4) for i in top},
        )


def reliability_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability diagram data and expected calibration error (ECE)."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1.0 else (confidence <= hi))
        k = int(mask.sum())
        if k == 0:
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "count": 0, "confidence": None, "accuracy": None})
            continue
        c, a = float(confidence[mask].mean()), float(correct[mask].mean())
        ece += k / n * abs(c - a)
        bins.append(
            {"lo": round(lo, 2), "hi": round(hi, 2), "count": k, "confidence": round(c, 4), "accuracy": round(a, 4)}
        )
    return {"bins": bins, "ece": round(ece, 4)}
