"""FastAPI service: classify and route tickets, queue views, agent feedback, retraining, the cascade
evaluation, and the drift monitor."""

from __future__ import annotations

import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__, data, drift
from .cascade import Policy, Router
from .evaluate import DEFAULT_BUDGET, DEFAULT_TOLERANCE, Bench, build_bench, report
from .feedback import FeedbackStore, retrain
from .llm import HAIKU_SIM, OPUS_SIM, ClaudeTier, SimulatedLLM
from .logs import get_logger, log_event
from .metrics import Metrics
from .taxonomy import BY_NAME, INTENT_NAMES, PRIORITIES, TEAMS

STATIC = Path(__file__).parent / "static"
NEW_INTENT = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


@dataclass
class Settings:
    tickets: int = int(os.environ.get("TICKETROUTER_TICKETS", "20000"))
    seed: int = int(os.environ.get("TICKETROUTER_SEED", "7"))
    db: str = os.environ.get("TICKETROUTER_DB", "data/feedback.db")
    llm: str = os.environ.get("TICKETROUTER_LLM", "sim")  # sim | claude
    budget: float = float(os.environ.get("TICKETROUTER_BUDGET", str(DEFAULT_BUDGET)))
    tolerance: float = DEFAULT_TOLERANCE
    queue_seed: int = 60


class ClassifyIn(BaseModel):
    text: str = Field(min_length=3, max_length=8000)
    id: str | None = Field(default=None, max_length=40)


class FeedbackIn(BaseModel):
    ticket_id: str = Field(max_length=40)
    intent: str
    priority: str
    agent: str = Field(default="agent", max_length=40)


class State:
    def __init__(self, settings: Settings, haiku=None, opus=None):
        self.settings = settings
        self.lock = threading.Lock()
        if settings.llm == "claude" and haiku is None:
            haiku, opus = ClaudeTier("claude-haiku-5-5"), ClaudeTier("claude-opus-5-5")
        # Evaluation always uses the simulated tiers so it is free and repeatable.
        self.bench: Bench = build_bench(n=settings.tickets, seed=settings.seed)
        self.report = report(self.bench, tolerance=settings.tolerance, budget=settings.budget)
        p = self.report["operating_point"]["policy"]
        self.policy = Policy(p["t_rules"], p["t_classical"], p["t_haiku"])
        self.haiku = haiku or SimulatedLLM(HAIKU_SIM)
        self.opus = opus or SimulatedLLM(OPUS_SIM)
        self.router = Router(self.bench.classical, self.haiku, self.opus, self.policy)
        self.store = FeedbackStore(settings.db)
        self.metrics = Metrics()
        self.queue: deque[dict] = deque(maxlen=1000)
        self.by_id: dict[str, dict] = {}
        self.model_version = 1
        self.stream = data.drift_stream(self.bench.pool, per_week=max(50, len(self.bench.pool) // 20))
        self.drift = drift.detect(self.bench.classical, self.stream, [t.text for t in self.bench.train])
        self._seed_queue()

    def _seed_queue(self) -> None:
        """Route a sample of the latest week so the queue is not empty on first load. That week
        includes tickets about the newly launched product."""
        last = max(t.week for t in self.stream)
        recent = [t for t in self.stream if t.week == last]
        sample = random.Random(self.settings.seed).sample(recent, min(self.settings.queue_seed, len(recent)))
        for t in sorted(sample, key=lambda t: t.id):
            self.add(t.text, t.id, gold=t)

    def add(self, text: str, ticket_id: str | None = None, gold=None) -> dict:
        start = time.perf_counter()
        d = self.router.route(text, gold=gold)
        elapsed = time.perf_counter() - start
        with self.lock:
            tid = ticket_id or f"N{len(self.by_id) + 1:05d}"
            item = {
                "id": tid,
                "text": text,
                "received_at": time.time(),
                "decision": d.to_dict(),
                "model_version": self.model_version,
                "corrected": None,
            }
            self.queue.appendleft(item)
            self.by_id[tid] = item
        self.metrics.observe_route(d.tier, d.cost, d.safety, elapsed)
        return item


def create_app(settings: Settings | None = None, haiku=None, opus=None) -> FastAPI:
    settings = settings or Settings()
    log = get_logger()
    t0 = time.perf_counter()
    state = State(settings, haiku=haiku, opus=opus)
    log_event(
        log,
        "ready",
        seconds=round(time.perf_counter() - t0, 1),
        tickets=settings.tickets,
        policy=state.policy.label(),
        llm=settings.llm,
    )
    app = FastAPI(title="Ticket Router", version=__version__)
    app.state.s = state

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        state.metrics.observe_http(path, response.status_code)
        if path not in ("/healthz", "/metrics"):
            log_event(
                log,
                "request",
                method=request.method,
                path=path,
                status=response.status_code,
                ms=round((time.perf_counter() - start) * 1000, 2),
            )
        return response

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "version": __version__,
            "model_version": state.model_version,
            "llm": settings.llm,
            "queue": len(state.queue),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return state.metrics.render()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/taxonomy")
    def taxonomy():
        return {
            "intents": [{"name": n, "team": BY_NAME[n].team, "priority": BY_NAME[n].priority} for n in INTENT_NAMES],
            "priorities": list(PRIORITIES),
            "teams": list(TEAMS),
        }

    @app.post("/api/classify")
    def classify(body: ClassifyIn):
        if body.id and body.id in state.by_id:
            raise HTTPException(409, f"ticket {body.id} already exists")
        item = state.add(body.text.strip(), body.id)
        d = item["decision"]
        log_event(
            log,
            "routed",
            ticket=item["id"],
            tier=d["tier"],
            intent=d["intent"],
            team=d["team"],
            confidence=round(d["confidence"], 3),
            cost=d["cost"],
            safety=d["safety"],
        )
        return item

    @app.get("/api/queue")
    def queue(team: str | None = None, tier: str | None = None, limit: int = 50):
        items = [
            i
            for i in state.queue
            if (team is None or i["decision"]["team"] == team) and (tier is None or i["decision"]["tier"] == tier)
        ]
        counts: dict[str, int] = {}
        tiers: dict[str, int] = {}
        for i in state.queue:
            counts[i["decision"]["team"]] = counts.get(i["decision"]["team"], 0) + 1
            tiers[i["decision"]["tier"]] = tiers.get(i["decision"]["tier"], 0) + 1
        return {
            "items": items[: max(1, min(limit, 500))],
            "total": len(items),
            "by_team": counts,
            "by_tier": tiers,
            "policy": state.policy.to_dict(),
        }

    @app.get("/api/tickets/{ticket_id}")
    def ticket(ticket_id: str):
        if ticket_id not in state.by_id:
            raise HTTPException(404, "no such ticket")
        return state.by_id[ticket_id]

    @app.post("/api/feedback")
    def feedback(body: FeedbackIn):
        item = state.by_id.get(body.ticket_id)
        if item is None:
            raise HTTPException(404, "no such ticket")
        if body.priority not in PRIORITIES:
            raise HTTPException(422, f"priority must be one of {list(PRIORITIES)}")
        if body.intent not in INTENT_NAMES and not NEW_INTENT.match(body.intent):
            raise HTTPException(422, "intent must be a known intent or a new snake_case name")
        cid = state.store.add(
            body.ticket_id, item["text"], item["decision"]["intent"], body.intent, body.priority, body.agent
        )
        item["corrected"] = {"intent": body.intent, "priority": body.priority, "agent": body.agent}
        state.metrics.corrections += 1
        log_event(
            log,
            "feedback",
            ticket=body.ticket_id,
            predicted=item["decision"]["intent"],
            corrected=body.intent,
            new_intent=body.intent not in INTENT_NAMES,
        )
        return {"id": cid, "ticket_id": body.ticket_id, "stored": True, "corrections": state.store.count()}

    @app.get("/api/feedback")
    def feedback_list():
        return {"items": state.store.all()}

    @app.post("/api/retrain")
    def retrain_now():
        b = state.bench
        corrections = state.store.latest_per_ticket()
        test_texts = [t.text for t in b.test]
        gold = np.array([t.intent for t in b.test])
        before = float((np.array(state.router.classical.predict_many(test_texts)[0]) == gold).mean())
        start = time.perf_counter()
        model, pending = retrain(
            [t.text for t in b.train], [t.intent for t in b.train], [t.priority for t in b.train], corrections
        )
        after = float((np.array(model.predict_many(test_texts)[0]) == gold).mean())
        with state.lock:
            state.router.classical = model
            state.model_version += 1
        state.store.mark_used()
        state.metrics.retrains += 1
        out = {
            "model_version": state.model_version,
            "corrections_used": len(corrections),
            "pending_new_intents": pending,
            "classes": len(model.classes),
            "holdout_accuracy_before": round(before, 4),
            "holdout_accuracy_after": round(after, 4),
            "seconds": round(time.perf_counter() - start, 1),
        }
        log_event(log, "retrained", **out)
        return out

    @app.get("/api/eval")
    def evaluation(points: bool = True):
        rep = dict(state.report)
        if not points:
            rep.pop("sweep_points")
        return rep

    @app.get("/api/drift")
    def drift_view():
        return state.drift

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    return app
