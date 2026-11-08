"""Prometheus text exposition without extra dependencies."""

from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.routed = defaultdict(int)  # tier
        self.cost = defaultdict(float)  # tier
        self.safety = 0
        self.corrections = 0
        self.retrains = 0
        self.http = defaultdict(int)  # (path, status)
        self.buckets = [0] * len(BUCKETS)
        self.dur_sum = 0.0
        self.dur_count = 0

    def observe_route(self, tier: str, cost: float, safety: bool, seconds: float) -> None:
        with self.lock:
            self.routed[tier] += 1
            self.cost[tier] += cost
            self.safety += int(safety)
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    self.buckets[i] += 1
            self.dur_sum += seconds
            self.dur_count += 1

    def observe_http(self, path: str, status: int) -> None:
        with self.lock:
            self.http[(path, str(status))] += 1

    def render(self) -> str:
        out: list[str] = []
        with self.lock:

            def family(name, kind, help_text, samples):
                out.append(f"# HELP {name} {help_text}")
                out.append(f"# TYPE {name} {kind}")
                for labels, value in samples:
                    lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    out.append(f"{name}{{{lab}}} {value}" if lab else f"{name} {value}")

            family(
                "ticketrouter_tickets_routed_total",
                "counter",
                "Tickets routed, by the tier that decided.",
                [({"tier": t}, v) for t, v in sorted(self.routed.items())],
            )
            family(
                "ticketrouter_cost_usd_total",
                "counter",
                "Model spend in USD by deciding tier (cumulative path).",
                [({"tier": t}, f"{v:.8f}") for t, v in sorted(self.cost.items())],
            )
            family(
                "ticketrouter_safety_routes_total",
                "counter",
                "Tickets sent to a human safety queue.",
                [({}, self.safety)],
            )
            family(
                "ticketrouter_feedback_corrections_total",
                "counter",
                "Agent corrections received.",
                [({}, self.corrections)],
            )
            family("ticketrouter_retrains_total", "counter", "Classical tier retrains.", [({}, self.retrains)])
            family(
                "ticketrouter_http_requests_total",
                "counter",
                "HTTP requests by path and status.",
                [({"path": p, "status": s}, v) for (p, s), v in sorted(self.http.items())],
            )
            out.append(
                "# HELP ticketrouter_route_duration_seconds Time to route one ticket (local tiers, "
                "plus simulated LLM time is not slept)."
            )
            out.append("# TYPE ticketrouter_route_duration_seconds histogram")
            for b, c in zip(BUCKETS, self.buckets, strict=True):
                out.append(f'ticketrouter_route_duration_seconds_bucket{{le="{b}"}} {c}')
            out.append(f'ticketrouter_route_duration_seconds_bucket{{le="+Inf"}} {self.dur_count}')
            out.append(f"ticketrouter_route_duration_seconds_sum {self.dur_sum:.6f}")
            out.append(f"ticketrouter_route_duration_seconds_count {self.dur_count}")
        return "\n".join(out) + "\n"
