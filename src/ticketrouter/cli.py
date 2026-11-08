"""Command line: serve the API, run the evaluation and CI gate, the feedback simulation, and drift."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evaluate import DEFAULT_BUDGET, DEFAULT_TOLERANCE, build_bench, gate, markdown_table, report


def _write(path: str | None, obj) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2, default=str))
        print(f"wrote {path}")


def cmd_serve(a) -> None:
    import uvicorn

    from .app import Settings, create_app

    app = create_app(Settings(tickets=a.tickets))
    uvicorn.run(
        app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", a.port)), log_level="warning"
    )


def cmd_eval(a) -> int:
    bench = build_bench(n=a.tickets)
    rep = report(bench, tolerance=a.tolerance, budget=a.budget)
    table = markdown_table(rep)
    op = rep["operating_point"]
    print(table)
    print(f"\noperating point: {op['policy']['label']}")
    print(f"  intent accuracy {op['intent_accuracy']:.4f}, cost ${op['cost_per_1k']:.4f} per 1k tickets")
    print(f"  classical tier ECE {rep['calibration']['ece']:.4f}, LLM tiers simulated: {rep['llm_tiers_simulated']}")
    out = {k: v for k, v in rep.items() if k != "sweep_points"} if not a.points else rep
    _write(a.out, out)
    ok, msgs = gate(rep)
    for m in msgs:
        print(m)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write("## Cascade evaluation (LLM tiers simulated)\n\n" + table + "\n\n")
            f.write(f"Operating point: `{op['policy']['label']}`\n\n" + "\n".join(f"- {m}" for m in msgs) + "\n")
    if a.gate and not ok:
        return 1
    return 0


def cmd_feedback(a) -> int:
    from .cascade import Policy
    from .feedback import simulate_weeks

    bench = build_bench(n=a.tickets)
    rep = report(bench)
    p = rep["operating_point"]["policy"]
    rows = simulate_weeks(
        bench.train,
        bench.test[: a.holdout],
        bench.pool,
        bench.haiku,
        bench.opus,
        Policy(p["t_rules"], p["t_classical"], p["t_haiku"]),
        weeks=a.weeks,
    )
    print("| Week | Labeled examples | Classical acc | Classical ECE | Cascade acc | Cost per 1k | LLM share |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['week']} | {r['labeled_examples']:,} | {r['classical_accuracy']:.1%} | {r['classical_ece']:.3f} | "
            f"{r['cascade_accuracy']:.1%} | ${r['cascade_cost_per_1k']:.3f} | {r['llm_share']:.0%} |"
        )
    _write(a.out, rows)
    return 0


def cmd_drift(a) -> int:
    from . import data, drift

    bench = build_bench(n=a.tickets)
    stream = data.drift_stream(bench.pool)
    res = drift.detect(bench.classical, stream, [t.text for t in bench.train])
    print(f"baseline low-confidence share {res['baseline_share']:.1%}, first alarm week {res['first_alarm_week']}")
    for r in res["weeks"]:
        print(
            f"  week {r['week']:2d}  {r['low_conf_share']:.1%}  limit {r['limit']:.1%}  {'ALARM' if r['alarm'] else ''}"
        )
    for c in res["surfaced"]:
        print(
            f"  cluster {c['cluster']}: {c['size']} tickets, novelty {c['novelty']}, terms {c['top_terms'][:6]}, "
            f"hidden majority {c['hidden_majority_intent']} ({c['hidden_purity']:.0%})"
        )
    _write(a.out, res)
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ticketrouter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the API and web UI")
    s.add_argument("--port", type=int, default=8836)
    s.add_argument("--tickets", type=int, default=int(os.environ.get("TICKETROUTER_TICKETS", "20000")))
    e = sub.add_parser("eval", help="cascade evaluation, optional CI gate")
    e.add_argument("--tickets", type=int, default=20000)
    e.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    e.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="USD per 1,000 tickets")
    e.add_argument("--out")
    e.add_argument("--points", action="store_true", help="include every sweep point in --out")
    e.add_argument("--gate", action="store_true", help="exit 1 if the operating point fails the gate")
    e.add_argument("--summary", help="append a markdown summary here (GitHub job summary)")
    f = sub.add_parser("feedback-sim", help="simulate weeks of agent corrections and retraining")
    f.add_argument("--tickets", type=int, default=20000)
    f.add_argument("--weeks", type=int, default=8)
    f.add_argument("--holdout", type=int, default=3000)
    f.add_argument("--out")
    d = sub.add_parser("drift", help="run the drift monitor on a stream with a product launch")
    d.add_argument("--tickets", type=int, default=20000)
    d.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        cmd_serve(a)
        return
    sys.exit({"eval": cmd_eval, "feedback-sim": cmd_feedback, "drift": cmd_drift}[a.cmd](a))
