from ticketrouter.cascade import choose_operating_point, pareto_frontier, sweep
from ticketrouter.evaluate import gate


def pts(*pairs):
    return [{"cost_per_1k": c, "intent_accuracy": a, "policy": {"label": f"p{i}"}} for i, (c, a) in enumerate(pairs)]


def test_frontier_known_example():
    p = pts((0.0, 0.80), (0.1, 0.85), (0.2, 0.84), (0.3, 0.90), (0.3, 0.88), (1.0, 0.90), (2.0, 0.93))
    front = [(x["cost_per_1k"], x["intent_accuracy"]) for x in pareto_frontier(p)]
    assert front == [(0.0, 0.80), (0.1, 0.85), (0.3, 0.90), (2.0, 0.93)]


def test_frontier_points_are_not_dominated(bench):
    points = sweep(bench.table)
    front = pareto_frontier(points)
    for f in front:
        assert not any(
            q["cost_per_1k"] <= f["cost_per_1k"] and q["intent_accuracy"] > f["intent_accuracy"] for q in points
        )
    costs = [f["cost_per_1k"] for f in front]
    accs = [f["intent_accuracy"] for f in front]
    assert costs == sorted(costs) and accs == sorted(accs)


def test_operating_point_respects_budget():
    front = pareto_frontier(pts((0.0, 0.80), (0.1, 0.85), (0.3, 0.90), (2.0, 0.93)))
    op = choose_operating_point(front, budget_per_1k=0.5)
    assert op["cost_per_1k"] == 0.3 and op["within_budget"]


def test_operating_point_slack_prefers_cheaper():
    front = pareto_frontier(pts((0.1, 0.900), (0.4, 0.901)))
    assert choose_operating_point(front, 1.0, slack=0.0025)["cost_per_1k"] == 0.1
    assert choose_operating_point(front, 1.0, slack=0.0)["cost_per_1k"] == 0.4


def test_operating_point_over_budget_fallback():
    front = pareto_frontier(pts((0.6, 0.8), (1.0, 0.9)))
    op = choose_operating_point(front, budget_per_1k=0.5)
    assert op["cost_per_1k"] == 0.6 and not op["within_budget"]


def test_gate_pass_and_fail():
    rep = {
        "single_tier": {"opus": {"intent_accuracy": 0.93}},
        "operating_point": {"intent_accuracy": 0.925, "cost_per_1k": 0.3, "tolerance": 0.01, "budget_per_1k": 0.5},
    }
    assert gate(rep)[0]
    rep["operating_point"]["intent_accuracy"] = 0.90
    ok, msgs = gate(rep)
    assert not ok and msgs[0].startswith("FAIL")
    rep["operating_point"].update(intent_accuracy=0.93, cost_per_1k=0.7)
    assert not gate(rep)[0]
