from ticketrouter.evaluate import markdown_table, report


def test_report_structure(bench):
    rep = report(bench)
    assert rep["dataset"]["intents"] == 25 and rep["llm_tiers_simulated"]
    op = rep["operating_point"]
    assert op["cost_per_1k"] <= op["budget_per_1k"]
    assert sum(v["share"] for v in rep["per_tier"].values()) > 0.999
    assert len(rep["per_intent"]) == 25 and rep["top_confusions"]
    assert rep["top_confusions"][0]["gold"] != rep["top_confusions"][0]["predicted"]


def test_cascade_beats_cheap_baselines_and_costs_less_than_opus(bench):
    rep = report(bench)
    c, s = rep["cascade"], rep["single_tier"]
    assert c["intent_accuracy"] >= s["classical"]["intent_accuracy"]
    assert c["intent_accuracy"] > s["haiku"]["intent_accuracy"]
    assert c["cost_per_1k"] < s["opus"]["cost_per_1k"] / 3
    assert rep["safety"]["routed_to_human_queue"] > 0.85


def test_markdown_table_marks_simulated_rows(bench):
    table = markdown_table(report(bench))
    assert "Opus for everything (simulated)" in table and table.count("\n") == 6
