import json

import pytest
from fastapi.testclient import TestClient

from ticketrouter.app import Settings, create_app
from ticketrouter.llm import ClaudeTier


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "fb.db"
    return TestClient(create_app(Settings(tickets=3000, db=str(db), queue_seed=40)))


def test_healthz_and_index(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["llm"] == "sim" and h["queue"] == 40
    r = client.get("/")
    assert r.status_code == 200 and "Ticket Router" in r.text


def test_taxonomy(client):
    t = client.get("/api/taxonomy").json()
    assert len(t["intents"]) == 25 and t["priorities"] == ["urgent", "high", "normal", "low"]


def test_classify_routine_ticket(client):
    item = client.post("/api/classify", json={"text": "Please refund the $400 charge for the seats we removed"}).json()
    d = item["decision"]
    assert d["tier"] in ("rules", "classical", "haiku", "opus") and d["intent"] and d["cost"] >= 0
    assert item["id"].startswith("N") and d["path"][0]["tier"] == "rules"


def test_classify_safety_ticket_goes_to_human_queue(client):
    d = client.post(
        "/api/classify", json={"text": "Is there an outage? The API returns 500 errors for all requests."}
    ).json()["decision"]
    assert d["tier"] == "safety" and d["team"] == "incident_response" and d["priority"] == "urgent"


def test_classify_validation_and_duplicates(client):
    assert client.post("/api/classify", json={"text": "x"}).status_code == 422
    assert client.post("/api/classify", json={"text": "pw reset pls", "id": "DUP1"}).status_code == 200
    assert client.post("/api/classify", json={"text": "pw reset pls", "id": "DUP1"}).status_code == 409


def test_queue_filters(client):
    q = client.get("/api/queue").json()
    assert q["total"] >= 40 and sum(q["by_team"].values()) == q["total"]
    team = next(iter(q["by_team"]))
    f = client.get(f"/api/queue?team={team}").json()
    assert f["total"] == q["by_team"][team] and all(i["decision"]["team"] == team for i in f["items"])
    s = client.get("/api/queue?tier=safety").json()
    assert all(i["decision"]["tier"] == "safety" for i in s["items"])


def test_feedback_and_retrain(client):
    item = client.post("/api/classify", json={"text": "Pulse forecasts show flat lines for every metric"}).json()
    assert (
        client.post("/api/feedback", json={"ticket_id": "nope", "intent": "sso_setup", "priority": "low"}).status_code
        == 404
    )
    assert (
        client.post(
            "/api/feedback", json={"ticket_id": item["id"], "intent": "Bad Name!", "priority": "low"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/feedback", json={"ticket_id": item["id"], "intent": "sso_setup", "priority": "asap"}
        ).status_code
        == 422
    )
    ok = client.post(
        "/api/feedback",
        json={"ticket_id": item["id"], "intent": "pulse_forecasting", "priority": "normal", "agent": "test"},
    ).json()
    assert ok["stored"] and ok["corrections"] >= 1
    assert client.get(f"/api/tickets/{item['id']}").json()["corrected"]["intent"] == "pulse_forecasting"
    r = client.post("/api/retrain").json()
    assert r["model_version"] == 2 and r["pending_new_intents"] == {"pulse_forecasting": 1}
    assert 0 < r["holdout_accuracy_after"] <= 1
    assert client.get("/healthz").json()["model_version"] == 2


def test_eval_endpoint(client):
    e = client.get("/api/eval").json()
    assert e["llm_tiers_simulated"] is True
    assert set(e["single_tier"]) == {"rules", "classical", "haiku", "opus"}
    assert e["frontier"] and e["sweep_points"] and e["operating_point"]["policy"]["label"]
    assert "sweep_points" not in client.get("/api/eval?points=false").json()


def test_drift_endpoint(client):
    d = client.get("/api/drift").json()
    assert len(d["weeks"]) == 12 and "surfaced" in d


def test_metrics_endpoint(client):
    text = client.get("/metrics").text
    assert "ticketrouter_tickets_routed_total" in text and "ticketrouter_route_duration_seconds_count" in text
    assert 'ticketrouter_http_requests_total{path="/api/classify"' in text


def test_service_with_fake_claude_client(fake_client, tmp_path):
    reply = json.dumps(
        {
            "intent": "refund_request",
            "priority": "normal",
            "team": "billing",
            "confidence": 0.3,
            "reason": "asks for money back",
        }
    )
    haiku = ClaudeTier("claude-haiku-5-5", client=fake_client(reply))
    opus = ClaudeTier("claude-opus-5-5", client=fake_client(reply.replace("0.3", "0.95")))
    c = TestClient(create_app(Settings(tickets=1500, db=str(tmp_path / "f.db"), queue_seed=0), haiku=haiku, opus=opus))
    d = c.post("/api/classify", json={"text": "zzz qqq unusual words only"}).json()["decision"]
    if d["tier"] in ("haiku", "opus"):
        assert d["simulated_llm"] is False and d["cost"] > 0
    assert opus.client.messages.calls or haiku.client.messages.calls or d["tier"] in ("rules", "classical")
