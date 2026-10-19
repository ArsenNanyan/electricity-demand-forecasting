"""API tests — use the seasonal model so no trained artifacts are required."""
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)
HIST = list(np.linspace(1, 5, 168))


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_seasonal_forecast_ok():
    r = client.post("/forecast", json={
        "history": HIST, "origin": "2014-06-01T00:00:00",
        "model": "Seasonal naive (24h)"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 24
    assert len(body["timestamps"]) == 24
    assert body["units"] == "kW (hourly mean)"


def test_rejects_wrong_history_length():
    r = client.post("/forecast", json={
        "history": HIST[:100], "origin": "2014-06-01T00:00:00",
        "model": "Seasonal naive (24h)"})
    assert r.status_code == 422
