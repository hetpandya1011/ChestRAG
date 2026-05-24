"""Smoke test: the API boots and /health responds."""
from fastapi.testclient import TestClient

from services.api.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
