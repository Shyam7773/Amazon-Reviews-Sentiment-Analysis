# tests/test_integration_api.py
from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "version" in data

def test_predict_endpoint():
    r = client.post("/predict", json={"text": "This is fantastic!"})
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ["positive", "negative"]
