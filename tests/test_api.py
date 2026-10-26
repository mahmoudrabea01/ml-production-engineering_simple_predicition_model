import pytest
from fastapi.testclient import TestClient

import production_app.api as api_module


@pytest.fixture
def client():
    """Create a test client without starting a real server."""
    with TestClient(api_module.app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Provide the three features expected by the API."""
    return {
        "mean_concave_points": 0.1471,
        "worst_radius": 25.38,
        "worst_concave_points": 0.2654,
    }


def test_health_success(client):
    """Verify that the health endpoint returns HTTP 200."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_success(client, valid_payload):
    """Verify that valid inputs return HTTP 200."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] in [0, 1]
    assert result["predicted_label"] in ["malignant", "benign"]


def test_predict_invalid_input(client, valid_payload):
    """Verify that missing features return HTTP 422."""
    invalid_payload = valid_payload.copy()
    invalid_payload.pop("worst_radius")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_model_unavailable(client, valid_payload, monkeypatch):
    """Verify HTTP 503 without touching the real model file."""
    monkeypatch.setattr(api_module, "model", None)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 503


def test_health_model_unavailable(client, monkeypatch):
    """Verify that health reports an unavailable model."""
    monkeypatch.setattr(api_module, "model", None)

    response = client.get("/health")

    assert response.status_code == 503
