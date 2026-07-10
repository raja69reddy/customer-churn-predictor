"""Tests for the FastAPI prediction service, driven with httpx against the ASGI app."""
import pytest
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient

from api.main import app

SAMPLE_CUSTOMER = {
    "customer_id": "TEST-API-001",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 85.5,
    "total_charges": 1026.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_probability(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_segment"] in {"High", "Medium", "Low"}


def test_predict_invalid_monthly_charges_returns_422(client):
    bad_customer = {**SAMPLE_CUSTOMER, "monthly_charges": -5}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_invalid_total_charges_returns_422(client):
    bad_customer = {**SAMPLE_CUSTOMER, "total_charges": 0, "monthly_charges": 500}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_high_risk_customers_returns_list(client):
    response = client.get("/customers/high-risk", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) <= 5
    assert all(row["risk_segment"] == "High" for row in body)


def test_churn_rate_returns_float(client):
    response = client.get("/analytics/churn-rate")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["churn_rate"], float)
    assert 0.0 <= body["churn_rate"] <= 1.0


def test_model_info_returns_version(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert body["model_version"]


def test_unknown_customer_returns_404(client):
    response = client.get("/customer/NOPE")
    assert response.status_code == 404
