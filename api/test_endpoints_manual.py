"""Manual smoke test that exercises every live API endpoint with httpx.

Run this against a locally running server:
    uvicorn api.main:app --reload
    python -m api.test_endpoints_manual
"""

import json

import httpx

BASE_URL = "http://localhost:8000"

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


def show(name: str, response: httpx.Response) -> None:
    print(f"\n=== {name} -> {response.status_code} ===")
    try:
        print(json.dumps(response.json(), indent=2)[:500])
    except ValueError:
        print(response.text[:300])
    assert response.status_code < 500, f"{name} returned a server error"


def main() -> None:
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    show("GET /health", client.get("/health"))
    show("GET /", client.get("/"))
    show("GET /model/info", client.get("/model/info"))
    show("GET /model/performance", client.get("/model/performance"))

    show("POST /predict", client.post("/predict", json=SAMPLE_CUSTOMER))

    batch = {
        "customers": [
            {**SAMPLE_CUSTOMER, "customer_id": "TEST-API-01"},
            {
                **SAMPLE_CUSTOMER,
                "customer_id": "TEST-API-02",
                "contract": "Two year",
                "tenure": 60,
            },
            {**SAMPLE_CUSTOMER, "customer_id": "TEST-API-03", "senior_citizen": 1},
        ]
    }
    show("POST /predict/batch", client.post("/predict/batch", json=batch))

    show(
        "GET /customers/high-risk",
        client.get("/customers/high-risk", params={"limit": 3}),
    )
    show("GET /customers/risk-summary", client.get("/customers/risk-summary"))
    show("GET /customer/CUST-05036", client.get("/customer/CUST-05036"))
    show("GET /customer/CUST-05036/predict", client.get("/customer/CUST-05036/predict"))
    show(
        "GET /customer/CUST-05036/explanation",
        client.get("/customer/CUST-05036/explanation"),
    )
    show("GET /customer/NOPE (expect 404)", client.get("/customer/NOPE"))

    show("GET /analytics/churn-rate", client.get("/analytics/churn-rate"))
    show("GET /analytics/risk-distribution", client.get("/analytics/risk-distribution"))
    show("GET /analytics/revenue-at-risk", client.get("/analytics/revenue-at-risk"))
    show(
        "GET /analytics/top-features",
        client.get("/analytics/top-features", params={"limit": 5}),
    )
    show("GET /analytics/retention-targets", client.get("/analytics/retention-targets"))

    bad_charges = {**SAMPLE_CUSTOMER, "customer_id": "BAD-001", "monthly_charges": -5}
    show(
        "POST /predict invalid monthly_charges (expect 422)",
        client.post("/predict", json=bad_charges),
    )

    bad_total = {
        **SAMPLE_CUSTOMER,
        "customer_id": "BAD-002",
        "total_charges": 0,
        "monthly_charges": 500,
    }
    show(
        "POST /predict invalid total_charges (expect 422)",
        client.post("/predict", json=bad_total),
    )

    print("\nAll endpoints exercised successfully.")


if __name__ == "__main__":
    main()
