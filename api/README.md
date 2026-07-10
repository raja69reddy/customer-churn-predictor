# Churn Prediction API

FastAPI service that scores customer churn risk, looks up individual customers, and
exposes model performance / retention analytics for the customer-churn-predictor project.

## Starting the server

```bash
# From the repo root, with the venv activated and .env configured
uvicorn api.main:app --reload
```

The server starts on `http://localhost:8000`. On startup it loads the current
Production model from the MLflow Model Registry (see `api/predictor.py`).

- Interactive docs (Swagger UI): `http://localhost:8000/docs`
- Alternate docs (ReDoc): `http://localhost:8000/redoc`

## Architecture

| File | Responsibility |
|---|---|
| `api/main.py` | FastAPI app, routing, middleware, error handling |
| `api/models.py` | Pydantic request/response schemas |
| `api/predictor.py` | `APIPredictor` — loads the MLflow model and scores customers |
| `api/database.py` | PostgreSQL data-access helpers (customer lookup, prediction persistence, analytics) |

## Endpoints

### General

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API metadata |
| GET | `/health` | Health check |

### Predictions

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Score a single customer supplied in the request body |
| POST | `/predict/batch` | Score a batch of customers |
| GET | `/model/info` | Active model name, MLflow version, and registry metrics |
| GET | `/model/performance` | Performance metrics for all trained models |

### Customer lookup

| Method | Endpoint | Description |
|---|---|---|
| GET | `/customer/{customer_id}` | Raw profile details for a customer |
| GET | `/customer/{customer_id}/predict` | Score an existing customer by ID (also persists the prediction) |
| GET | `/customer/{customer_id}/explanation` | AI (or template-fallback) explanation of a customer's risk |
| GET | `/customers/high-risk?limit=100` | High-risk customers sorted by churn probability |
| GET | `/customers/risk-summary` | Per-segment counts, averages, and revenue at risk |

### Analytics

| Method | Endpoint | Description |
|---|---|---|
| GET | `/analytics/churn-rate` | Overall historical churn rate |
| GET | `/analytics/risk-distribution` | Customer counts per predicted risk segment |
| GET | `/analytics/revenue-at-risk` | Monthly revenue at risk by segment |
| GET | `/analytics/top-features?limit=15` | Top churn-driving features by importance |
| GET | `/analytics/retention-targets` | Retention priority tier counts and revenue exposure |

## Request / response examples

### `POST /predict`

Request:
```json
{
  "customer_id": "CUST-DEMO-001",
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
  "total_charges": 1026.0
}
```

Response:
```json
{
  "customer_id": "CUST-DEMO-001",
  "churn_probability": 0.9594,
  "risk_segment": "High",
  "top_risk_factors": [
    {"feature": "tenure_group_New Customer", "shap_value": 1.78, "direction": "increases risk"}
  ],
  "recommended_actions": ["Offer a loyalty discount or incentive to switch to a longer-term contract"],
  "model_version": "churn-predictor_v1",
  "predicted_at": "2026-07-10T15:39:12.092137"
}
```

### `GET /customer/{customer_id}/predict`

```bash
curl http://localhost:8000/customer/CUST-05036/predict
```

Scores the customer using their stored `raw_customers` profile and persists the
result to `churn_predictions` (archiving the prior prediction to
`churn_predictions_history` first).

### Validation errors

`monthly_charges` must be greater than 0, `tenure` must be between 0 and 100, and
`total_charges` cannot be more than $10 below `monthly_charges` (a tolerance that
accounts for legitimate data noise on new customers). Violating any of these
returns a `422` with a Pydantic-style error body:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "BAD-001", "monthly_charges": -5, ...}'
```

```json
{
  "detail": [
    {"type": "greater_than", "loc": ["body", "monthly_charges"], "msg": "Input should be greater than 0"}
  ]
}
```

Unknown customer IDs return a `404`:
```bash
curl http://localhost:8000/customer/NOPE
# {"detail": "Customer 'NOPE' not found."}
```

Unexpected server-side failures (e.g. a model scoring error) return a friendly `500`
rather than a raw traceback.

## curl cheatsheet

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_customer.json

curl http://localhost:8000/customers/high-risk?limit=10

curl http://localhost:8000/analytics/churn-rate
```

## Manual smoke test

```bash
uvicorn api.main:app --reload &
python -m api.test_endpoints_manual
```

Exercises every endpoint above against a running server and asserts none return a 5xx.

## Authentication

Not yet implemented — all endpoints are currently open. Adding an API key or OAuth2
bearer token scheme is tracked as future work (see the main `README.md`).
