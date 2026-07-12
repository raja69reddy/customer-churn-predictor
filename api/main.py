"""FastAPI prediction service for the customer churn predictor."""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.database import (
    get_churn_rate as db_get_churn_rate,
    get_customer_from_db,
    get_high_risk_customers as db_get_high_risk_customers,
    get_risk_summary as db_get_risk_summary,
    save_prediction_to_db,
)
from api.models import (
    BatchPredictionInput,
    BatchPredictionOutput,
    CustomerInput,
    PredictionOutput,
)
from api.predictor import APIPredictor
from src.utils.db import get_engine


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Converts a DataFrame to JSON-safe records, turning NaN/NaT into None."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


CUSTOMER_INPUT_FIELDS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("churn_api")

predictor = APIPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_model()
    log.info("Loaded model: %s", predictor.model_version)
    yield


app = FastAPI(
    title="Customer Churn Predictor API",
    description=(
        "REST API for scoring customer churn risk, looking up individual customers, "
        "and retrieving model performance / retention analytics for the "
        "customer-churn-predictor project."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    log.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred while processing your request. Please try again later."
        },
    )


@app.get("/")
def root():
    return {
        "name": "Customer Churn Predictor API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------


def _run_model(fn, *args, **kwargs):
    """Runs a predictor call, converting unexpected model/runtime failures into a friendly 500."""
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Model error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The prediction model failed to score this customer. Please verify the input and try again.",
        ) from exc


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Scores a single customer for churn risk."""
    return _run_model(predictor.predict, customer)


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(payload: BatchPredictionInput):
    """Scores a batch of customers for churn risk."""
    predictions = _run_model(predictor.predict_batch, payload.customers)
    return BatchPredictionOutput(predictions=predictions)


@app.get("/model/info")
def model_info():
    """Returns the active model's name, MLflow version, and registry metadata."""
    return predictor.get_model_info()


@app.get("/model/performance")
def model_performance():
    """Returns current performance metrics for all trained models."""
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM vw_model_performance", engine)
    return _df_to_records(df)


# ---------------------------------------------------------------------------
# Customer lookup endpoints
# ---------------------------------------------------------------------------


def _load_customer_row(customer_id: str) -> dict:
    customer = get_customer_from_db(customer_id)
    if customer is None:
        raise HTTPException(
            status_code=404, detail=f"Customer '{customer_id}' not found."
        )
    return customer


@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):
    """Returns raw profile details for a single customer."""
    return _load_customer_row(customer_id)


@app.get("/customer/{customer_id}/predict", response_model=PredictionOutput)
def predict_customer(customer_id: str):
    """Scores an existing customer (by ID) for churn risk and persists the prediction."""
    customer = _load_customer_row(customer_id)
    customer_input = CustomerInput(
        customer_id=customer_id, **{f: customer[f] for f in CUSTOMER_INPUT_FIELDS}
    )
    result = _run_model(predictor.predict, customer_input)
    save_prediction_to_db(result.model_dump())
    return result


@app.get("/customer/{customer_id}/explanation")
def explain_customer(customer_id: str):
    """Returns an AI-generated (or template-fallback) explanation of an existing customer's churn risk."""
    customer = _load_customer_row(customer_id)
    customer_dict = {f: customer[f] for f in CUSTOMER_INPUT_FIELDS}

    result = _run_model(predictor.explain, customer_dict)
    return {"customer_id": customer_id, **result}


@app.get("/customers/high-risk")
def high_risk_customers(limit: int = 100):
    """Returns high-risk customers, sorted by churn probability descending."""
    return db_get_high_risk_customers(limit=limit)


@app.get("/customers/risk-summary")
def risk_summary():
    """Returns per-segment counts, average churn probability, and revenue at risk."""
    return db_get_risk_summary()


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------


@app.get("/analytics/churn-rate")
def churn_rate():
    """Returns the overall historical churn rate from raw_customers."""
    return db_get_churn_rate()


@app.get("/analytics/risk-distribution")
def risk_distribution():
    """Returns customer counts per predicted risk segment."""
    engine = get_engine()
    df = pd.read_sql(
        "SELECT risk_segment, COUNT(*) AS customer_count FROM churn_predictions GROUP BY risk_segment",
        engine,
    )
    return _df_to_records(df)


@app.get("/analytics/revenue-at-risk")
def revenue_at_risk():
    """Returns monthly revenue at risk, broken down by segment plus a combined High+Medium total."""
    return db_get_risk_summary()


@app.get("/analytics/top-features")
def top_features(limit: int = 15):
    """Returns the top churn-driving features by model feature importance."""
    df = pd.read_csv("data/processed/rf_feature_importance.csv")
    df = df.sort_values("importance", ascending=False).head(limit)
    return _df_to_records(df)


@app.get("/analytics/retention-targets")
def retention_targets():
    """Returns retention priority tier counts and revenue exposure."""
    df = pd.read_csv("data/processed/retention_targets.csv")
    summary = (
        df.groupby("priority_tier")
        .agg(
            customer_count=("customer_id", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            total_monthly_revenue=("monthly_charges", "sum"),
        )
        .reset_index()
    )
    return _df_to_records(summary)
