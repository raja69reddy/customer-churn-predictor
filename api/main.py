"""FastAPI prediction service for the customer churn predictor."""
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import BatchPredictionInput, BatchPredictionOutput, CustomerInput, PredictionOutput
from api.predictor import APIPredictor
from src.utils.db import get_engine

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Converts a DataFrame to JSON-safe records, turning NaN/NaT into None."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


CUSTOMER_INPUT_FIELDS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure", "phone_service",
    "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method", "monthly_charges", "total_charges",
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
    log.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred while processing your request. Please try again later."},
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

@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Scores a single customer for churn risk."""
    return predictor.predict(customer)


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(payload: BatchPredictionInput):
    """Scores a batch of customers for churn risk."""
    predictions = predictor.predict_batch(payload.customers)
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

def _load_customer_row(customer_id: str) -> pd.Series:
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM raw_customers WHERE customer_id = %(cid)s", engine, params={"cid": customer_id}
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    return df.iloc[0]


@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):
    """Returns raw profile details for a single customer."""
    customer = _load_customer_row(customer_id)
    return customer.to_dict()


@app.get("/customer/{customer_id}/predict", response_model=PredictionOutput)
def predict_customer(customer_id: str):
    """Scores an existing customer (by ID) for churn risk."""
    customer = _load_customer_row(customer_id)
    customer_input = CustomerInput(customer_id=customer_id, **{f: customer[f] for f in CUSTOMER_INPUT_FIELDS})
    return predictor.predict(customer_input)


@app.get("/customer/{customer_id}/explanation")
def explain_customer(customer_id: str):
    """Returns an AI-generated (or template-fallback) explanation of an existing customer's churn risk."""
    customer = _load_customer_row(customer_id)
    customer_dict = {f: customer[f] for f in CUSTOMER_INPUT_FIELDS}

    result = predictor.explain(customer_dict)
    return {"customer_id": customer_id, **result}


@app.get("/customers/high-risk")
def high_risk_customers(limit: int = 100):
    """Returns high-risk customers, sorted by churn probability descending."""
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM vw_churn_predictions WHERE risk_segment = 'High' "
        "ORDER BY churn_probability DESC LIMIT %(limit)s",
        engine, params={"limit": limit},
    )
    return _df_to_records(df)


@app.get("/customers/risk-summary")
def risk_summary():
    """Returns per-segment counts, average churn probability, and revenue at risk."""
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM vw_risk_segments", engine)
    return _df_to_records(df)
