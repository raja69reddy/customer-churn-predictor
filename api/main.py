"""FastAPI prediction service for the customer churn predictor."""
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import BatchPredictionInput, BatchPredictionOutput, CustomerInput, PredictionOutput
from api.predictor import APIPredictor
from src.utils.db import get_engine

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
    return df.to_dict(orient="records")
