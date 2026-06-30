"""FastAPI prediction service — placeholder, to be implemented on Day 5."""
from fastapi import FastAPI

app = FastAPI(title="Customer Churn Predictor API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
