"""Reusable PostgreSQL data-access helpers for the churn prediction API."""
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine


def _records(df: pd.DataFrame) -> list[dict]:
    """Converts a DataFrame to JSON-safe records, turning NaN/NaT into None."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def get_customer_from_db(customer_id: str) -> Optional[dict]:
    """Returns a customer's raw_customers row as a dict, or None if not found."""
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM raw_customers WHERE customer_id = %(cid)s", engine, params={"cid": customer_id}
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def save_prediction_to_db(prediction: dict) -> None:
    """Archives any existing prediction for this customer into history, then saves the new one."""
    engine = get_engine()
    customer_id = prediction["customer_id"]

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO churn_predictions_history "
                "(customer_id, churn_probability, risk_segment, model_version, predicted_at) "
                "SELECT customer_id, churn_probability, risk_segment, model_version, predicted_at "
                "FROM churn_predictions WHERE customer_id = :cid"
            ),
            {"cid": customer_id},
        )
        conn.execute(text("DELETE FROM churn_predictions WHERE customer_id = :cid"), {"cid": customer_id})
        conn.execute(
            text(
                "INSERT INTO churn_predictions "
                "(customer_id, churn_probability, risk_segment, model_version, predicted_at) "
                "VALUES (:cid, :prob, :segment, :version, :predicted_at)"
            ),
            {
                "cid": customer_id,
                "prob": prediction["churn_probability"],
                "segment": prediction["risk_segment"],
                "version": prediction["model_version"],
                "predicted_at": prediction.get("predicted_at") or datetime.now(),
            },
        )


def get_high_risk_customers(limit: int = 100) -> list[dict]:
    """Returns high-risk customers joined with their profile details, sorted by probability."""
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM vw_churn_predictions WHERE risk_segment = 'High' "
        "ORDER BY churn_probability DESC LIMIT %(limit)s",
        engine, params={"limit": limit},
    )
    return _records(df)


def get_risk_summary() -> list[dict]:
    """Returns per-segment counts, averages, and revenue-at-risk from vw_risk_segments."""
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM vw_risk_segments", engine)
    return _records(df)


def get_churn_rate() -> dict:
    """Returns the overall historical churn rate from raw_customers."""
    engine = get_engine()
    df = pd.read_sql("SELECT churn FROM raw_customers", engine)
    rate = (df["churn"] == "Yes").mean()
    return {
        "churn_rate": round(float(rate), 4),
        "churn_rate_pct": round(float(rate) * 100, 2),
        "total_customers": len(df),
    }
