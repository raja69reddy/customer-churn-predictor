"""Re-scores only customers whose churn_predictions are more than N days stale."""

import argparse
import os
from datetime import datetime

import mlflow
import pandas as pd
from sqlalchemy import text

from src.models import mlflow_setup
from src.models.batch_scorer import load_production_predictor
from src.utils.db import get_engine

STALE_DAYS = 7


def load_stale_customers(engine, stale_days: int = STALE_DAYS) -> pd.DataFrame:
    query = text("""
        SELECT rc.*
        FROM raw_customers rc
        JOIN churn_predictions cp ON cp.customer_id = rc.customer_id
        WHERE cp.predicted_at < NOW() - (:stale_days || ' days')::INTERVAL
        """)
    return pd.read_sql(query, engine, params={"stale_days": stale_days})


def run(stale_days: int = STALE_DAYS) -> pd.DataFrame:
    predictor, version = load_production_predictor()
    print(f"Using model version {version} from MLflow registry")

    engine = get_engine()
    stale_customers = load_stale_customers(engine, stale_days)
    print(
        f"Found {len(stale_customers)} customers scored more than {stale_days} days ago"
    )

    if stale_customers.empty:
        print("Re-scored 0 customers")
        return stale_customers

    scored = predictor.predict_batch(stale_customers)
    stale_ids = list(scored["customer_id"])

    with engine.begin() as conn:
        # Archive the about-to-be-overwritten rows so score drift/segment-transition queries
        # (sql/queries/prediction_analysis.sql) have a "previous" prediction to compare against.
        conn.execute(
            text(
                "INSERT INTO churn_predictions_history "
                "(customer_id, churn_probability, risk_segment, model_version, predicted_at) "
                "SELECT customer_id, churn_probability, risk_segment, model_version, predicted_at "
                "FROM churn_predictions WHERE customer_id = ANY(:ids)"
            ),
            {"ids": stale_ids},
        )

        for _, row in scored.iterrows():
            conn.execute(
                text(
                    "UPDATE churn_predictions "
                    "SET churn_probability = :prob, risk_segment = :segment, "
                    "model_version = :version, predicted_at = NOW() "
                    "WHERE customer_id = :cid"
                ),
                {
                    "prob": float(row["churn_probability"]),
                    "segment": row["risk_segment"],
                    "version": predictor.model_name,
                    "cid": row["customer_id"],
                },
            )

    print(f"Re-scored {len(scored)} customers")

    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/incremental_scores_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
    scored[["customer_id", "churn_probability", "risk_segment"]].to_csv(
        out_path, index=False
    )
    print(f"Saved incremental results to {out_path}")

    mlflow_setup.setup_experiment()
    with mlflow.start_run(run_name="incremental_scoring"):
        mlflow.log_param("stale_days", stale_days)
        mlflow.log_param("model_version", predictor.model_name)
        mlflow.log_metric("customers_rescored", len(scored))
    print("Logged incremental run to MLflow")

    return scored


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score stale churn predictions.")
    parser.add_argument("--stale-days", type=int, default=STALE_DAYS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.stale_days)
