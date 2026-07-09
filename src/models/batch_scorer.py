"""Production batch-scoring pipeline — scores customers using the MLflow Production model.

Usage:
    python -m src.models.batch_scorer --mode full --output both
    python -m src.models.batch_scorer --mode new --output db
    python -m src.models.batch_scorer --mode full --threshold 0.6 --output csv
"""
import argparse
import logging
import os
import time
from datetime import date, datetime

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import text
from tqdm import tqdm

from src.models import mlflow_registry, mlflow_setup
from src.models.predict import ChurnPredictor
from src.utils.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 500
DEFAULT_HIGH_THRESHOLD = 0.7
DEFAULT_MEDIUM_THRESHOLD = 0.4


def load_production_predictor() -> tuple[ChurnPredictor, str]:
    mlflow_setup.configure_tracking()
    client = MlflowClient()
    model_name = mlflow_registry.REGISTERED_MODEL_NAME

    versions = client.search_model_versions(f"name='{model_name}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        raise RuntimeError(f"No Production version found for '{model_name}' in MLflow registry.")
    version = production[0].version

    predictor = ChurnPredictor()
    predictor.model = mlflow_registry.get_production_model(model_name)
    predictor.model_name = f"{model_name}_v{version}"
    return predictor, version


def classify_risk(probability: float, high_threshold: float, medium_threshold: float) -> str:
    if probability >= high_threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


def load_customers_to_score(engine, mode: str) -> pd.DataFrame:
    if mode == "full":
        return pd.read_sql("SELECT * FROM raw_customers", engine)

    query = """
        SELECT rc.*
        FROM raw_customers rc
        LEFT JOIN churn_predictions cp ON cp.customer_id = rc.customer_id
        WHERE cp.customer_id IS NULL
    """
    return pd.read_sql(query, engine)


def score_in_chunks(predictor: ChurnPredictor, customers: pd.DataFrame, high_threshold: float, medium_threshold: float) -> pd.DataFrame:
    chunks = [customers.iloc[i:i + CHUNK_SIZE] for i in range(0, len(customers), CHUNK_SIZE)]
    scored_chunks = []

    with tqdm(total=len(customers), desc="Scoring customers", unit="cust") as pbar:
        for chunk in chunks:
            scored_chunk = predictor.predict_batch(chunk)
            scored_chunks.append(scored_chunk)
            pbar.update(len(chunk))

    scored = pd.concat(scored_chunks, ignore_index=True)
    scored["risk_segment"] = scored["churn_probability"].apply(
        lambda p: classify_risk(p, high_threshold, medium_threshold)
    )
    return scored


def save_to_csv(scored: pd.DataFrame, model_version: str) -> str:
    out_dir = "data/processed/predictions"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"predictions_{date.today().isoformat()}.csv")

    to_save = scored[["customer_id", "churn_probability", "risk_segment"]].copy()
    to_save["model_version"] = model_version
    to_save.to_csv(path, index=False)
    return path


def save_to_db(engine, scored: pd.DataFrame, model_version: str, mode: str) -> int:
    to_save = scored[["customer_id", "churn_probability", "risk_segment"]].copy()
    to_save["model_version"] = model_version

    with engine.begin() as conn:
        if mode == "full":
            # Archive the outgoing baseline before wiping it, so drift/segment-transition
            # queries (sql/queries/prediction_analysis.sql) can compare against it.
            conn.execute(
                text(
                    "INSERT INTO churn_predictions_history "
                    "(customer_id, churn_probability, risk_segment, model_version, predicted_at) "
                    "SELECT customer_id, churn_probability, risk_segment, model_version, predicted_at "
                    "FROM churn_predictions"
                )
            )
            conn.execute(text("TRUNCATE TABLE churn_predictions"))
        else:
            customer_ids = list(to_save["customer_id"])
            if customer_ids:
                conn.execute(
                    text("DELETE FROM churn_predictions WHERE customer_id = ANY(:ids)"),
                    {"ids": customer_ids},
                )
        to_save.to_sql("churn_predictions", conn, if_exists="append", index=False)

    return len(to_save)


def run(mode: str = "full", threshold: float = None, output: str = "both") -> None:
    start = time.time()
    log.info("Starting batch scoring — mode=%s, threshold=%s, output=%s", mode, threshold, output)

    high_threshold = threshold if threshold is not None else DEFAULT_HIGH_THRESHOLD
    medium_threshold = min(DEFAULT_MEDIUM_THRESHOLD, high_threshold / 2) if threshold is not None else DEFAULT_MEDIUM_THRESHOLD

    predictor, version = load_production_predictor()
    log.info("Using model version %s from MLflow registry", version)

    engine = get_engine()
    customers = load_customers_to_score(engine, mode)
    log.info("Loaded %d customers to score (mode=%s)", len(customers), mode)

    if customers.empty:
        log.info("No customers to score — nothing to do.")
        return

    scored = score_in_chunks(predictor, customers, high_threshold, medium_threshold)
    log.info("Scored %d customers", len(scored))

    distribution = scored["risk_segment"].value_counts()
    high = int(distribution.get("High", 0))
    medium = int(distribution.get("Medium", 0))
    low = int(distribution.get("Low", 0))

    csv_path = None
    db_rows = None
    if output in ("csv", "both"):
        csv_path = save_to_csv(scored, predictor.model_name)
        log.info("Saved predictions to %s", csv_path)
    if output in ("db", "both"):
        db_rows = save_to_db(engine, scored, predictor.model_name, mode)
        log.info("Saved %d predictions to churn_predictions table", db_rows)

    elapsed = time.time() - start

    mlflow_setup.setup_experiment()
    with mlflow.start_run(run_name="batch_scoring"):
        mlflow.log_params({"mode": mode, "threshold": high_threshold, "output": output, "model_version": predictor.model_name})
        mlflow.log_metric("customers_scored", len(scored))
        mlflow.log_metric("high_risk_count", high)
        mlflow.log_metric("medium_risk_count", medium)
        mlflow.log_metric("low_risk_count", low)
        mlflow.log_metric("elapsed_seconds", elapsed)
        mlflow.set_tag("run_type", "batch_scoring")
    log.info("Logged batch scoring run to MLflow")

    summary = pd.DataFrame([
        {"metric": "mode", "value": mode},
        {"metric": "output", "value": output},
        {"metric": "high_threshold", "value": high_threshold},
        {"metric": "medium_threshold", "value": medium_threshold},
        {"metric": "customers_scored", "value": len(scored)},
        {"metric": "high_risk", "value": high},
        {"metric": "medium_risk", "value": medium},
        {"metric": "low_risk", "value": low},
        {"metric": "csv_path", "value": csv_path or "-"},
        {"metric": "db_rows_saved", "value": db_rows if db_rows is not None else "-"},
        {"metric": "elapsed_seconds", "value": round(elapsed, 2)},
    ])
    print("\nBatch Scoring Summary")
    print("=" * 40)
    print(summary.to_string(index=False))

    top10 = scored.sort_values("churn_probability", ascending=False).head(10)
    print("\nTop 10 highest risk customers:")
    print(top10[["customer_id", "churn_probability", "risk_segment"]].to_string(index=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-score customers for churn risk.")
    parser.add_argument("--mode", choices=["full", "new"], default="full", help="Score all customers or only unscored ones.")
    parser.add_argument("--threshold", type=float, default=None, help="Custom High-risk probability threshold (default 0.7).")
    parser.add_argument("--output", choices=["csv", "db", "both"], default="both", help="Where to save results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(mode=args.mode, threshold=args.threshold, output=args.output)
