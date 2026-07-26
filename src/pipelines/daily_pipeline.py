"""Daily MLOps pipeline — the single entrypoint meant to be run on a schedule (cron / Windows
Task Scheduler) once a day. Orchestrates ingestion, scoring, drift detection, conditional
retraining, and a daily report, all logged to one MLflow run.

Data-source note: this project's synthetic dataset (mock_data/gen_customers.py) is generated
with a fixed seed, so data/raw/customers.csv never actually grows new customer_ids between runs
— Step 1/2 below will typically report 0 new customers, which is the correct result given this
static dataset, not a bug. In a real deployment, data/raw/customers.csv (or wherever new exports
land) would be refreshed by an upstream system before this pipeline runs.

Requires the full ML stack — not demo-mode safe.
"""

import os
from datetime import datetime

import mlflow
import pandas as pd

from src.data import ingestion
from src.models import (
    batch_scorer,
    mlflow_setup,
    update_prediction_schema,
    update_registry,
)
from src.models.retrain_pipeline import check_retrain_needed, save_retrain_report
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("daily_pipeline")

OUTPUT_DIR = "data/processed"
RAW_CSV_PATH = "data/raw/customers.csv"


def check_for_new_data(csv_path: str = RAW_CSV_PATH) -> dict:
    """Step 1: compares customer_ids in the source CSV against raw_customers to see if
    there's new data to ingest."""
    csv_ids = set(pd.read_csv(csv_path, usecols=["customer_id"])["customer_id"])

    engine = get_engine()
    existing_ids = set(
        pd.read_sql("SELECT customer_id FROM raw_customers", engine)["customer_id"]
    )

    new_ids = csv_ids - existing_ids
    result = {"new_customer_count": len(new_ids), "has_new_data": len(new_ids) > 0}
    log.info(
        "check_for_new_data: %d new customer_id(s) found in %s",
        result["new_customer_count"],
        csv_path,
    )
    return result


def run_incremental_ingestion(csv_path: str = RAW_CSV_PATH) -> int:
    """Step 2: ingests any new rows found in check_for_new_data() via the existing incremental
    ingestion mode (a no-op insert of 0 rows if there's nothing new)."""
    inserted = ingestion.ingest(csv_path, mode="incremental")
    log.info("run_incremental_ingestion: inserted %d new row(s)", inserted)
    return inserted


def rescore_all_customers() -> None:
    """Step 3: re-scores every customer with the currently active model, then restores the
    retention_priority/recommended_action/estimated_revenue_at_risk/days_since_last_score
    columns that a full batch_scorer run always wipes to NULL (see
    src/models/update_prediction_schema.py) — folding that previously-manual post-batch-scoring
    step into the pipeline itself."""
    batch_scorer.run(mode="full")
    update_prediction_schema.run()
    log.info(
        "rescore_all_customers: full rescore + prediction-schema backfill complete"
    )


def check_drift() -> dict:
    """Step 4: reuses src.models.retrain_pipeline.check_retrain_needed(), which itself reuses
    src.monitoring.model_monitor's PSI drift report."""
    drift = check_retrain_needed()
    log.info(
        "check_drift: retrain_needed=%s (max_psi=%.4f)",
        drift["retrain_needed"],
        drift["max_psi"],
    )
    return drift


def retrain_if_needed(drift: dict) -> dict:
    """Step 5: retrains and auto-promotes-if-better only when check_drift() says it's needed.
    Skips (and says so) otherwise, to avoid an expensive GridSearchCV retrain every single day
    when nothing has actually drifted."""
    if not drift["retrain_needed"]:
        log.info("retrain_if_needed: skipped — no drift detected above threshold")
        return {"retrained": False, "reason": "no drift detected"}

    result = save_retrain_report()
    update_registry.run()
    log.info("retrain_if_needed: retrained — promoted=%s", result["promoted"])
    return {"retrained": True, "promoted": result["promoted"], "report": result}


def generate_daily_report(steps: dict, output_dir: str = OUTPUT_DIR) -> str:
    """Step 6: saves a markdown summary of everything the pipeline did today."""
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        f"# Daily Pipeline Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Step 1-2: New Data / Ingestion",
        f"- New customers found: {steps['new_data']['new_customer_count']}",
        f"- Rows ingested: {steps['ingested_rows']}",
        "",
        "## Step 3: Rescoring",
        "- All customers rescored and prediction schema backfilled.",
        "",
        "## Step 4: Drift Check",
        f"- Retrain needed: {steps['drift']['retrain_needed']} "
        f"(max PSI={steps['drift']['max_psi']}, threshold={steps['drift']['threshold']})",
        "",
        "## Step 5: Retraining",
    ]
    if steps["retrain"]["retrained"]:
        lines.append(f"- Retrained — promoted: {steps['retrain']['promoted']}")
    else:
        lines.append(f"- Skipped — {steps['retrain']['reason']}")

    report_path = os.path.join(output_dir, "daily_pipeline_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("generate_daily_report: saved to %s", report_path)
    return report_path


def run() -> dict:
    """Runs the full daily pipeline end-to-end, then logs a summary to MLflow.

    Note: steps aren't wrapped in one shared MLflow run — batch_scorer.run() and
    retrain_pipeline's own logging each open their own top-level `mlflow.start_run()`
    internally (not nested=True), so trying to nest everything under one outer run here
    raises "Run already active". Instead, each step logs its own run as it already did
    before this pipeline existed, and this function additionally logs one orchestration-level
    summary run once every step has finished."""
    new_data = check_for_new_data()
    inserted_rows = run_incremental_ingestion()
    rescore_all_customers()
    drift = check_drift()
    retrain_result = retrain_if_needed(drift)

    steps = {
        "new_data": new_data,
        "ingested_rows": inserted_rows,
        "drift": drift,
        "retrain": retrain_result,
    }
    report_path = generate_daily_report(steps)

    mlflow_setup.setup_experiment()
    with mlflow.start_run(run_name="daily_pipeline_summary"):
        mlflow.log_metric("new_customer_count", new_data["new_customer_count"])
        mlflow.log_metric("ingested_rows", inserted_rows)
        mlflow.log_metric("drift_max_psi", drift["max_psi"])
        mlflow.log_param("retrain_needed", drift["retrain_needed"])
        mlflow.log_param("retrained", retrain_result["retrained"])
        if retrain_result["retrained"]:
            mlflow.log_param("promoted", retrain_result["promoted"])
        mlflow.log_artifact(report_path)
        run_id = mlflow.active_run().info.run_id

    log.info("Daily pipeline complete — MLflow summary run_id=%s", run_id)
    return {**steps, "report_path": report_path, "mlflow_run_id": run_id}


if __name__ == "__main__":
    result = run()

    print("=== Daily Pipeline Summary ===")
    print(f"New customers found: {result['new_data']['new_customer_count']}")
    print(f"Rows ingested: {result['ingested_rows']}")
    print(
        f"Drift check: retrain_needed={result['drift']['retrain_needed']} "
        f"(max_psi={result['drift']['max_psi']})"
    )
    if result["retrain"]["retrained"]:
        print(f"Retrained — promoted: {result['retrain']['promoted']}")
    else:
        print(f"Retrain skipped — {result['retrain']['reason']}")
    print(f"MLflow run: {result['mlflow_run_id']}")
    print(f"Saved report to {result['report_path']}")
