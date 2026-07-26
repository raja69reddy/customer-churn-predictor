"""Automated retraining pipeline — checks for model drift, retrains XGBoost on the latest
processed data, compares it against the currently active model, and promotes it only if it's
a genuine improvement.

This is an orchestration layer over two modules that already exist: src/monitoring/model_monitor.py
(the PSI drift check used as the "is retraining needed" signal) and src/models/train.py's
ModelTrainer (the actual training/tuning logic, unchanged from Day 6/7). Every retrain attempt
(promoted or not) is logged to both MLflow and model_registry, matching the existing convention
that model_registry accumulates a full history of every training run, not just the active one.

Requires the full ML stack — not demo-mode safe.
"""

import os
from datetime import datetime

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sqlalchemy import text

from src.models import mlflow_setup
from src.models.train import ModelTrainer
from src.monitoring.model_monitor import (
    DEFAULT_ALERT_THRESHOLD,
    generate_monitoring_report,
)
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("retrain_pipeline")

OUTPUT_DIR = "data/processed"
MODEL_VERSION = (
    "v3"  # v1=baseline xgboost, v2=manually tuned (Day 6/7), v3=auto-retrained
)
DEFAULT_PROMOTE_THRESHOLD = 0.01


def check_retrain_needed(drift_threshold: float = DEFAULT_ALERT_THRESHOLD) -> dict:
    """Returns whether retraining is recommended, based on model_monitor's PSI drift report
    (prediction-score drift or any feature drift exceeding drift_threshold)."""
    report = generate_monitoring_report()
    max_feature_psi = float(report["feature_drift"]["psi"].max())
    max_psi = max(report["prediction_drift"]["psi"], max_feature_psi)

    result = {
        "retrain_needed": max_psi > drift_threshold,
        "max_psi": round(max_psi, 4),
        "threshold": drift_threshold,
        "prediction_psi": report["prediction_drift"]["psi"],
        "overall_status": report["overall_status"],
    }
    log.info(
        "check_retrain_needed: max_psi=%.4f threshold=%.2f -> retrain_needed=%s",
        result["max_psi"],
        drift_threshold,
        result["retrain_needed"],
    )
    return result


def retrain_best_model() -> tuple:
    """Retrains XGBoost (the model type behind every 'xgboost_tuned' active model so far) on
    the latest processed_customers data. Returns (new_model, trainer, best_params)."""
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    log.info("Retraining XGBoost on %d rows (latest processed_customers)", len(df))
    new_model, params = trainer.tune_xgboost()
    log.info("Retrain complete — new model trained with params=%s", params)
    return new_model, trainer, params


def _load_active_model() -> tuple:
    engine = get_engine()
    active = pd.read_sql(
        "SELECT model_name FROM model_registry WHERE is_active = TRUE LIMIT 1", engine
    )
    if active.empty:
        raise RuntimeError("No active model found in model_registry.")
    model_name = active.iloc[0]["model_name"]

    trainer = ModelTrainer()
    trainer.load_processed_data()
    old_model = trainer.load_model(model_name)
    return old_model, model_name


def compare_old_vs_new(old_model, new_model, trainer: ModelTrainer = None) -> dict:
    """Scores both models on the same current test split and returns a metrics comparison."""
    if trainer is None:
        trainer = ModelTrainer()
        df = trainer.load_processed_data()
        trainer.split_data(df)

    def _metrics(model):
        y_pred = model.predict(trainer.X_test)
        y_proba = model.predict_proba(trainer.X_test)[:, 1]
        return {
            "accuracy": round(float(accuracy_score(trainer.y_test, y_pred)), 4),
            "auc": round(float(roc_auc_score(trainer.y_test, y_proba)), 4),
            "f1": round(float(f1_score(trainer.y_test, y_pred)), 4),
        }

    old_metrics = _metrics(old_model)
    new_metrics = _metrics(new_model)
    result = {
        "old": old_metrics,
        "new": new_metrics,
        "auc_delta": round(new_metrics["auc"] - old_metrics["auc"], 4),
    }
    log.info(
        "compare_old_vs_new: old_auc=%.4f new_auc=%.4f delta=%+.4f",
        old_metrics["auc"],
        new_metrics["auc"],
        result["auc_delta"],
    )
    return result


def promote_if_better(
    new_model, comparison: dict, threshold: float = DEFAULT_PROMOTE_THRESHOLD
) -> bool:
    """Promotes new_model to the deployable model artifact (overwrites models/xgboost_tuned.pkl)
    only if its AUC beats the currently active model's AUC by more than `threshold`. Does not
    touch model_registry — see log_retrain_run() for that. Returns whether promotion happened.
    """
    promoted = comparison["auc_delta"] > threshold
    if not promoted:
        log.info(
            "Not promoting — AUC delta %+.4f does not exceed threshold %.2f",
            comparison["auc_delta"],
            threshold,
        )
        return False

    trainer = ModelTrainer()
    trainer.save_model(new_model, "xgboost_tuned")
    log.info(
        "Promoted new model artifact — AUC delta %+.4f exceeds threshold %.2f",
        comparison["auc_delta"],
        threshold,
    )
    return True


def log_retrain_run(comparison: dict, promoted: bool, params: dict = None) -> dict:
    """Logs the retrain attempt to both MLflow (metrics/params/promoted tag) and model_registry
    (a new row, is_active=promoted — every retrain attempt is recorded, not just successful
    promotions, matching the existing model_registry convention of keeping full training history).
    """
    mlflow_setup.setup_experiment()
    with mlflow.start_run(run_name="retrain_pipeline"):
        mlflow.log_params(params or {})
        mlflow.log_metrics(
            {
                "old_auc": comparison["old"]["auc"],
                "new_auc": comparison["new"]["auc"],
                "auc_delta": comparison["auc_delta"],
            }
        )
        mlflow.set_tag("promoted", str(promoted))
        mlflow_run_id = mlflow.active_run().info.run_id

    engine = get_engine()
    with engine.begin() as conn:
        if promoted:
            conn.execute(text("UPDATE model_registry SET is_active = FALSE"))
        conn.execute(
            text(
                "INSERT INTO model_registry "
                "(model_name, model_version, accuracy, auc_score, f1_score, "
                "trained_at, is_active) "
                "VALUES (:name, :version, :accuracy, :auc, :f1, :trained_at, :is_active)"
            ),
            {
                "name": "xgboost_tuned",
                "version": MODEL_VERSION,
                "accuracy": comparison["new"]["accuracy"],
                "auc": comparison["new"]["auc"],
                "f1": comparison["new"]["f1"],
                "trained_at": datetime.now(),
                "is_active": promoted,
            },
        )

    log.info(
        "Logged retrain run to MLflow (run_id=%s) and model_registry (promoted=%s)",
        mlflow_run_id,
        promoted,
    )
    return {"mlflow_run_id": mlflow_run_id, "promoted": promoted}


def save_retrain_report(
    threshold: float = DEFAULT_PROMOTE_THRESHOLD, output_dir: str = OUTPUT_DIR
) -> dict:
    """Runs the full retrain pipeline end-to-end and saves a markdown report to output_dir:
    check drift -> retrain -> compare -> promote-if-better -> log. Always retrains and compares
    (regardless of the drift check) so the report shows real numbers every time it's run; the
    drift check result is included for context but doesn't gate the retrain in this function —
    see check_retrain_needed() to gate retraining on drift in an external scheduler."""
    os.makedirs(output_dir, exist_ok=True)

    drift = check_retrain_needed()
    old_model, old_model_name = _load_active_model()
    new_model, trainer, params = retrain_best_model()
    comparison = compare_old_vs_new(old_model, new_model, trainer)
    promoted = promote_if_better(new_model, comparison, threshold)
    log_result = log_retrain_run(comparison, promoted, params)

    lines = [
        "# Retrain Pipeline Report",
        "",
        f"Previously active model: {old_model_name}",
        f"Drift check: max_psi={drift['max_psi']} (threshold={drift['threshold']}) — "
        f"retrain_needed={drift['retrain_needed']}",
        "",
        "## Old vs New Metrics",
        f"- Old: accuracy={comparison['old']['accuracy']}, auc={comparison['old']['auc']}, "
        f"f1={comparison['old']['f1']}",
        f"- New: accuracy={comparison['new']['accuracy']}, auc={comparison['new']['auc']}, "
        f"f1={comparison['new']['f1']}",
        f"- AUC delta: {comparison['auc_delta']:+.4f} (promote threshold: {threshold})",
        "",
        f"## Decision: {'PROMOTED' if promoted else 'NOT PROMOTED'}",
        f"MLflow run: {log_result['mlflow_run_id']}",
    ]

    report_path = os.path.join(output_dir, "retrain_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved retrain report to %s", report_path)
    return {
        "report_path": report_path,
        "drift": drift,
        "comparison": comparison,
        "promoted": promoted,
        "mlflow_run_id": log_result["mlflow_run_id"],
    }


if __name__ == "__main__":
    result = save_retrain_report()

    print("=== Drift Check ===")
    for key, value in result["drift"].items():
        print(f"  {key}: {value}")

    print("\n=== Old vs New Metrics ===")
    print(f"  Old: {result['comparison']['old']}")
    print(f"  New: {result['comparison']['new']}")
    print(f"  AUC delta: {result['comparison']['auc_delta']:+.4f}")

    print(f"\nPromoted: {result['promoted']}")
    print(f"MLflow run: {result['mlflow_run_id']}")
    print(f"Saved report to {result['report_path']}")
