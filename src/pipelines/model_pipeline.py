"""Model lifecycle pipeline — AUC-threshold performance check, conditional retraining,
validation, deployment, and rollback.

Distinct from Day 27's src/models/retrain_pipeline.py (PSI-drift-triggered, does its own
train/compare/promote in one shot): this module is AUC-threshold-triggered and exposes
retrain/validate/deploy/rollback as separate, individually-callable lifecycle steps — the kind
of building blocks a CI/CD-style model pipeline would orchestrate. Reuses
retrain_pipeline.retrain_best_model() for the actual training, not a reimplementation.

Rollback caveat (important, verified against this project's actual file layout): models/*.pkl
is a single mutable file per model_name, overwritten on every save — model_registry rows are
metadata (metrics + an is_active flag), not versioned weight artifacts. rollback_model()
therefore reverts *which registry row is marked active* (i.e. which metrics/version are treated
as current), not the actual bytes in models/<name>.pkl. In practice this project's retrains are
deterministic (same data + fixed random_state, see Day 27 notes), so repeated
xgboost_tuned rollbacks/retrains tend to converge on the same weights anyway — but this isn't
guaranteed for a genuinely different historical model file that's since been overwritten.

Requires the full ML stack — not demo-mode safe.
"""

import time
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sqlalchemy import text

from src.models import mlflow_setup
from src.models.retrain_pipeline import retrain_best_model
from src.models.train import ModelTrainer
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging
from src.utils.pipeline_tracking import record_pipeline_run

log = setup_logging("model_pipeline")

DEFAULT_AUC_THRESHOLD = 0.85
DEPLOY_MODEL_VERSION = "v4"


def check_model_performance(threshold: float = DEFAULT_AUC_THRESHOLD) -> dict:
    """Checks the currently active model's AUC (from model_registry) against threshold."""
    engine = get_engine()
    active = pd.read_sql(
        "SELECT id, model_name, auc_score FROM model_registry WHERE is_active = TRUE LIMIT 1",
        engine,
    )
    if active.empty:
        raise RuntimeError("No active model found in model_registry.")

    row = active.iloc[0]
    result = {
        "model_name": row["model_name"],
        "registry_id": int(row["id"]),
        "auc_score": float(row["auc_score"]),
        "threshold": threshold,
        "performance_ok": float(row["auc_score"]) >= threshold,
    }
    log.info(
        "check_model_performance: %s AUC=%.4f threshold=%.2f -> ok=%s",
        result["model_name"],
        result["auc_score"],
        threshold,
        result["performance_ok"],
    )
    return result


def trigger_retraining(performance: dict = None) -> dict:
    """Retrains only if check_model_performance() found performance below threshold."""
    if performance is None:
        performance = check_model_performance()

    if performance["performance_ok"]:
        log.info("trigger_retraining: skipped — performance is above threshold")
        return {"retrained": False, "reason": "performance above threshold"}

    new_model, trainer, params = retrain_best_model()
    log.info(
        "trigger_retraining: retrained a new model (performance was below threshold)"
    )
    return {"retrained": True, "model": new_model, "trainer": trainer, "params": params}


def validate_new_model(
    model, trainer: ModelTrainer, min_auc: float = DEFAULT_AUC_THRESHOLD
) -> dict:
    """Runs basic validation checks on a newly trained model before it's eligible for
    deployment: AUC above min_auc, predictions are valid probabilities, no NaN/inf outputs.
    """
    y_proba = model.predict_proba(trainer.X_test)[:, 1]
    auc = float(roc_auc_score(trainer.y_test, y_proba))

    checks = {
        "auc_above_minimum": auc >= min_auc,
        "no_nan_predictions": not bool(np.isnan(y_proba).any()),
        "no_inf_predictions": not bool(np.isinf(y_proba).any()),
        "predictions_in_valid_range": bool(((y_proba >= 0) & (y_proba <= 1)).all()),
    }
    passed = all(checks.values())

    result = {"auc": round(auc, 4), "checks": checks, "passed": passed}
    log.info("validate_new_model: auc=%.4f passed=%s checks=%s", auc, passed, checks)
    return result


def deploy_new_model(
    model,
    validation: dict,
    model_name: str = "xgboost_tuned",
    model_version: str = DEPLOY_MODEL_VERSION,
) -> dict:
    """Promotes a validated model to production: overwrites models/<model_name>.pkl and
    inserts a new active model_registry row (deactivating all others). Refuses to deploy if
    validation didn't pass."""
    if not validation["passed"]:
        log.warning(
            "deploy_new_model: refused — validation did not pass (%s)", validation
        )
        return {"deployed": False, "reason": "validation failed"}

    trainer = ModelTrainer()
    trainer.save_model(model, model_name)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE model_registry SET is_active = FALSE"))
        result = conn.execute(
            text(
                "INSERT INTO model_registry "
                "(model_name, model_version, accuracy, auc_score, f1_score, "
                "trained_at, is_active) "
                "VALUES (:name, :version, :accuracy, :auc, :f1, :trained_at, TRUE) "
                "RETURNING id"
            ),
            {
                "name": model_name,
                "version": model_version,
                "accuracy": 0.0,
                "auc": validation["auc"],
                "f1": 0.0,
                "trained_at": datetime.now(),
            },
        )
        registry_id = result.scalar()

    log.info(
        "deploy_new_model: deployed %s (registry_id=%d, AUC=%.4f)",
        model_name,
        registry_id,
        validation["auc"],
    )
    return {"deployed": True, "registry_id": int(registry_id), "model_name": model_name}


def rollback_model() -> dict:
    """Reverts model_registry.is_active to the previous most-recently-trained row (see module
    docstring for the important caveat: this reverts the active *registry entry*, not
    necessarily the exact historical weights on disk, since models/<name>.pkl is overwritten
    in place on every save)."""
    engine = get_engine()
    registry = pd.read_sql(
        "SELECT id, model_name, model_version, auc_score, trained_at, is_active "
        "FROM model_registry ORDER BY trained_at DESC",
        engine,
    )
    if len(registry) < 2:
        raise RuntimeError("Need at least 2 model_registry rows to roll back.")

    current = registry[registry["is_active"]].iloc[0]
    previous = registry[registry["id"] != current["id"]].iloc[0]

    with engine.begin() as conn:
        conn.execute(text("UPDATE model_registry SET is_active = FALSE"))
        conn.execute(
            text("UPDATE model_registry SET is_active = TRUE WHERE id = :id"),
            {"id": int(previous["id"])},
        )

    result = {
        "rolled_back_from": current["model_name"],
        "rolled_back_from_id": int(current["id"]),
        "rolled_back_to": previous["model_name"],
        "rolled_back_to_id": int(previous["id"]),
        "rolled_back_to_auc": float(previous["auc_score"]),
    }
    log.info(
        "rollback_model: %s (id=%d) -> %s (id=%d)",
        result["rolled_back_from"],
        result["rolled_back_from_id"],
        result["rolled_back_to"],
        result["rolled_back_to_id"],
    )
    return result


def log_pipeline_run(
    performance: dict,
    retrain_result: dict,
    deploy_result: dict = None,
    duration_seconds: float = 0.0,
) -> str:
    """Logs the full model pipeline run (check -> retrain -> validate -> deploy) to MLflow and
    the shared pipeline_runs history table."""
    mlflow_setup.setup_experiment()
    with mlflow.start_run(run_name="model_pipeline"):
        mlflow.log_metric("active_auc", performance["auc_score"])
        mlflow.log_param("performance_ok", performance["performance_ok"])
        mlflow.log_param("retrained", retrain_result["retrained"])
        if deploy_result is not None:
            mlflow.log_param("deployed", deploy_result.get("deployed", False))
        run_id = mlflow.active_run().info.run_id

    status = "success"
    details = (
        f"performance_ok={performance['performance_ok']}, "
        f"retrained={retrain_result['retrained']}"
    )
    if deploy_result is not None:
        details += f", deployed={deploy_result.get('deployed', False)}"

    record_pipeline_run(
        pipeline_name="model_pipeline",
        status=status,
        rows_processed=0,
        duration_seconds=duration_seconds,
        details=details,
    )
    log.info(
        "Logged model pipeline run to MLflow (run_id=%s) and pipeline_runs", run_id
    )
    return run_id


if __name__ == "__main__":
    start = time.time()

    performance = check_model_performance()
    print("=== Performance Check ===")
    for key, value in performance.items():
        print(f"  {key}: {value}")

    retrain_result = trigger_retraining(performance)
    print(f"\n=== Retraining ===\n  retrained: {retrain_result['retrained']}")
    if not retrain_result["retrained"]:
        print(f"  reason: {retrain_result['reason']}")

    deploy_result = None
    if retrain_result["retrained"]:
        validation = validate_new_model(
            retrain_result["model"], retrain_result["trainer"]
        )
        print(
            f"\n=== Validation ===\n  auc: {validation['auc']}\n  passed: {validation['passed']}"
        )
        deploy_result = deploy_new_model(retrain_result["model"], validation)
        print(f"\n=== Deployment ===\n  {deploy_result}")

    run_id = log_pipeline_run(
        performance, retrain_result, deploy_result, duration_seconds=time.time() - start
    )
    print(f"\nMLflow run: {run_id}")
