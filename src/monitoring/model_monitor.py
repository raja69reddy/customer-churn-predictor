"""Model monitoring — Population Stability Index (PSI) drift detection for both the model's
predicted scores and its input features.

Reference-vs-current design note: this project has a single static 7,043-row dataset (no real
post-deployment data stream yet — see [[project_churn_predictor]] Day 24/25 notes on similar
proxy limitations), so "current production data" is proxied by the model's held-out test split
(X_test/y_test) and "reference/expected" by its training split (X_train/y_train) — the standard
train/serving-skew check used when a true time-based production sample isn't available yet.
Both splits come from ModelTrainer.split_data(), which uses a fixed random_state=42, so PSI
values here are deterministic and reproducible across runs.

Requires the full ML stack (xgboost/sklearn) and a trained active model — not demo-mode safe.
"""

import os

import numpy as np
import pandas as pd

from src.models.train import ModelTrainer
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("model_monitor")

OUTPUT_DIR = "data/processed"
PSI_BUCKETS = 10
DEFAULT_ALERT_THRESHOLD = 0.2


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "Stable"
    if psi < 0.2:
        return "Moderate"
    return "Significant"


def calculate_psi(expected, actual, buckets: int = PSI_BUCKETS) -> float:
    """Population Stability Index between an expected (reference) and actual (current)
    distribution. Buckets are quantile edges of `expected`, so each reference bucket holds
    ~1/buckets of the reference population by construction.

    PSI < 0.1: no significant change. 0.1-0.2: moderate change. >= 0.2: significant change.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        # Expected distribution has too little spread (e.g. a near-constant feature) to form
        # more than one real bucket — PSI is undefined/meaningless here, treat as stable.
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    epsilon = 1e-4
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def _load_trainer_and_model() -> tuple:
    engine = get_engine()
    active = pd.read_sql(
        "SELECT model_name FROM model_registry WHERE is_active = TRUE LIMIT 1", engine
    )
    if active.empty:
        raise RuntimeError("No active model found in model_registry.")
    model_name = active.iloc[0]["model_name"]

    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)
    model = trainer.load_model(model_name)
    return trainer, model, model_name


def check_prediction_drift(trainer: ModelTrainer = None, model=None) -> dict:
    """PSI between the model's predicted churn probabilities on the training split
    (expected/reference) vs the held-out test split (actual/current)."""
    if trainer is None or model is None:
        trainer, model, _ = _load_trainer_and_model()

    expected_scores = model.predict_proba(trainer.X_train)[:, 1]
    actual_scores = model.predict_proba(trainer.X_test)[:, 1]

    psi = calculate_psi(expected_scores, actual_scores)
    result = {
        "metric": "prediction_scores",
        "psi": psi,
        "status": classify_psi(psi),
        "expected_mean": round(float(expected_scores.mean()), 4),
        "actual_mean": round(float(actual_scores.mean()), 4),
    }
    log.info(
        "Prediction drift PSI=%.4f (%s) — expected_mean=%.4f actual_mean=%.4f",
        psi,
        result["status"],
        result["expected_mean"],
        result["actual_mean"],
    )
    return result


def check_feature_drift(trainer: ModelTrainer = None) -> pd.DataFrame:
    """PSI per model input feature between the training split (expected) and test split
    (actual). Returns a table sorted by PSI descending."""
    if trainer is None:
        trainer, _, _ = _load_trainer_and_model()

    rows = []
    for feature in trainer.get_feature_columns():
        psi = calculate_psi(trainer.X_train[feature], trainer.X_test[feature])
        rows.append({"feature": feature, "psi": psi, "status": classify_psi(psi)})

    result = (
        pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    )
    log.info("Computed feature drift PSI for %d features", len(result))
    return result


def generate_monitoring_report() -> dict:
    """Runs prediction drift + feature drift checks and returns a combined report dict."""
    trainer, model, model_name = _load_trainer_and_model()

    prediction_drift = check_prediction_drift(trainer, model)
    feature_drift = check_feature_drift(trainer)

    report = {
        "model_name": model_name,
        "prediction_drift": prediction_drift,
        "feature_drift": feature_drift,
        "overall_status": classify_psi(
            max(prediction_drift["psi"], feature_drift["psi"].max())
        ),
    }
    log.info(
        "Generated monitoring report for model=%s — overall_status=%s",
        model_name,
        report["overall_status"],
    )
    return report


def alert_on_drift(
    threshold: float = DEFAULT_ALERT_THRESHOLD, report: dict = None
) -> list:
    """Returns a list of alert dicts for every metric (prediction score or feature) whose
    PSI exceeds `threshold` (default 0.2, the standard "significant drift" cutoff)."""
    if report is None:
        report = generate_monitoring_report()

    alerts = []
    pred = report["prediction_drift"]
    if pred["psi"] > threshold:
        alerts.append(
            {
                "metric": "prediction_scores",
                "psi": pred["psi"],
                "message": f"Prediction score drift PSI={pred['psi']:.4f} exceeds threshold "
                f"{threshold} — model outputs on current data no longer resemble training data.",
            }
        )

    for _, row in report["feature_drift"].iterrows():
        if row["psi"] > threshold:
            alerts.append(
                {
                    "metric": row["feature"],
                    "psi": row["psi"],
                    "message": f"Feature '{row['feature']}' drift PSI={row['psi']:.4f} exceeds "
                    f"threshold {threshold} — consider investigating input data quality or "
                    "retraining.",
                }
            )

    if alerts:
        log.warning(
            "%d drift alert(s) raised at threshold=%.2f", len(alerts), threshold
        )
    else:
        log.info("No drift alerts raised at threshold=%.2f", threshold)
    return alerts


def save_monitoring_report(output_dir: str = OUTPUT_DIR) -> dict:
    """Runs the full monitoring report and saves a CSV + markdown summary to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    report = generate_monitoring_report()
    alerts = alert_on_drift(report=report)

    csv_path = os.path.join(output_dir, "model_monitoring_report.csv")
    report["feature_drift"].to_csv(csv_path, index=False)

    lines = [
        "# Model Monitoring Report",
        "",
        f"Model: {report['model_name']}",
        f"Overall status: {report['overall_status']}",
        "",
        "## Prediction Score Drift",
        f"- PSI: {report['prediction_drift']['psi']}",
        f"- Status: {report['prediction_drift']['status']}",
        f"- Expected (train) mean: {report['prediction_drift']['expected_mean']}",
        f"- Actual (test) mean: {report['prediction_drift']['actual_mean']}",
        "",
        "## Feature Drift",
        report["feature_drift"].to_string(index=False),
        "",
        "## Alerts",
    ]
    if alerts:
        for alert in alerts:
            lines.append(f"- {alert['message']}")
    else:
        lines.append("- No features or prediction scores exceeded the drift threshold.")

    report_path = os.path.join(output_dir, "model_monitoring_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved monitoring report to %s and %s", csv_path, report_path)
    return {
        "csv_path": csv_path,
        "report_path": report_path,
        "alert_count": len(alerts),
    }


if __name__ == "__main__":
    report = generate_monitoring_report()

    print("=== Prediction Drift ===")
    for key, value in report["prediction_drift"].items():
        print(f"  {key}: {value}")

    print("\n=== Feature Drift (PSI) ===")
    print(report["feature_drift"].to_string(index=False))

    print(f"\nOverall status: {report['overall_status']}")

    alerts = alert_on_drift(report=report)
    print(f"\n=== Alerts (threshold={DEFAULT_ALERT_THRESHOLD}) ===")
    if alerts:
        for alert in alerts:
            print(f"  ⚠️ {alert['message']}")
    else:
        print("  None — no drift detected above threshold.")

    result = save_monitoring_report()
    print(
        f"\nSaved monitoring report to {result['report_path']} and {result['csv_path']}"
    )
