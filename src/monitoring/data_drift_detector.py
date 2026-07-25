"""Data drift detection for the raw business features (tenure, monthly charges, churn rate) —
a business-facing companion to model_monitor.py's model-level PSI checks.

Uses the same reference-vs-current design as model_monitor.py: "current production data" is
proxied by the model's held-out test split, "reference/expected" by its training split, both
derived from ModelTrainer.split_data()'s fixed random_state=42 (see model_monitor.py's module
docstring for why — this project has one static dataset, no real post-deployment data stream).

Requires the full ML stack (sklearn, via ModelTrainer) — not demo-mode safe.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.train import ModelTrainer  # noqa: E402
from src.monitoring.model_monitor import calculate_psi, classify_psi  # noqa: E402
from src.utils.db import get_engine  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

log = setup_logging("data_drift_detector")

OUTPUT_DIR = "data/processed"
CHURN_RATE_ALERT_THRESHOLD_PCT = 5.0


def _load_train_test_raw() -> pd.DataFrame:
    """Returns raw_customers rows (tenure, monthly_charges, churn) tagged with which split
    ('train'/'test') each customer fell into, using the same fixed split as model_monitor.py.
    """
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    train_ids = set(trainer.df.loc[trainer.X_train.index, "customer_id"])
    test_ids = set(trainer.df.loc[trainer.X_test.index, "customer_id"])

    engine = get_engine()
    raw = pd.read_sql(
        "SELECT customer_id, tenure, monthly_charges, churn FROM raw_customers", engine
    )
    raw["split"] = raw["customer_id"].apply(
        lambda cid: (
            "train" if cid in train_ids else ("test" if cid in test_ids else None)
        )
    )
    raw = raw[raw["split"].notna()]
    return raw


def detect_tenure_drift(df: pd.DataFrame = None) -> dict:
    """PSI between train (expected) and test (actual) tenure distributions."""
    if df is None:
        df = _load_train_test_raw()

    train_tenure = df.loc[df["split"] == "train", "tenure"]
    test_tenure = df.loc[df["split"] == "test", "tenure"]

    psi = calculate_psi(train_tenure, test_tenure)
    result = {
        "feature": "tenure",
        "psi": psi,
        "status": classify_psi(psi),
        "train_mean": round(float(train_tenure.mean()), 2),
        "test_mean": round(float(test_tenure.mean()), 2),
    }
    log.info(
        "Tenure drift PSI=%.4f (%s) — train_mean=%.2f test_mean=%.2f",
        psi,
        result["status"],
        result["train_mean"],
        result["test_mean"],
    )
    return result


def detect_charges_drift(df: pd.DataFrame = None) -> dict:
    """PSI between train (expected) and test (actual) monthly_charges distributions."""
    if df is None:
        df = _load_train_test_raw()

    train_charges = df.loc[df["split"] == "train", "monthly_charges"]
    test_charges = df.loc[df["split"] == "test", "monthly_charges"]

    psi = calculate_psi(train_charges, test_charges)
    result = {
        "feature": "monthly_charges",
        "psi": psi,
        "status": classify_psi(psi),
        "train_mean": round(float(train_charges.mean()), 2),
        "test_mean": round(float(test_charges.mean()), 2),
    }
    log.info(
        "Charges drift PSI=%.4f (%s) — train_mean=%.2f test_mean=%.2f",
        psi,
        result["status"],
        result["train_mean"],
        result["test_mean"],
    )
    return result


def detect_churn_rate_drift(df: pd.DataFrame = None) -> dict:
    """Compares the overall churn rate between train (expected) and test (actual) splits.
    Uses the 2-category PSI formula directly (not calculate_psi's quantile bucketing, which
    assumes a continuous distribution) since churn rate is a single binary proportion.
    """
    if df is None:
        df = _load_train_test_raw()

    train_churn = (df.loc[df["split"] == "train", "churn"] == "Yes").mean()
    test_churn = (df.loc[df["split"] == "test", "churn"] == "Yes").mean()

    epsilon = 1e-4
    expected_pct = [max(1 - train_churn, epsilon), max(train_churn, epsilon)]
    actual_pct = [max(1 - test_churn, epsilon), max(test_churn, epsilon)]
    psi = sum((a - e) * math.log(a / e) for e, a in zip(expected_pct, actual_pct))
    psi = round(psi, 4)

    delta_pct_points = round((test_churn - train_churn) * 100, 2)
    result = {
        "feature": "churn_rate",
        "psi": psi,
        "status": classify_psi(psi),
        "train_churn_rate_pct": round(train_churn * 100, 2),
        "test_churn_rate_pct": round(test_churn * 100, 2),
        "delta_pct_points": delta_pct_points,
        "drifted": abs(delta_pct_points) > CHURN_RATE_ALERT_THRESHOLD_PCT,
    }
    log.info(
        "Churn rate drift: train=%.2f%% test=%.2f%% (delta=%.2f pts, psi=%.4f)",
        result["train_churn_rate_pct"],
        result["test_churn_rate_pct"],
        delta_pct_points,
        psi,
    )
    return result


def generate_drift_report() -> str:
    """Builds a formatted markdown drift report covering tenure, charges, and churn rate."""
    df = _load_train_test_raw()
    tenure = detect_tenure_drift(df)
    charges = detect_charges_drift(df)
    churn_rate = detect_churn_rate_drift(df)

    lines = [
        "# Data Drift Report",
        "",
        "## Tenure",
        f"- PSI: {tenure['psi']} ({tenure['status']})",
        f"- Train mean: {tenure['train_mean']} months | Test mean: {tenure['test_mean']} months",
        "",
        "## Monthly Charges",
        f"- PSI: {charges['psi']} ({charges['status']})",
        f"- Train mean: ${charges['train_mean']} | Test mean: ${charges['test_mean']}",
        "",
        "## Churn Rate",
        f"- Train: {churn_rate['train_churn_rate_pct']}% | "
        f"Test: {churn_rate['test_churn_rate_pct']}% "
        f"(delta {churn_rate['delta_pct_points']:+.2f} pts)",
        f"- PSI: {churn_rate['psi']} ({churn_rate['status']})",
        f"- Drifted (> {CHURN_RATE_ALERT_THRESHOLD_PCT} pt threshold): {churn_rate['drifted']}",
    ]
    report = "\n".join(lines)
    log.info("Generated data drift report")
    return report


def visualize_drift(
    feature: str, df: pd.DataFrame = None, output_dir: str = OUTPUT_DIR
) -> str:
    """Saves an overlaid train-vs-test histogram for `feature` ('tenure' or 'monthly_charges')
    as a PNG in output_dir. Returns the saved path."""
    if feature not in ("tenure", "monthly_charges"):
        raise ValueError("feature must be 'tenure' or 'monthly_charges'")
    if df is None:
        df = _load_train_test_raw()

    os.makedirs(output_dir, exist_ok=True)
    train_values = df.loc[df["split"] == "train", feature]
    test_values = df.loc[df["split"] == "test", feature]

    plt.figure(figsize=(8, 5))
    plt.hist(
        train_values, bins=30, alpha=0.5, label="Train (expected)", color="steelblue"
    )
    plt.hist(
        test_values, bins=30, alpha=0.5, label="Test (current)", color="darkorange"
    )
    plt.title(f"Drift check: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, f"drift_{feature}.png")
    plt.savefig(path)
    plt.close()

    log.info("Saved drift visualization for %s to %s", feature, path)
    return path


if __name__ == "__main__":
    df = _load_train_test_raw()

    print("=== Tenure Drift ===")
    print(detect_tenure_drift(df))

    print("\n=== Charges Drift ===")
    print(detect_charges_drift(df))

    print("\n=== Churn Rate Drift ===")
    print(detect_churn_rate_drift(df))

    print("\n=== Full Report ===")
    print(generate_drift_report())

    tenure_png = visualize_drift("tenure", df)
    charges_png = visualize_drift("monthly_charges", df)
    print(f"\nSaved drift plots to {tenure_png} and {charges_png}")
