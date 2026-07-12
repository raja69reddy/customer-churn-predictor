"""Monitors churn_predictions quality — score ranges, segment sanity, and score drift over time."""

import json
import os
from datetime import datetime

import pandas as pd

from src.utils.db import get_engine

SNAPSHOT_PATH = "data/processed/prediction_monitor_snapshot.json"
REPORT_PATH = "data/processed/prediction_monitoring_report.txt"

VALID_SEGMENTS = {"High", "Medium", "Low"}
MIN_SEGMENT_PCT = 1.0
MAX_SEGMENT_PCT = 95.0


def _load_predictions() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM churn_predictions", engine)


def check_score_distribution(df: pd.DataFrame = None) -> dict:
    df = df if df is not None else _load_predictions()
    scores = df["churn_probability"]

    passed = bool(
        (scores >= 0).all() and (scores <= 1).all() and not scores.isnull().any()
    )
    return {
        "passed": passed,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "n": int(len(scores)),
    }


def check_segment_distribution(df: pd.DataFrame = None) -> dict:
    df = df if df is not None else _load_predictions()
    segments = df["risk_segment"]

    valid_values = set(segments.dropna().unique()).issubset(VALID_SEGMENTS)
    percentages = (segments.value_counts(normalize=True) * 100).to_dict()

    realistic_spread = len(percentages) == len(VALID_SEGMENTS) and all(
        MIN_SEGMENT_PCT <= pct <= MAX_SEGMENT_PCT for pct in percentages.values()
    )

    return {
        "passed": valid_values and realistic_spread,
        "valid_values": valid_values,
        "realistic_spread": realistic_spread,
        "percentages": percentages,
    }


def detect_score_drift(df: pd.DataFrame = None) -> dict:
    df = df if df is not None else _load_predictions()
    current_mean = float(df["churn_probability"].mean())

    previous_mean = None
    has_previous = False
    if os.path.exists(SNAPSHOT_PATH):
        with open(SNAPSHOT_PATH) as f:
            snapshot = json.load(f)
        previous_mean = snapshot.get("mean_score")
        has_previous = previous_mean is not None

    drift = abs(current_mean - previous_mean) if has_previous else 0.0

    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(
            {"mean_score": current_mean, "snapshot_at": datetime.now().isoformat()}, f
        )

    return {
        "has_previous_snapshot": has_previous,
        "previous_mean": previous_mean,
        "current_mean": current_mean,
        "drift": drift,
    }


def alert_on_drift(drift_result: dict = None, threshold: float = 0.1) -> bool:
    drift_result = drift_result if drift_result is not None else detect_score_drift()
    drift = drift_result["drift"]

    if drift_result["has_previous_snapshot"] and drift > threshold:
        print(f"ALERT: score drift {drift:.4f} exceeds threshold {threshold}")
        return True

    print(f"No significant drift detected (drift={drift:.4f}, threshold={threshold})")
    return False


def monitor_predictions() -> dict:
    df = _load_predictions()

    score_check = check_score_distribution(df)
    segment_check = check_segment_distribution(df)
    drift_result = detect_score_drift(df)

    return {
        "score_distribution": score_check,
        "segment_distribution": segment_check,
        "drift": drift_result,
        "all_checks_passed": score_check["passed"] and segment_check["passed"],
    }


def generate_monitoring_report(results: dict = None) -> str:
    results = results if results is not None else monitor_predictions()

    lines = [
        "=" * 55,
        "  Prediction Monitoring Report",
        f"  Generated: {datetime.now().isoformat()}",
        "=" * 55,
    ]

    sd = results["score_distribution"]
    lines.append(f"\n[1] Score Distribution - {'PASS' if sd['passed'] else 'FAIL'}")
    lines.append(
        f"    n={sd['n']}  min={sd['min']:.4f}  max={sd['max']:.4f}  "
        f"mean={sd['mean']:.4f}  std={sd['std']:.4f}"
    )

    sg = results["segment_distribution"]
    lines.append(f"\n[2] Segment Distribution - {'PASS' if sg['passed'] else 'FAIL'}")
    for segment, pct in sg["percentages"].items():
        lines.append(f"    {segment}: {pct:.2f}%")

    drift = results["drift"]
    lines.append("\n[3] Score Drift")
    if drift["has_previous_snapshot"]:
        lines.append(
            f"    previous mean: {drift['previous_mean']:.4f}  "
            f"current mean: {drift['current_mean']:.4f}  drift: {drift['drift']:.4f}"
        )
    else:
        lines.append("    No previous snapshot - this run establishes the baseline.")

    overall_status = (
        "ALL CHECKS PASSED" if results["all_checks_passed"] else "SOME CHECKS FAILED"
    )
    lines.append(f"\nOverall: {overall_status}")
    lines.append("=" * 55)

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    return report


if __name__ == "__main__":
    results = monitor_predictions()
    print(generate_monitoring_report(results))
    alert_on_drift(results["drift"])
