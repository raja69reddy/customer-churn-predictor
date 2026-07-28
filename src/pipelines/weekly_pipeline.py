"""Weekly churn pipeline — week-over-week risk trend, per-customer risk-segment transitions,
and a saved digest report.

Data-source note: real prediction history (churn_predictions + churn_predictions_history) spans
about a month, enough for genuine week-over-week windows (this week = last 7 days, last week =
the 7 days before that) — unlike src/analysis/trend_analyzer.py's 6-month trend, which needed a
different (tenure-based) proxy because a month isn't enough for that. Since scoring is
deterministic for a fixed model (see Day 26 dashboard notes), most customers' risk_segment will
be unchanged week-over-week UNLESS the active model itself changed between the two windows
(model_registry has had multiple different active models across this project's build history) —
zero transitions found is therefore a plausible, correct result, not a bug.

Requires the full ML stack indirectly (via src/pipelines/daily_pipeline.py's siblings) but this
module itself is pure pandas/SQL.
"""

import os
from datetime import datetime

import mlflow
import pandas as pd

from src.models import mlflow_setup
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging
from src.utils.pipeline_tracking import record_pipeline_run

log = setup_logging("weekly_pipeline")

OUTPUT_DIR = "data/processed"


def _load_window_snapshot(days_back_start: int, days_back_end: int) -> pd.DataFrame:
    """Each customer's latest prediction within (now - days_back_start) to (now - days_back_end)."""
    engine = get_engine()
    query = f"""
        WITH combined AS (
            SELECT customer_id, churn_probability, risk_segment, predicted_at
            FROM churn_predictions
            UNION ALL
            SELECT customer_id, churn_probability, risk_segment, predicted_at
            FROM churn_predictions_history
        )
        SELECT DISTINCT ON (customer_id) customer_id, churn_probability, risk_segment, predicted_at
        FROM combined
        WHERE predicted_at >= NOW() - INTERVAL '{days_back_start} days'
          AND predicted_at < NOW() - INTERVAL '{days_back_end} days'
        ORDER BY customer_id, predicted_at DESC
    """
    return pd.read_sql(query, engine)


def _load_weekly_transitions() -> pd.DataFrame:
    """Joins this week's and last week's per-customer snapshots into one comparison table."""
    this_week = _load_window_snapshot(7, 0)
    last_week = _load_window_snapshot(14, 7)

    merged = this_week.merge(
        last_week, on="customer_id", suffixes=("_this_week", "_last_week")
    )
    log.info(
        "Loaded weekly transitions: %d customers in both windows (this_week=%d, last_week=%d)",
        len(merged),
        len(this_week),
        len(last_week),
    )
    return merged


def compare_week_over_week() -> dict:
    """This week vs last week: avg churn probability and High-risk customer count."""
    this_week = _load_window_snapshot(7, 0)
    last_week = _load_window_snapshot(14, 7)

    result = {
        "this_week_customers": len(this_week),
        "last_week_customers": len(last_week),
        "this_week_avg_probability": (
            round(float(this_week["churn_probability"].mean()), 4)
            if not this_week.empty
            else None
        ),
        "last_week_avg_probability": (
            round(float(last_week["churn_probability"].mean()), 4)
            if not last_week.empty
            else None
        ),
        "this_week_high_risk": int((this_week["risk_segment"] == "High").sum()),
        "last_week_high_risk": int((last_week["risk_segment"] == "High").sum()),
    }
    if (
        result["this_week_avg_probability"] is not None
        and result["last_week_avg_probability"] is not None
    ):
        result["avg_probability_delta"] = round(
            result["this_week_avg_probability"] - result["last_week_avg_probability"], 4
        )
    else:
        result["avg_probability_delta"] = None

    log.info(
        "WoW: this_week_avg=%s last_week_avg=%s delta=%s high_risk %d -> %d",
        result["this_week_avg_probability"],
        result["last_week_avg_probability"],
        result["avg_probability_delta"],
        result["last_week_high_risk"],
        result["this_week_high_risk"],
    )
    return result


def identify_new_high_risk(transitions: pd.DataFrame = None) -> pd.DataFrame:
    """Customers whose risk_segment moved TO High this week from something else last week."""
    if transitions is None:
        transitions = _load_weekly_transitions()

    new_high_risk = transitions[
        (transitions["risk_segment_this_week"] == "High")
        & (transitions["risk_segment_last_week"] != "High")
    ][
        [
            "customer_id",
            "risk_segment_last_week",
            "risk_segment_this_week",
            "churn_probability_last_week",
            "churn_probability_this_week",
        ]
    ].reset_index(
        drop=True
    )

    log.info("Identified %d customer(s) newly moved to High risk", len(new_high_risk))
    return new_high_risk


def identify_recovered(transitions: pd.DataFrame = None) -> pd.DataFrame:
    """Customers whose risk_segment moved TO Low this week from something else last week."""
    if transitions is None:
        transitions = _load_weekly_transitions()

    recovered = transitions[
        (transitions["risk_segment_this_week"] == "Low")
        & (transitions["risk_segment_last_week"] != "Low")
    ][
        [
            "customer_id",
            "risk_segment_last_week",
            "risk_segment_this_week",
            "churn_probability_last_week",
            "churn_probability_this_week",
        ]
    ].reset_index(
        drop=True
    )

    log.info("Identified %d customer(s) recovered to Low risk", len(recovered))
    return recovered


def run_weekly_summary() -> dict:
    """Runs the full weekly analysis: WoW comparison + new-high-risk + recovered lists."""
    wow = compare_week_over_week()
    transitions = _load_weekly_transitions()
    new_high_risk = identify_new_high_risk(transitions)
    recovered = identify_recovered(transitions)

    log.info(
        "Weekly summary: %d new high risk, %d recovered",
        len(new_high_risk),
        len(recovered),
    )
    return {
        "week_over_week": wow,
        "new_high_risk": new_high_risk,
        "recovered": recovered,
    }


def send_weekly_digest(summary: dict = None, output_dir: str = OUTPUT_DIR) -> str:
    """Saves the weekly summary as a markdown digest to output_dir. Returns the saved path."""
    if summary is None:
        summary = run_weekly_summary()

    os.makedirs(output_dir, exist_ok=True)
    wow = summary["week_over_week"]

    lines = [
        f"# Weekly Churn Digest — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Week-over-Week",
        f"- Customers scored: {wow['last_week_customers']} -> {wow['this_week_customers']}",
        f"- Avg churn probability: {wow['last_week_avg_probability']} -> "
        f"{wow['this_week_avg_probability']} (delta {wow['avg_probability_delta']})",
        f"- High risk customers: {wow['last_week_high_risk']} -> {wow['this_week_high_risk']}",
        "",
        f"## New High Risk ({len(summary['new_high_risk'])})",
        (
            summary["new_high_risk"].to_string(index=False)
            if not summary["new_high_risk"].empty
            else "None."
        ),
        "",
        f"## Recovered to Low Risk ({len(summary['recovered'])})",
        (
            summary["recovered"].to_string(index=False)
            if not summary["recovered"].empty
            else "None."
        ),
    ]

    path = os.path.join(output_dir, "weekly_digest.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved weekly digest to %s", path)
    return path


def log_weekly_run(summary: dict = None, duration_seconds: float = 0.0) -> str:
    """Logs the weekly run to MLflow and to the shared pipeline_runs history table."""
    if summary is None:
        summary = run_weekly_summary()

    wow = summary["week_over_week"]
    mlflow_setup.setup_experiment()
    with mlflow.start_run(run_name="weekly_pipeline"):
        mlflow.log_metrics(
            {
                "this_week_avg_probability": wow["this_week_avg_probability"] or 0.0,
                "this_week_high_risk": wow["this_week_high_risk"],
                "new_high_risk_count": len(summary["new_high_risk"]),
                "recovered_count": len(summary["recovered"]),
            }
        )
        run_id = mlflow.active_run().info.run_id

    record_pipeline_run(
        pipeline_name="weekly_pipeline",
        status="success",
        rows_processed=wow["this_week_customers"],
        duration_seconds=duration_seconds,
        details=(
            f"new_high_risk={len(summary['new_high_risk'])}, "
            f"recovered={len(summary['recovered'])}"
        ),
    )
    log.info("Logged weekly run to MLflow (run_id=%s) and pipeline_runs", run_id)
    return run_id


if __name__ == "__main__":
    import time

    start = time.time()
    summary = run_weekly_summary()

    print("=== Week-over-Week ===")
    for key, value in summary["week_over_week"].items():
        print(f"  {key}: {value}")

    print(f"\n=== New High Risk ({len(summary['new_high_risk'])}) ===")
    print(summary["new_high_risk"].to_string(index=False))

    print(f"\n=== Recovered ({len(summary['recovered'])}) ===")
    print(summary["recovered"].to_string(index=False))

    digest_path = send_weekly_digest(summary)
    print(f"\nSaved digest to {digest_path}")

    run_id = log_weekly_run(summary, duration_seconds=time.time() - start)
    print(f"MLflow run: {run_id}")
