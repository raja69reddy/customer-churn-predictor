"""Unit tests for the Day 9 batch prediction pipeline."""

import pytest
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from sqlalchemy import text

from src.models import (
    batch_scorer,
    incremental_scorer,
    prediction_monitor,
    retention_targets,
)
from src.utils.db import get_engine


def test_batch_scorer_runs_without_errors():
    batch_scorer.run(mode="full", threshold=None, output="db")


@pytest.fixture(scope="module")
def predictions_df():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM churn_predictions", engine)


def test_all_customers_have_predictions(predictions_df):
    engine = get_engine()
    raw_count = pd.read_sql("SELECT COUNT(*) n FROM raw_customers", engine).iloc[0]["n"]
    assert len(predictions_df) == raw_count


def test_scores_between_0_and_1(predictions_df):
    assert (predictions_df["churn_probability"] >= 0).all()
    assert (predictions_df["churn_probability"] <= 1).all()


def test_risk_segments_are_valid_values(predictions_df):
    assert set(predictions_df["risk_segment"].unique()).issubset(
        {"High", "Medium", "Low"}
    )


def test_incremental_scorer_only_updates_stale_records():
    engine = get_engine()

    sample_ids = pd.read_sql(
        "SELECT customer_id FROM churn_predictions ORDER BY customer_id LIMIT 10",
        engine,
    )["customer_id"].tolist()

    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE churn_predictions SET predicted_at = NOW() - INTERVAL '10 days' "
                "WHERE customer_id = ANY(:ids)"
            ),
            {"ids": sample_ids},
        )

    before = pd.read_sql(
        "SELECT customer_id, predicted_at FROM churn_predictions", engine
    )

    scored = incremental_scorer.run(stale_days=7)

    after = pd.read_sql(
        "SELECT customer_id, predicted_at FROM churn_predictions", engine
    )

    assert set(scored["customer_id"]) == set(sample_ids)

    merged = before.merge(after, on="customer_id", suffixes=("_before", "_after"))
    changed_ids = set(
        merged.loc[
            merged["predicted_at_before"] != merged["predicted_at_after"], "customer_id"
        ]
    )
    assert changed_ids == set(sample_ids)


def test_retention_targets_creates_valid_tiers():
    targets = retention_targets.run()
    assert set(targets["priority_tier"].unique()).issubset(
        {"Tier 1", "Tier 2", "Tier 3"}
    )
    assert (targets["risk_segment"] == "High").all()


def test_prediction_monitor_passes_all_checks():
    results = prediction_monitor.monitor_predictions()
    assert results["score_distribution"]["passed"]
    assert results["segment_distribution"]["passed"]
