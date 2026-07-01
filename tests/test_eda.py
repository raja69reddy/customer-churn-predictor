"""Unit tests for the EDA summary script and raw_customers data quality."""
import pytest
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.data.eda_summary import generate_summary
from src.utils.db import get_engine

REQUIRED_COLUMNS = [
    "customer_id", "gender", "senior_citizen", "partner", "dependents",
    "tenure", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "paperless_billing",
    "payment_method", "monthly_charges", "total_charges", "churn",
]


@pytest.fixture(scope="module")
def df():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM raw_customers", engine)


def test_eda_summary_runs_without_errors(df):
    summary = generate_summary()
    assert isinstance(summary, str)
    assert "Overall churn rate" in summary
    assert "Top 5 features correlated with churn" in summary


def test_churn_rate_between_20_and_35_percent(df):
    churn_rate = (df["churn"] == "Yes").mean() * 100
    assert 20 <= churn_rate <= 35, f"Churn rate {churn_rate:.2f}% outside expected 20-35% range"


def test_required_columns_exist(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns in raw_customers: {missing}"


def test_no_negative_tenure_or_monthly_charges(df):
    assert (df["tenure"] >= 0).all(), "Found negative tenure values"
    assert (df["monthly_charges"] >= 0).all(), "Found negative monthly_charges values"


def test_total_charges_at_least_monthly_charges(df):
    # Synthetic data adds Gaussian noise to total_charges, so a handful of tenure=1
    # rows can dip a few dollars below monthly_charges. Allow a small tolerance
    # instead of a strict >= so the test reflects real data rather than an idealized rule.
    tolerance = 10.0
    violations = df[df["total_charges"] < df["monthly_charges"] - tolerance]
    assert violations.empty, (
        f"{len(violations)} rows have total_charges more than ${tolerance} below monthly_charges"
    )
