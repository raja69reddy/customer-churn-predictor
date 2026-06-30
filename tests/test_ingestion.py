"""Unit tests for the ingestion pipeline and raw CSV data quality."""
import os
import pytest
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "customers.csv")

REQUIRED_COLUMNS = [
    "customer_id", "gender", "senior_citizen", "partner", "dependents",
    "tenure", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "paperless_billing",
    "payment_method", "monthly_charges", "total_charges", "churn",
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV_PATH)


def test_csv_file_exists():
    assert os.path.exists(CSV_PATH), f"CSV not found at {CSV_PATH}"


def test_required_columns_exist(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_customer_id_is_unique(df):
    dupes = df["customer_id"].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate customer_id values"


def test_churn_only_yes_no(df):
    invalid = df[~df["churn"].isin(["Yes", "No"])]["churn"].unique()
    assert len(invalid) == 0, f"Invalid churn values found: {invalid}"


def test_monthly_charges_positive(df):
    bad = (df["monthly_charges"] <= 0).sum()
    assert bad == 0, f"{bad} rows have non-positive monthly_charges"


def test_total_charges_positive(df):
    bad = (df["total_charges"] <= 0).sum()
    assert bad == 0, f"{bad} rows have non-positive total_charges"


def test_tenure_within_range(df):
    bad = (~df["tenure"].between(0, 72)).sum()
    assert bad == 0, f"{bad} rows have tenure outside 0-72"
