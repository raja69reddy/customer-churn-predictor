"""Unit tests for FeatureEngineer (src/features/engineering.py)."""

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "tenure": [5, 15, 30, 60, 1, 72],
            "monthly_charges": [50.0, 60.0, 70.0, 80.0, 20.0, 100.0],
            "total_charges": [250.0, 900.0, 2100.0, 4800.0, 20.0, 7200.0],
            "contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "payment_method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            "phone_service": ["Yes", "Yes", "No", "Yes", "No", "Yes"],
            "internet_service": [
                "DSL",
                "Fiber optic",
                "No",
                "DSL",
                "No",
                "Fiber optic",
            ],
            "online_security": [
                "Yes",
                "No",
                "No internet service",
                "Yes",
                "No internet service",
                "No",
            ],
            "online_backup": [
                "No",
                "No",
                "No internet service",
                "Yes",
                "No internet service",
                "Yes",
            ],
            "device_protection": [
                "No",
                "No",
                "No internet service",
                "Yes",
                "No internet service",
                "Yes",
            ],
            "tech_support": [
                "No",
                "No",
                "No internet service",
                "Yes",
                "No internet service",
                "No",
            ],
            "streaming_tv": [
                "No",
                "Yes",
                "No internet service",
                "Yes",
                "No internet service",
                "Yes",
            ],
            "streaming_movies": [
                "No",
                "Yes",
                "No internet service",
                "Yes",
                "No internet service",
                "Yes",
            ],
            "churn": ["Yes", "No", "No", "Yes", "No", "No"],
        }
    )


@pytest.fixture(scope="module")
def raw_df():
    return FeatureEngineer().load_raw_data()


def test_tenure_group_creates_correct_buckets(sample_df):
    fe = FeatureEngineer()
    df = fe.create_tenure_group(sample_df)
    expected = {
        "C1": "New Customer",  # tenure=5  -> 0-12
        "C2": "Growing Customer",  # tenure=15 -> 12-24
        "C3": "Established Customer",  # tenure=30 -> 24-48
        "C4": "Loyal Customer",  # tenure=60 -> 48+
        "C5": "New Customer",  # tenure=1  -> 0-12
        "C6": "Loyal Customer",  # tenure=72 -> 48+
    }
    actual = dict(zip(df["customer_id"], df["tenure_group"].astype(str)))
    assert actual == expected


def test_contract_risk_score_values(sample_df):
    fe = FeatureEngineer()
    df = fe.create_contract_risk_score(sample_df)
    assert set(df["contract_risk_score"].unique()).issubset({1, 2, 3})
    expected = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    for contract, score in zip(df["contract"], df["contract_risk_score"]):
        assert score == expected[contract]


def test_payment_risk_score_values(sample_df):
    fe = FeatureEngineer()
    df = fe.create_payment_risk_score(sample_df)
    assert set(df["payment_risk_score"].unique()).issubset({1, 2, 3})
    expected = {
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 1,
    }
    for method, score in zip(df["payment_method"], df["payment_risk_score"]):
        assert score == expected[method]


def test_churn_label_is_binary(sample_df):
    fe = FeatureEngineer()
    df = fe.create_churn_label(sample_df)
    assert set(df["churn_label"].unique()).issubset({0, 1})
    assert df["churn_label"].tolist() == [1, 0, 0, 1, 0, 0]


def test_services_count_non_negative(raw_df):
    fe = FeatureEngineer()
    df = fe.clean_data(raw_df)
    df = fe.create_services_count(df)
    assert (df["services_count"] >= 0).all()


def test_smote_balances_classes(raw_df):
    fe = FeatureEngineer()
    df = fe.clean_data(raw_df)
    df = fe.create_churn_label(df)

    X = df[["tenure", "monthly_charges", "total_charges"]]
    y = df["churn_label"]

    X_resampled, y_resampled = fe.apply_smote(X, y)
    counts = pd.Series(y_resampled).value_counts()

    assert counts[0] == counts[1]
    assert len(X_resampled) == len(y_resampled)


def test_preprocessor_saves_and_loads_correctly(raw_df, tmp_path):
    fe = FeatureEngineer()
    df = fe.clean_data(raw_df)

    transformed = fe.fit_transform(df)
    save_path = str(tmp_path / "preprocessor.pkl")
    fe.save_pipeline(save_path)

    fe2 = FeatureEngineer()
    loaded_pipeline = fe2.load_pipeline(save_path)
    reloaded_transformed = loaded_pipeline.transform(df)

    assert np.allclose(transformed, reloaded_transformed)
