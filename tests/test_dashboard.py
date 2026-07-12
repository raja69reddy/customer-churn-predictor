"""Unit tests for the Streamlit dashboard's data sources and importable components."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.models.predict import ChurnPredictor
from src.utils.db import get_engine

RETENTION_TARGETS_PATH = "data/processed/retention_targets.csv"


def test_vw_churn_overview_returns_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM vw_churn_overview", engine)
    assert not df.empty


def test_vw_churn_predictions_returns_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM vw_churn_predictions", engine)
    assert not df.empty


def test_vw_model_performance_returns_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM vw_model_performance", engine)
    assert not df.empty


def test_churn_predictor_loads_correctly():
    predictor = ChurnPredictor()
    model = predictor.load_best_model()
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_retention_targets_csv_exists():
    assert os.path.exists(RETENTION_TARGETS_PATH), f"{RETENTION_TARGETS_PATH} not found"
    df = pd.read_csv(RETENTION_TARGETS_PATH)
    assert not df.empty


def test_all_dashboard_components_import_correctly():
    from dashboard.components import charts, metrics

    assert callable(metrics.display_kpi_card)
    assert callable(metrics.display_kpi_row)
    assert callable(metrics.format_percentage)
    assert callable(metrics.format_number)
    assert callable(metrics.get_risk_color)

    assert callable(charts.churn_distribution_pie)
    assert callable(charts.risk_segment_bar)
    assert callable(charts.probability_histogram)
    assert callable(charts.feature_importance_bar)
    assert callable(charts.roc_curve_plot)
    assert callable(charts.shap_summary_plot)
