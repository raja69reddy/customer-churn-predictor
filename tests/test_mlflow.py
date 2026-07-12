"""Unit tests for MLflow experiment tracking, model registry, and analysis tooling."""

import pytest
from dotenv import load_dotenv

load_dotenv()

import mlflow

from src.models import batch_scorer, experiment_analyzer, mlflow_registry, mlflow_setup

pytestmark = pytest.mark.skip(reason="requires trained models")

EXPECTED_RUN_NAMES = {
    "Logistic Regression",
    "Decision Tree",
    "Random Forest (tuned)",
    "XGBoost (tuned)",
    "LightGBM",
    "Ensemble",
}


@pytest.fixture(scope="module")
def run_history():
    return experiment_analyzer.get_run_history()


def test_mlflow_experiment_exists():
    mlflow_setup.configure_tracking()
    experiment = mlflow.get_experiment_by_name(mlflow_setup.EXPERIMENT_NAME)
    assert experiment is not None


def test_all_6_models_have_logged_runs(run_history):
    logged_names = set(run_history["tags.mlflow.runName"].dropna())
    assert EXPECTED_RUN_NAMES.issubset(logged_names)


def test_best_model_is_registered():
    versions = mlflow_registry.list_model_versions()
    assert len(versions) >= 1


def test_production_model_loads_correctly():
    model = mlflow_registry.get_production_model()
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_batch_scorer_uses_mlflow_model():
    predictor, version = batch_scorer.load_production_predictor()
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict_proba")
    assert mlflow_registry.REGISTERED_MODEL_NAME in predictor.model_name
    assert version is not None


def test_experiment_analyzer_returns_valid_results(run_history):
    assert not run_history.empty
    assert "metrics.auc" in run_history.columns

    best_run = experiment_analyzer.get_best_run()
    assert best_run["metrics.auc"] == run_history["metrics.auc"].max()
