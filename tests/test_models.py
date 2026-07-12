"""Unit tests for trained baseline models and the model_registry table."""
import pytest
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.utils.db import get_engine

pytestmark = pytest.mark.skip(reason="requires trained models")

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]


@pytest.fixture(scope="module")
def trainer():
    t = ModelTrainer()
    df = t.load_processed_data()
    t.split_data(df)
    return t


@pytest.fixture(scope="module")
def models(trainer):
    return {name: trainer.load_model(name) for name in MODEL_NAMES}


@pytest.fixture(scope="module")
def registry():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM model_registry", engine)


def test_all_models_load_correctly(models):
    for name in MODEL_NAMES:
        assert models[name] is not None
        assert hasattr(models[name], "predict")
        assert hasattr(models[name], "predict_proba")


def test_model_predictions_are_valid_probabilities(trainer, models):
    for name, model in models.items():
        y_proba = model.predict_proba(trainer.X_test)[:, 1]
        assert (y_proba >= 0).all() and (y_proba <= 1).all(), f"{name} produced probabilities outside [0, 1]"


def test_model_accuracy_above_70_percent(trainer, models):
    evaluator = ModelEvaluator()
    for name, model in models.items():
        metrics = evaluator.evaluate_model(model, trainer.X_test, trainer.y_test)
        assert metrics["accuracy"] > 0.70, f"{name} accuracy {metrics['accuracy']:.4f} is not above 70%"


def test_model_auc_above_0_75(trainer, models):
    evaluator = ModelEvaluator()
    for name, model in models.items():
        metrics = evaluator.evaluate_model(model, trainer.X_test, trainer.y_test)
        assert metrics["auc"] > 0.75, f"{name} AUC {metrics['auc']:.4f} is not above 0.75"


def test_model_registry_has_4_entries(registry):
    # Day 6 tuning adds extra rows (e.g. random_forest_tuned, xgboost_tuned) on top of the
    # 4 Day 5 baseline models, so the registry grows over time — check the baseline floor instead
    # of an exact count.
    assert len(registry) >= 4, f"Expected at least 4 model_registry entries, found {len(registry)}"
    assert set(MODEL_NAMES).issubset(set(registry["model_name"])), "Missing one or more baseline models in model_registry"


def test_best_model_has_is_active_true(registry):
    best = registry.loc[registry["auc_score"].idxmax()]
    assert bool(best["is_active"]) is True

    active_rows = registry[registry["is_active"]]
    assert len(active_rows) == 1
    assert active_rows.iloc[0]["model_name"] == best["model_name"]
