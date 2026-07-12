"""Unit tests for cross validation, SHAP explanations, tuning, and ChurnPredictor."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from sklearn.linear_model import LogisticRegression

from src.models.evaluate import ModelEvaluator
from src.models.predict import ChurnPredictor
from src.models.train import ModelTrainer

SAMPLE_CUSTOMER = {
    "tenure": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "phone_service": "Yes",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
}


@pytest.fixture(scope="module")
def trainer():
    t = ModelTrainer()
    df = t.load_processed_data()
    t.split_data(df)
    return t


@pytest.fixture(scope="module")
def evaluator():
    return ModelEvaluator()


@pytest.fixture(scope="module")
def predictor():
    p = ChurnPredictor()
    p.load_best_model()
    return p


def test_cross_validation_returns_valid_scores(trainer, evaluator):
    model = LogisticRegression(max_iter=1000)
    cv_results = evaluator.cross_validate_model(
        model, trainer.X_train, trainer.y_train, cv=5
    )

    for metric in ["accuracy", "auc", "f1"]:
        assert metric in cv_results
        assert len(cv_results[metric]) == 5
        assert ((cv_results[metric] >= 0) & (cv_results[metric] <= 1)).all()


def test_shap_values_have_correct_shape(trainer, evaluator):
    model = trainer.load_model("xgboost_tuned")
    shap_values = evaluator.explain_model(model, trainer.X_test)

    assert shap_values.values.shape == trainer.X_test.shape


def test_tuned_model_has_higher_auc_than_baseline(trainer, evaluator):
    baseline = trainer.load_model("random_forest")
    tuned = trainer.load_model("random_forest_tuned")

    baseline_auc = evaluator.evaluate_model(baseline, trainer.X_test, trainer.y_test)[
        "auc"
    ]
    tuned_auc = evaluator.evaluate_model(tuned, trainer.X_test, trainer.y_test)["auc"]

    assert tuned_auc > baseline_auc


def test_predict_single_returns_valid_probability(predictor):
    result = predictor.predict_single(SAMPLE_CUSTOMER)
    assert 0 <= result["churn_probability"] <= 1


def test_get_risk_segment_returns_valid_segment(predictor):
    assert predictor.get_risk_segment(0.9) == "High"
    assert predictor.get_risk_segment(0.7) == "High"
    assert predictor.get_risk_segment(0.5) == "Medium"
    assert predictor.get_risk_segment(0.4) == "Medium"
    assert predictor.get_risk_segment(0.1) == "Low"
    assert predictor.get_risk_segment(0.39) == "Low"


def test_optimal_threshold_between_0_3_and_0_7():
    threshold_path = "models/optimal_threshold.txt"
    assert os.path.exists(
        threshold_path
    ), "models/optimal_threshold.txt not found — run find_optimal_threshold first"

    with open(threshold_path) as f:
        threshold = float(f.read().strip())

    assert 0.3 <= threshold <= 0.7
