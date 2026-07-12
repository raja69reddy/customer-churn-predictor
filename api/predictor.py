"""API-facing predictor — wraps ChurnPredictor's feature engineering with the MLflow Production model."""

from datetime import datetime

import pandas as pd
from mlflow.tracking import MlflowClient

from api.models import CustomerInput, PredictionOutput, RiskFactor
from src.models import mlflow_registry, mlflow_setup
from src.models.predict import RETENTION_ACTIONS, ChurnPredictor
from src.utils.db import get_engine


class APIPredictor:
    """Loads the MLflow Production model and scores customers for the API layer."""

    def __init__(self):
        self._churn_predictor = ChurnPredictor()
        self.model_name = None
        self.model_version = None

    def load_model(self):
        mlflow_setup.configure_tracking()
        client = MlflowClient()
        registered_name = mlflow_registry.REGISTERED_MODEL_NAME

        versions = client.search_model_versions(f"name='{registered_name}'")
        production = [v for v in versions if v.current_stage == "Production"]
        if not production:
            raise RuntimeError(
                f"No Production version found for '{registered_name}' in MLflow registry."
            )
        version = production[0].version

        self._churn_predictor.model = mlflow_registry.get_production_model(
            registered_name
        )
        self.model_name = registered_name
        self.model_version = f"{registered_name}_v{version}"
        self._churn_predictor.model_name = self.model_version
        return self._churn_predictor.model

    def _ensure_loaded(self) -> None:
        if self._churn_predictor.model is None:
            self.load_model()

    def preprocess(self, customer_input: CustomerInput) -> dict:
        """Converts a validated CustomerInput into the raw-field dict ChurnPredictor expects."""
        return customer_input.model_dump(exclude={"customer_id"})

    def predict(self, customer_input: CustomerInput) -> PredictionOutput:
        self._ensure_loaded()
        customer_dict = self.preprocess(customer_input)

        result = self._churn_predictor.predict_single(customer_dict)
        factors = self._churn_predictor.explain_prediction(customer_dict, top_n=5)
        actions = RETENTION_ACTIONS.get(
            result["risk_segment"], RETENTION_ACTIONS["Low"]
        )

        return PredictionOutput(
            customer_id=customer_input.customer_id or "UNSPECIFIED",
            churn_probability=result["churn_probability"],
            risk_segment=result["risk_segment"],
            top_risk_factors=[RiskFactor(**factor) for factor in factors],
            recommended_actions=actions,
            model_version=self.model_version,
            predicted_at=datetime.now(),
        )

    def predict_batch(self, customers: list[CustomerInput]) -> list[PredictionOutput]:
        self._ensure_loaded()
        return [self.predict(customer) for customer in customers]

    def explain(self, customer_dict: dict) -> dict:
        """Returns churn probability, risk segment, and an AI (or template-fallback) explanation."""
        self._ensure_loaded()
        result = self._churn_predictor.predict_single(customer_dict)
        explanation = self._churn_predictor.explain_with_gpt(customer_dict, result)
        return {
            "churn_probability": result["churn_probability"],
            "risk_segment": result["risk_segment"],
            "explanation": explanation,
        }

    def get_model_info(self) -> dict:
        self._ensure_loaded()
        engine = get_engine()
        registry_row = pd.read_sql(
            "SELECT * FROM model_registry WHERE is_active = TRUE LIMIT 1", engine
        )

        info = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "mlflow_stage": "Production",
        }
        if not registry_row.empty:
            row = registry_row.iloc[0]
            trained_at = row["trained_at"]
            info.update(
                {
                    "accuracy": float(row["accuracy"]),
                    "auc_score": float(row["auc_score"]),
                    "f1_score": float(row["f1_score"]),
                    "trained_at": (
                        trained_at.isoformat()
                        if hasattr(trained_at, "isoformat")
                        else str(trained_at)
                    ),
                }
            )
        return info
