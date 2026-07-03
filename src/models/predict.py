"""Churn prediction — scores individual customers or batches with the active best model."""
import joblib
import pandas as pd
import shap

from src.features.engineering import FeatureEngineer
from src.utils.db import get_engine

FEATURE_COLUMNS = [
    "charge_per_month", "services_count", "contract_risk_score", "payment_risk_score",
    "tenure_group_Established Customer", "tenure_group_Growing Customer",
    "tenure_group_Loyal Customer", "tenure_group_New Customer",
]


class ChurnPredictor:
    """Loads the active best model and scores customers for churn risk."""

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_best_model(self):
        engine = get_engine()
        registry = pd.read_sql("SELECT * FROM model_registry WHERE is_active = TRUE", engine)
        if registry.empty:
            raise RuntimeError("No active model found in model_registry.")

        self.model_name = registry.iloc[0]["model_name"]
        self.model = joblib.load(f"models/{self.model_name}.pkl")
        return self.model

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        fe = FeatureEngineer()
        df = fe.create_tenure_group(df)
        df = fe.create_charge_per_month(df)
        df = fe.create_services_count(df)
        df = fe.create_contract_risk_score(df)
        df = fe.create_payment_risk_score(df)
        df = pd.get_dummies(df, columns=["tenure_group"], prefix="tenure_group")

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = False

        return df[FEATURE_COLUMNS]

    def get_risk_segment(self, probability: float) -> str:
        if probability >= 0.7:
            return "High"
        if probability >= 0.4:
            return "Medium"
        return "Low"

    def predict_single(self, customer_dict: dict) -> dict:
        if self.model is None:
            self.load_best_model()

        df = pd.DataFrame([customer_dict])
        X = self._engineer_features(df)
        probability = float(self.model.predict_proba(X)[0, 1])

        return {
            "churn_probability": probability,
            "risk_segment": self.get_risk_segment(probability),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self.load_best_model()

        X = self._engineer_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]

        result = df.copy()
        result["churn_probability"] = probabilities
        result["risk_segment"] = [self.get_risk_segment(p) for p in probabilities]
        return result

    def explain_prediction(self, customer_dict: dict, top_n: int = 3) -> list:
        if self.model is None:
            self.load_best_model()

        df = pd.DataFrame([customer_dict])
        X = self._engineer_features(df).astype(float)

        explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X)
        values = shap_values.values[0]

        ranking = sorted(zip(X.columns, values), key=lambda pair: abs(pair[1]), reverse=True)
        top_factors = ranking[:top_n]

        return [
            {"feature": feature, "shap_value": float(value), "direction": "increases risk" if value > 0 else "decreases risk"}
            for feature, value in top_factors
        ]
