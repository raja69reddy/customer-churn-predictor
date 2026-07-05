"""Churn prediction — scores individual customers or batches with the active best model."""
import os

import joblib
import pandas as pd
import shap
from openai import OpenAI

from src.features.engineering import FeatureEngineer
from src.utils.db import get_engine

RETENTION_ACTIONS = {
    "High": [
        "Offer a loyalty discount or incentive to switch to a longer-term contract",
        "Assign a customer success rep for proactive outreach this week",
        "Offer a free service add-on (e.g. online security) to increase switching cost",
    ],
    "Medium": [
        "Send a satisfaction survey to identify pain points",
        "Highlight underused services or bundle benefits",
        "Offer a modest loyalty discount tied to contract renewal",
    ],
    "Low": [
        "No immediate action needed — monitor at the next billing cycle",
    ],
}

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

    def explain_with_gpt(self, customer_dict: dict, prediction: dict) -> str:
        top_factors = self.explain_prediction(customer_dict, top_n=3)
        factor_text = ", ".join(
            f"{factor['feature'].replace('_', ' ')} ({factor['direction']})" for factor in top_factors
        )

        prompt = (
            f"A telecom customer has a predicted churn probability of {prediction['churn_probability']:.0%} "
            f"(risk segment: {prediction['risk_segment']}). "
            f"The top contributing factors, from a SHAP explanation, are: {factor_text}. "
            "In 2-3 plain-English sentences, explain why this customer is likely to churn, "
            "referencing the top factors by name."
        )

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_key":
            return self._fallback_explanation(prediction, top_factors)

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API call failed ({e}); falling back to template explanation.")
            return self._fallback_explanation(prediction, top_factors)

    def _fallback_explanation(self, prediction: dict, top_factors: list) -> str:
        names = [factor["feature"].replace("_", " ") for factor in top_factors]
        return (
            "[template fallback — OPENAI_API_KEY not configured] "
            f"This customer has a {prediction['risk_segment'].lower()} churn risk "
            f"({prediction['churn_probability']:.0%} probability), driven mainly by {names[0]}, "
            f"{names[1]}, and {names[2]}."
        )

    def format_explanation(self, explanation: str, prediction: dict) -> str:
        actions = RETENTION_ACTIONS.get(prediction["risk_segment"], RETENTION_ACTIONS["Low"])

        lines = [
            "Churn Risk Explanation",
            "=" * 40,
            f"Risk segment: {prediction['risk_segment']}  "
            f"(churn probability: {prediction['churn_probability']:.1%})",
            "",
            explanation,
            "",
            "Recommended Retention Actions:",
        ]
        lines.extend(f"  - {action}" for action in actions)
        return "\n".join(lines)
