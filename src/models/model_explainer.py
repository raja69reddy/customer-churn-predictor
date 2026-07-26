"""Per-customer prediction explainability — full SHAP breakdown, plain-English summaries, and
comparison against the population average for a single customer_id.

This is a customer_id-first companion to src/models/predict.py's ChurnPredictor (which takes an
in-memory customer dict, not an id) and src/analysis/churn_drivers.py (which analyzes drivers
across the whole population, not one customer) — reuses ChurnPredictor's model loading and
feature engineering rather than reimplementing it.

Requires the full ML stack (xgboost/shap) — not demo-mode safe.
"""

import os

import pandas as pd
import shap

from src.models.predict import ChurnPredictor
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("model_explainer")

OUTPUT_DIR = "data/processed"
NUMERIC_COMPARISON_FEATURES = [
    "charge_per_month",
    "services_count",
    "contract_risk_score",
    "payment_risk_score",
]

_predictor = ChurnPredictor()


def _load_customer(customer_id: str) -> dict:
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM raw_customers WHERE customer_id = %(cid)s",
        engine,
        params={"cid": customer_id},
    )
    if df.empty:
        raise ValueError(f"Customer '{customer_id}' not found.")
    return df.iloc[0].to_dict()


def explain_prediction(customer_id: str) -> dict:
    """Full SHAP breakdown for one customer: churn probability, risk segment, SHAP base value
    (expected value the explainer starts from), and every feature's contribution — everything
    a waterfall chart needs."""
    if _predictor.model is None:
        _predictor.load_best_model()

    customer_dict = _load_customer(customer_id)
    prediction = _predictor.predict_single(customer_dict)

    X = _predictor._engineer_features(pd.DataFrame([customer_dict])).astype(float)
    explainer = shap.TreeExplainer(
        _predictor.model, feature_perturbation="tree_path_dependent"
    )
    shap_values = explainer(X)

    factors = [
        {
            "feature": feature,
            "feature_value": float(X.iloc[0][feature]),
            "shap_value": float(value),
            "direction": "increases risk" if value > 0 else "decreases risk",
        }
        for feature, value in zip(X.columns, shap_values.values[0])
    ]
    factors.sort(key=lambda f: abs(f["shap_value"]), reverse=True)

    result = {
        "customer_id": customer_id,
        "churn_probability": prediction["churn_probability"],
        "risk_segment": prediction["risk_segment"],
        "base_value": float(shap_values.base_values[0]),
        "factors": factors,
    }
    log.info(
        "Explained prediction for %s: probability=%.4f segment=%s",
        customer_id,
        result["churn_probability"],
        result["risk_segment"],
    )
    return result


def get_top_factors(customer_id: str, n: int = 5) -> list:
    """Top n SHAP factors (by absolute impact) for a customer."""
    explanation = explain_prediction(customer_id)
    top_factors = explanation["factors"][:n]
    log.info("Top %d factors retrieved for %s", len(top_factors), customer_id)
    return top_factors


def generate_explanation_text(factors: list) -> str:
    """Turns a list of SHAP factor dicts (feature, shap_value, direction) into a plain-English
    sentence, e.g. 'The top factors are contract risk score (increases risk), ...'."""
    if not factors:
        return "No contributing factors were provided."

    described = [
        f"{factor['feature'].replace('_', ' ')} ({factor['direction']})"
        for factor in factors
    ]
    if len(described) == 1:
        factor_text = described[0]
    else:
        factor_text = ", ".join(described[:-1]) + f", and {described[-1]}"

    return f"The top factors influencing this prediction are {factor_text}."


def compare_customer_to_average(customer_id: str) -> dict:
    """Compares a customer's engineered model features (plus raw tenure/monthly_charges, more
    interpretable for a business audience) against the population average from
    processed_customers/raw_customers."""
    if _predictor.model is None:
        _predictor.load_best_model()

    customer_dict = _load_customer(customer_id)
    X = _predictor._engineer_features(pd.DataFrame([customer_dict])).astype(float)

    engine = get_engine()
    numeric_avgs = pd.read_sql(
        f"SELECT {', '.join(f'AVG({c}) AS {c}' for c in NUMERIC_COMPARISON_FEATURES)} "
        "FROM processed_customers",
        engine,
    ).iloc[0]
    raw_avgs = pd.read_sql(
        "SELECT AVG(tenure) AS tenure, AVG(monthly_charges) AS monthly_charges "
        "FROM raw_customers",
        engine,
    ).iloc[0]

    comparison = {}
    for feature in NUMERIC_COMPARISON_FEATURES:
        customer_value = float(X.iloc[0][feature])
        population_average = float(numeric_avgs[feature])
        comparison[feature] = {
            "customer_value": round(customer_value, 4),
            "population_average": round(population_average, 4),
            "diff": round(customer_value - population_average, 4),
        }
    for feature in ("tenure", "monthly_charges"):
        customer_value = float(customer_dict[feature])
        population_average = float(raw_avgs[feature])
        comparison[feature] = {
            "customer_value": round(customer_value, 4),
            "population_average": round(population_average, 4),
            "diff": round(customer_value - population_average, 4),
        }

    log.info(
        "Compared %s to population average across %d features",
        customer_id,
        len(comparison),
    )
    return comparison


def save_explanation(customer_id: str, output_dir: str = OUTPUT_DIR) -> str:
    """Builds the full explanation (SHAP factors + plain-English text + population comparison)
    for a customer and saves it as markdown to output_dir. Returns the saved path."""
    os.makedirs(output_dir, exist_ok=True)

    explanation = explain_prediction(customer_id)
    top_factors = explanation["factors"][:5]
    text = generate_explanation_text(top_factors)
    comparison = compare_customer_to_average(customer_id)

    lines = [
        f"# Explanation — {customer_id}",
        "",
        f"Churn probability: {explanation['churn_probability']:.1%} "
        f"(risk segment: {explanation['risk_segment']})",
        "",
        text,
        "",
        "## Top 5 Risk Factors",
    ]
    for factor in top_factors:
        lines.append(
            f"- {factor['feature']}: value={factor['feature_value']:.4f}, "
            f"SHAP={factor['shap_value']:+.4f} ({factor['direction']})"
        )

    lines.append("")
    lines.append("## Customer vs Population Average")
    for feature, values in comparison.items():
        lines.append(
            f"- {feature}: customer={values['customer_value']}, "
            f"average={values['population_average']}, diff={values['diff']:+.4f}"
        )

    safe_id = customer_id.replace("/", "_")
    path = os.path.join(output_dir, f"explanation_{safe_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved explanation for %s to %s", customer_id, path)
    return path


if __name__ == "__main__":
    import sys

    customer_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not customer_id:
        print("Usage: python -m src.models.model_explainer <customer_id>")
        raise SystemExit(1)

    top_factors = get_top_factors(customer_id, n=5)
    print(f"=== Top 5 Risk Factors for {customer_id} ===")
    for factor in top_factors:
        print(
            f"  {factor['feature']}: SHAP={factor['shap_value']:+.4f} ({factor['direction']})"
        )

    print(f"\n=== Explanation Text ===\n{generate_explanation_text(top_factors)}")

    print("\n=== Customer vs Population Average ===")
    for feature, values in compare_customer_to_average(customer_id).items():
        print(f"  {feature}: {values}")

    path = save_explanation(customer_id)
    print(f"\nSaved explanation to {path}")
