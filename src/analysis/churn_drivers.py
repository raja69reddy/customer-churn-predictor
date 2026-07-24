"""SHAP-based churn driver analysis — top features overall, top features per risk segment,
and business-facing recommendations built from them.

Requires the full ML stack (xgboost/lightgbm/shap/sklearn) and a trained active model, so
unlike src/analysis/revenue_analysis.py this module is not demo-mode safe — it's meant to be
run locally / in CI with the full requirements.txt, not imported by the lightweight dashboard.
"""

import os

import numpy as np
import pandas as pd

from src.models.train import ModelTrainer
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("churn_drivers")

OUTPUT_DIR = "data/processed"

# Human-readable, actionable framing for each engineered feature. Features not in this map
# (there shouldn't be any, given FEATURE_COLUMNS is fixed) fall back to a generic message.
INSIGHT_TEMPLATES = {
    "charge_per_month": (
        "Charge-per-month is the single largest churn driver — target proactive discount "
        "or loyalty offers at customers with high historical spend and early risk signals."
    ),
    "tenure_group_New Customer": (
        "New customers (under ~12 months) churn far more than established ones — invest in "
        "a structured onboarding / early-engagement program for the first 90 days."
    ),
    "tenure_group_Growing Customer": (
        "Growing-tenure customers (roughly 1-2 years) are a moderate-risk group — a periodic "
        "check-in or loyalty milestone offer can reinforce the relationship before it forms."
    ),
    "tenure_group_Established Customer": (
        "Established customers (roughly 2-4 years) are comparatively low risk — maintain "
        "service quality rather than heavy discounting."
    ),
    "tenure_group_Loyal Customer": (
        "Loyal customers (4+ years) are the lowest-risk group — focus retention spend "
        "elsewhere and consider them for referral/advocacy programs instead."
    ),
    "contract_risk_score": (
        "Contract type is a major lever — incentivize switching from month-to-month to "
        "one- or two-year contracts with a meaningful signup discount."
    ),
    "payment_risk_score": (
        "Payment method correlates with churn — encourage enrollment in autopay (bank "
        "transfer or credit card) over manual payment methods like electronic/mailed check."
    ),
    "services_count": (
        "Customers with fewer add-on services churn more — bundle or trial promotions "
        "that increase service attachment raise switching cost and lower churn risk."
    ),
}
DEFAULT_INSIGHT = (
    "This feature has a measurable effect on churn risk — investigate further."
)


def _load_model_and_test_data():
    """Returns (model, model_name, trainer) for the currently active model, with
    trainer.X_test/y_test populated."""
    engine = get_engine()
    active = pd.read_sql(
        "SELECT model_name FROM model_registry WHERE is_active = TRUE LIMIT 1", engine
    )
    if active.empty:
        raise RuntimeError("No active model found in model_registry.")
    model_name = active.iloc[0]["model_name"]

    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)
    model = trainer.load_model(model_name)
    return model, model_name, trainer


def _shap_values_for(model, X: pd.DataFrame):
    import shap

    X_float = X.astype(float)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return explainer(X_float), X_float


def get_top_churn_drivers(n: int = 10) -> pd.DataFrame:
    """Returns the top n features by mean |SHAP value| across the test set."""
    model, model_name, trainer = _load_model_and_test_data()
    shap_values, X_float = _shap_values_for(model, trainer.X_test)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    result = (
        pd.DataFrame({"feature": X_float.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    log.info("Top %d churn drivers computed from model %s", len(result), model_name)
    return result


def get_driver_by_segment(segment: str, n: int = 5) -> pd.DataFrame:
    """Returns the top n features by mean |SHAP value|, restricted to test-set customers
    currently scored in the given risk_segment (High/Medium/Low)."""
    model, model_name, trainer = _load_model_and_test_data()

    test_customer_ids = trainer.df.loc[trainer.X_test.index, "customer_id"]
    engine = get_engine()
    segment_ids = pd.read_sql(
        "SELECT customer_id FROM churn_predictions WHERE risk_segment = %(segment)s",
        engine,
        params={"segment": segment},
    )["customer_id"]

    mask = test_customer_ids.isin(set(segment_ids)).values
    X_segment = trainer.X_test[mask]

    if X_segment.empty:
        log.warning(
            "No test-set overlap for segment=%s — returning empty driver table", segment
        )
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    shap_values, X_float = _shap_values_for(model, X_segment)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    result = (
        pd.DataFrame({"feature": X_float.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    log.info(
        "Top %d drivers for segment=%s computed from %d test customers (model %s)",
        len(result),
        segment,
        len(X_segment),
        model_name,
    )
    return result


def get_actionable_insights(top_drivers: pd.DataFrame = None) -> list:
    """Returns a list of {feature, insight} business recommendations built from the top
    overall churn drivers."""
    if top_drivers is None:
        top_drivers = get_top_churn_drivers()

    insights = [
        {
            "feature": row["feature"],
            "insight": INSIGHT_TEMPLATES.get(row["feature"], DEFAULT_INSIGHT),
        }
        for _, row in top_drivers.iterrows()
    ]
    return insights


def format_driver_report() -> str:
    """Builds a full markdown report: top drivers, per-segment drivers, and actionable
    insights."""
    top_drivers = get_top_churn_drivers()
    insights = get_actionable_insights(top_drivers)

    lines = ["# Churn Drivers Report", ""]
    lines.append("## Top Churn Drivers (SHAP)")
    lines.append("")
    lines.append("| Feature | Mean |SHAP| |")
    lines.append("|---|---|")
    for _, row in top_drivers.iterrows():
        lines.append(f"| {row['feature']} | {row['mean_abs_shap']:.4f} |")
    lines.append("")

    lines.append("## Actionable Insights")
    lines.append("")
    for item in insights:
        lines.append(f"- **{item['feature']}**: {item['insight']}")
    lines.append("")

    lines.append("## Top Drivers by Risk Segment")
    for segment in ["High", "Medium", "Low"]:
        lines.append("")
        lines.append(f"### {segment} Risk")
        segment_drivers = get_driver_by_segment(segment)
        if segment_drivers.empty:
            lines.append("_No test-set customers available for this segment._")
            continue
        lines.append("")
        lines.append("| Feature | Mean |SHAP| |")
        lines.append("|---|---|")
        for _, row in segment_drivers.iterrows():
            lines.append(f"| {row['feature']} | {row['mean_abs_shap']:.4f} |")

    return "\n".join(lines)


def save_drivers_report(output_dir: str = OUTPUT_DIR) -> dict:
    """Runs the full driver analysis and saves both the top-drivers CSV and the full
    markdown report to output_dir. Returns a dict of the saved paths."""
    os.makedirs(output_dir, exist_ok=True)

    top_drivers = get_top_churn_drivers()
    csv_path = os.path.join(output_dir, "churn_drivers.csv")
    top_drivers.to_csv(csv_path, index=False)

    report = format_driver_report()
    md_path = os.path.join(output_dir, "churn_drivers_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    log.info("Saved churn drivers report to %s and %s", csv_path, md_path)
    return {"csv_path": csv_path, "report_path": md_path}


if __name__ == "__main__":
    result = save_drivers_report()
    print(f"Saved churn drivers report to {result['report_path']}")
    print(f"Saved churn drivers CSV to {result['csv_path']}")
