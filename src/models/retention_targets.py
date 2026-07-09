"""Builds a prioritized retention outreach list from high-risk customers."""
import os

import pandas as pd

from src.utils.db import get_engine

OUTPUT_PATH = "data/processed/retention_targets.csv"


def assign_tier(monthly_charges: float) -> str:
    if monthly_charges > 70:
        return "Tier 1"
    if monthly_charges >= 40:
        return "Tier 2"
    return "Tier 3"


def load_high_risk_customers(engine) -> pd.DataFrame:
    query = """
        SELECT
            cp.customer_id,
            cp.churn_probability,
            cp.risk_segment,
            rc.contract,
            rc.monthly_charges,
            rc.tenure,
            rc.payment_method,
            rc.internet_service
        FROM churn_predictions cp
        JOIN raw_customers rc ON rc.customer_id = cp.customer_id
        WHERE cp.risk_segment = 'High'
    """
    return pd.read_sql(query, engine)


def run() -> pd.DataFrame:
    engine = get_engine()
    targets = load_high_risk_customers(engine)
    print(f"Loaded {len(targets)} high risk customers")

    targets["priority_tier"] = targets["monthly_charges"].apply(assign_tier)

    print("\nBy contract type:")
    print(targets["contract"].value_counts().to_string())

    print("\nBy monthly charges tier:")
    print(targets["priority_tier"].value_counts().sort_index().to_string())

    targets = targets.sort_values(
        by=["priority_tier", "churn_probability"], ascending=[True, False]
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    targets.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved targeting list to {OUTPUT_PATH}")

    print("\nTargeting Summary")
    print("=" * 40)
    tier_summary = targets.groupby("priority_tier").agg(
        customer_count=("customer_id", "count"),
        avg_churn_probability=("churn_probability", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        total_monthly_revenue=("monthly_charges", "sum"),
    ).round(2)
    print(tier_summary.to_string())

    return targets


if __name__ == "__main__":
    run()
