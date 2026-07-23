"""Revenue-at-risk, CLV, and retention-campaign ROI analysis.

Conventions used throughout this module (kept consistent with
sql/queries/customer_lifetime_value.sql and sql/queries/retention_roi.sql from Day 23, and
dashboard/pages/5_retention.py's ROI estimator from Day 16):
  - "Revenue at risk" / "top revenue at risk" scope to the High risk segment specifically —
    the most urgent cohort — not High+Medium (some other views in this project, e.g.
    vw_risk_segments, use a broader High+Medium "at risk" definition; this module is
    intentionally narrower and documents it here to avoid confusion).
  - CLV = monthly_charges * LEAST(60, 1 / churn_probability) — the standard expected-lifetime
    estimate under a geometric churn process, capped at 60 months.
  - Retention campaign assumptions default to $15/customer cost and a 30% success rate.
"""

import os

import pandas as pd

from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("revenue_analysis")

OUTPUT_DIR = "data/processed"
DEFAULT_CAMPAIGN_COST_PER_CUSTOMER = 15.0
DEFAULT_SUCCESS_RATE = 0.30
CLV_MONTH_CAP = 60


def _load_predictions() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        "SELECT cp.customer_id, cp.churn_probability, cp.risk_segment, rc.monthly_charges, "
        "rc.contract, rc.tenure "
        "FROM churn_predictions cp JOIN raw_customers rc ON rc.customer_id = cp.customer_id",
        engine,
    )


def calculate_revenue_at_risk(df: pd.DataFrame = None) -> float:
    """Returns total monthly_charges of High risk customers ($)."""
    if df is None:
        df = _load_predictions()
    total = df.loc[df["risk_segment"] == "High", "monthly_charges"].sum()
    log.info("Revenue at risk (High risk customers): $%.2f/month", total)
    return round(float(total), 2)


def calculate_clv_by_segment(df: pd.DataFrame = None) -> pd.DataFrame:
    """Returns avg/total CLV per risk segment.

    CLV = monthly_charges * LEAST(60, 1 / churn_probability).
    """
    if df is None:
        df = _load_predictions()

    df = df.copy()
    expected_months = (1.0 / df["churn_probability"].replace(0, pd.NA)).clip(
        upper=CLV_MONTH_CAP
    )
    df["clv"] = df["monthly_charges"] * expected_months.fillna(CLV_MONTH_CAP)

    summary = (
        df.groupby("risk_segment")
        .agg(
            customer_count=("customer_id", "count"),
            avg_clv=("clv", "mean"),
            total_clv=("clv", "sum"),
        )
        .round(2)
        .sort_values("total_clv", ascending=False)
        .reset_index()
    )
    log.info("Calculated CLV for %d segments", len(summary))
    return summary


def estimate_retention_savings(
    campaign_cost: float = DEFAULT_CAMPAIGN_COST_PER_CUSTOMER,
    success_rate: float = DEFAULT_SUCCESS_RATE,
    df: pd.DataFrame = None,
) -> dict:
    """Estimates the ROI of a retention campaign targeting High risk customers.

    campaign_cost is a per-customer cost (default $15, matching the dashboard's ROI
    estimator default). Revenue recovered is annualized (monthly_charges * 12).
    """
    if df is None:
        df = _load_predictions()

    high_risk = df[df["risk_segment"] == "High"]
    customers_targeted = len(high_risk)
    avg_monthly_charges = (
        float(high_risk["monthly_charges"].mean()) if customers_targeted else 0.0
    )

    total_cost = customers_targeted * campaign_cost
    expected_customers_retained = customers_targeted * success_rate
    revenue_recovered = expected_customers_retained * avg_monthly_charges * 12
    roi_pct = (
        ((revenue_recovered - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
    )

    result = {
        "customers_targeted": customers_targeted,
        "campaign_cost_per_customer": campaign_cost,
        "success_rate": success_rate,
        "total_cost": round(total_cost, 2),
        "expected_customers_retained": round(expected_customers_retained, 1),
        "revenue_recovered": round(revenue_recovered, 2),
        "roi_pct": round(roi_pct, 1),
    }
    log.info(
        "Retention ROI: %d customers targeted, $%.2f cost, $%.2f recovered, %.1f%% ROI",
        customers_targeted,
        total_cost,
        revenue_recovered,
        roi_pct,
    )
    return result


def get_top_revenue_at_risk(n: int = 10, df: pd.DataFrame = None) -> pd.DataFrame:
    """Returns the top n High risk customers by monthly_charges."""
    if df is None:
        df = _load_predictions()

    top = (
        df[df["risk_segment"] == "High"]
        .sort_values("monthly_charges", ascending=False)
        .head(n)[
            [
                "customer_id",
                "contract",
                "tenure",
                "monthly_charges",
                "churn_probability",
            ]
        ]
        .reset_index(drop=True)
    )
    log.info("Top %d revenue-at-risk customers retrieved", len(top))
    return top


def save_revenue_report(output_dir: str = OUTPUT_DIR) -> dict:
    """Runs the full revenue analysis and saves a CSV report to output_dir. Returns a summary
    dict of the headline numbers plus the saved path."""
    os.makedirs(output_dir, exist_ok=True)

    df = _load_predictions()
    revenue_at_risk = calculate_revenue_at_risk(df)
    clv_by_segment = calculate_clv_by_segment(df)
    roi = estimate_retention_savings(df=df)
    top_10 = get_top_revenue_at_risk(10, df)

    csv_path = os.path.join(output_dir, "revenue_analysis.csv")
    clv_by_segment.to_csv(csv_path, index=False)

    top_10_path = os.path.join(output_dir, "top_revenue_at_risk.csv")
    top_10.to_csv(top_10_path, index=False)

    log.info("Saved revenue report to %s and %s", csv_path, top_10_path)

    return {
        "revenue_at_risk": revenue_at_risk,
        "clv_by_segment_csv": csv_path,
        "top_10_csv": top_10_path,
        "retention_roi": roi,
    }


if __name__ == "__main__":
    df = _load_predictions()

    print("=== Revenue Analysis ===")
    print(
        f"\nTotal revenue at risk (High risk): ${calculate_revenue_at_risk(df):,.2f}/month"
    )

    print("\nCLV by segment:")
    print(calculate_clv_by_segment(df).to_string(index=False))

    print("\nTop 10 customers at risk:")
    print(get_top_revenue_at_risk(10, df).to_string(index=False))

    print("\nEstimated retention campaign ROI:")
    roi = estimate_retention_savings(df=df)
    for key, value in roi.items():
        print(f"  {key}: {value}")

    report = save_revenue_report()
    print(
        f"\nSaved reports to {report['clv_by_segment_csv']} and {report['top_10_csv']}"
    )
