"""Risk-segment profiling — characterizes High/Medium/Low risk customers by demographics,
contract, billing, and service usage, and turns the differences into recommendations.

Pure pandas/SQL (same style as revenue_analysis.py) — no ML model required, so this module
is demo-mode safe as long as a demo predictions dataframe with the columns from _load_customers()
is passed in via the optional `df` parameter on every function.
"""

import os

import pandas as pd

from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("segment_profiler")

OUTPUT_DIR = "data/processed"
SEGMENTS = ["High", "Medium", "Low"]


def _load_customers() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        "SELECT cp.customer_id, cp.churn_probability, cp.risk_segment, rc.tenure, "
        "rc.monthly_charges, rc.total_charges, rc.contract, rc.payment_method, "
        "rc.internet_service, rc.online_security, rc.tech_support, rc.senior_citizen, "
        "rc.paperless_billing "
        "FROM churn_predictions cp JOIN raw_customers rc ON rc.customer_id = cp.customer_id",
        engine,
    )


def _pct(series: pd.Series, match) -> float:
    if len(series) == 0:
        return 0.0
    return round(float((series == match).mean() * 100), 1)


def profile_segment(segment: str, df: pd.DataFrame = None) -> dict:
    """Returns summary stats for a single risk segment: size, avg tenure/charges, and the
    share of customers on month-to-month contracts, electronic check payment, no online
    security, and senior citizens — the levers most commonly tied to churn risk."""
    if df is None:
        df = _load_customers()

    subset = df[df["risk_segment"] == segment]
    profile = {
        "segment": segment,
        "customer_count": len(subset),
        "avg_tenure_months": (
            round(float(subset["tenure"].mean()), 1) if len(subset) else 0.0
        ),
        "avg_monthly_charges": (
            round(float(subset["monthly_charges"].mean()), 2) if len(subset) else 0.0
        ),
        "pct_month_to_month": _pct(subset["contract"], "Month-to-month"),
        "pct_electronic_check": _pct(subset["payment_method"], "Electronic check"),
        "pct_no_online_security": _pct(subset["online_security"], "No"),
        "pct_paperless_billing": _pct(subset["paperless_billing"], "Yes"),
        "pct_senior_citizen": _pct(subset["senior_citizen"], 1),
    }
    log.info("Profiled segment=%s: %d customers", segment, profile["customer_count"])
    return profile


def profile_high_risk(df: pd.DataFrame = None) -> dict:
    return profile_segment("High", df)


def profile_low_risk(df: pd.DataFrame = None) -> dict:
    return profile_segment("Low", df)


def compare_segments(df: pd.DataFrame = None) -> pd.DataFrame:
    """Returns a side-by-side comparison table, one row per segment (High/Medium/Low)."""
    if df is None:
        df = _load_customers()

    rows = [profile_segment(segment, df) for segment in SEGMENTS]
    table = pd.DataFrame(rows)
    log.info("Compared %d segments", len(table))
    return table


def get_segment_recommendations(profiles: pd.DataFrame = None) -> dict:
    """Turns each segment's profile into a short list of business recommendations, driven by
    whichever risk factors are most prevalent in that segment."""
    if profiles is None:
        profiles = compare_segments()

    recommendations = {}
    for _, row in profiles.iterrows():
        segment = row["segment"]
        recs = []
        if row["pct_month_to_month"] >= 50:
            recs.append(
                f"{row['pct_month_to_month']:.0f}% are on month-to-month contracts — "
                "prioritize annual/two-year contract upgrade offers."
            )
        if row["pct_no_online_security"] >= 50:
            recs.append(
                f"{row['pct_no_online_security']:.0f}% lack online security — bundle it in "
                "as a low-cost retention add-on."
            )
        if row["pct_electronic_check"] >= 40:
            recs.append(
                f"{row['pct_electronic_check']:.0f}% pay by electronic check — encourage "
                "autopay/credit-card enrollment, historically a lower-churn payment method."
            )
        if row["avg_tenure_months"] <= 12:
            recs.append(
                f"Average tenure is only {row['avg_tenure_months']:.0f} months — this "
                "segment skews new; focus on onboarding and early-lifecycle engagement."
            )
        if not recs:
            recs.append("No standout risk levers — this segment looks broadly healthy.")
        recommendations[segment] = recs

    log.info("Generated recommendations for %d segments", len(recommendations))
    return recommendations


def save_segment_profiles(output_dir: str = OUTPUT_DIR) -> dict:
    """Runs the full segment comparison and saves a CSV + markdown report to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    df = _load_customers()
    profiles = compare_segments(df)
    recommendations = get_segment_recommendations(profiles)

    csv_path = os.path.join(output_dir, "segment_profiles.csv")
    profiles.to_csv(csv_path, index=False)

    lines = ["# Segment Profiles Report", ""]
    for _, row in profiles.iterrows():
        lines.append(f"## {row['segment']} Risk Segment")
        lines.append(f"- Customers: {row['customer_count']}")
        lines.append(f"- Avg tenure: {row['avg_tenure_months']} months")
        lines.append(f"- Avg monthly charges: ${row['avg_monthly_charges']}")
        lines.append(f"- Month-to-month contracts: {row['pct_month_to_month']}%")
        lines.append(f"- Electronic check payment: {row['pct_electronic_check']}%")
        lines.append(f"- No online security: {row['pct_no_online_security']}%")
        lines.append(f"- Paperless billing: {row['pct_paperless_billing']}%")
        lines.append(f"- Senior citizens: {row['pct_senior_citizen']}%")
        lines.append("")
        lines.append("**Recommendations:**")
        for rec in recommendations[row["segment"]]:
            lines.append(f"- {rec}")
        lines.append("")

    report_path = os.path.join(output_dir, "segment_profiles_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved segment profiles to %s and %s", csv_path, report_path)
    return {"csv_path": csv_path, "report_path": report_path}


if __name__ == "__main__":
    df = _load_customers()

    print("=== Segment Comparison ===")
    comparison = compare_segments(df)
    print(comparison.to_string(index=False))

    print("\n=== Recommendations ===")
    for segment, recs in get_segment_recommendations(comparison).items():
        print(f"\n{segment}:")
        for rec in recs:
            print(f"  - {rec}")

    result = save_segment_profiles()
    print(
        f"\nSaved segment profiles to {result['csv_path']} and {result['report_path']}"
    )
