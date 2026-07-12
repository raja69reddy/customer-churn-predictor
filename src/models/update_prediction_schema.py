"""Adds retention/monitoring columns to churn_predictions and backfills them for all rows."""

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine

ALTER_STATEMENTS = [
    "ALTER TABLE churn_predictions ADD COLUMN IF NOT EXISTS retention_priority VARCHAR(10)",
    "ALTER TABLE churn_predictions ADD COLUMN IF NOT EXISTS recommended_action TEXT",
    "ALTER TABLE churn_predictions ADD COLUMN IF NOT EXISTS estimated_revenue_at_risk FLOAT",
    "ALTER TABLE churn_predictions ADD COLUMN IF NOT EXISTS days_since_last_score INTEGER",
]

RECOMMENDED_ACTIONS = {
    (
        "High",
        "Tier1",
    ): "Priority outreach: offer loyalty discount + contract upgrade incentive",
    (
        "High",
        "Tier2",
    ): "Standard outreach: satisfaction survey + modest loyalty discount",
    ("High", "Tier3"): "Low-cost outreach: email survey, monitor for escalation",
    (
        "Medium",
        None,
    ): "Monitor closely; consider proactive engagement before next cycle",
    ("Low", None): "No action needed - monitor periodically",
}


def assign_retention_priority(monthly_charges: float) -> str:
    if monthly_charges > 70:
        return "Tier1"
    if monthly_charges >= 40:
        return "Tier2"
    return "Tier3"


def recommended_action(risk_segment: str, retention_priority: str) -> str:
    if risk_segment == "High":
        return RECOMMENDED_ACTIONS[("High", retention_priority)]
    return RECOMMENDED_ACTIONS[(risk_segment, None)]


def run() -> None:
    engine = get_engine()

    with engine.begin() as conn:
        for statement in ALTER_STATEMENTS:
            conn.execute(text(statement))
    print(
        "Added retention_priority, recommended_action, estimated_revenue_at_risk, days_since_last_score columns"
    )

    df = pd.read_sql(
        """
        SELECT cp.customer_id, cp.churn_probability, cp.risk_segment, cp.predicted_at, rc.monthly_charges
        FROM churn_predictions cp
        JOIN raw_customers rc ON rc.customer_id = cp.customer_id
        """,
        engine,
    )

    df["retention_priority"] = None
    high_mask = df["risk_segment"] == "High"
    df.loc[high_mask, "retention_priority"] = df.loc[
        high_mask, "monthly_charges"
    ].apply(assign_retention_priority)

    df["recommended_action"] = df.apply(
        lambda row: recommended_action(row["risk_segment"], row["retention_priority"]),
        axis=1,
    )
    df["estimated_revenue_at_risk"] = (
        df["monthly_charges"] * df["churn_probability"]
    ).round(2)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text(
                    "UPDATE churn_predictions SET "
                    "retention_priority = :priority, "
                    "recommended_action = :action, "
                    "estimated_revenue_at_risk = :revenue, "
                    "days_since_last_score = EXTRACT(DAY FROM NOW() - predicted_at)::INTEGER "
                    "WHERE customer_id = :cid"
                ),
                {
                    "priority": row["retention_priority"],
                    "action": row["recommended_action"],
                    "revenue": float(row["estimated_revenue_at_risk"]),
                    "cid": row["customer_id"],
                },
            )

    print(f"Updated {len(df)} existing predictions with new column values")

    summary = pd.read_sql(
        "SELECT risk_segment, retention_priority, COUNT(*) AS n, "
        "ROUND(AVG(estimated_revenue_at_risk)::NUMERIC, 2) AS avg_revenue_at_risk, "
        "ROUND(AVG(days_since_last_score)::NUMERIC, 2) AS avg_days_since_score "
        "FROM churn_predictions GROUP BY risk_segment, retention_priority "
        "ORDER BY risk_segment, retention_priority",
        engine,
    )
    print("\nBackfill summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
