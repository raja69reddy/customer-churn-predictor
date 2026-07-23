"""Cohort analysis — weekly customer cohorts (implied from tenure) and their retention rates.

Methodology note: `raw_customers.created_at` is just the bulk ingestion timestamp (identical
for all 7,043 rows), not a real signup date, so it can't be used to build cohorts directly.
Instead, each customer's implied join date is derived as `reference_date - tenure` (tenure is
in months), then bucketed into weekly cohorts. Retention rate per cohort is a snapshot metric
(% of that cohort not churned as of now) rather than a true multi-period longitudinal curve —
this dataset has one observation per customer, not repeated period-by-period activity, so a
full "weeks since signup" retention curve isn't reconstructable. This is a standard, documented
proxy for cohort analysis on snapshot data.
"""

import os
from datetime import datetime

import pandas as pd

from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("cohort_analysis")

OUTPUT_DIR = "data/processed"
AVG_DAYS_PER_MONTH = 30.44


def _load_customers(reference_date: datetime = None) -> pd.DataFrame:
    """Loads raw_customers and adds implied_join_date / cohort_week columns."""
    reference_date = reference_date or datetime.now()
    engine = get_engine()
    df = pd.read_sql(
        "SELECT customer_id, tenure, contract, monthly_charges, churn FROM raw_customers",
        engine,
    )
    df["implied_join_date"] = reference_date - pd.to_timedelta(
        df["tenure"] * AVG_DAYS_PER_MONTH, unit="D"
    )
    df["cohort_week"] = (
        df["implied_join_date"].dt.to_period("W").apply(lambda p: p.start_time.date())
    )
    return df


def build_cohort_table(df: pd.DataFrame = None) -> pd.DataFrame:
    """Builds a weekly-cohort table: cohort_week, cohort_size, churned, churn_rate_pct.

    If df isn't provided, loads raw_customers directly (via _load_customers()).
    """
    if df is None or "cohort_week" not in df.columns:
        df = _load_customers()

    table = (
        df.groupby("cohort_week")
        .agg(
            cohort_size=("customer_id", "count"),
            churned=("churn", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    table["churn_rate_pct"] = round(100.0 * table["churned"] / table["cohort_size"], 2)
    table = table.sort_values("cohort_week").reset_index(drop=True)
    log.info(
        "Built cohort table: %d weekly cohorts, %d customers",
        len(table),
        int(table["cohort_size"].sum()),
    )
    return table


def calculate_retention_rate(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a retention_rate_pct column (100 - churn_rate_pct) to a cohort table."""
    df = cohort_df.copy()
    df["retention_rate_pct"] = round(100.0 - df["churn_rate_pct"], 2)
    return df


def _retention_by_contract(df: pd.DataFrame = None) -> pd.DataFrame:
    """Pivot: cohort_week (rows) x contract type (cols) -> retention rate (%).

    Used by plot_cohort_heatmap() to give the heatmap real 2D structure — a single
    snapshot retention rate per cohort week has no second dimension on its own.
    """
    if df is None or "cohort_week" not in df.columns:
        df = _load_customers()

    grouped = (
        df.groupby(["cohort_week", "contract"])
        .agg(
            cohort_size=("customer_id", "count"),
            churned=("churn", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    grouped["retention_rate_pct"] = round(
        100.0 - 100.0 * grouped["churned"] / grouped["cohort_size"], 2
    )
    pivot = grouped.pivot(
        index="cohort_week", columns="contract", values="retention_rate_pct"
    )
    return pivot.sort_index()


def plot_cohort_heatmap(
    retention_pivot: pd.DataFrame = None,
    output_path: str = None,
    recent_weeks: int = 20,
) -> str:
    """Saves a cohort-week x contract-type retention heatmap as a PNG. Returns the saved path.

    Only the most recent `recent_weeks` cohorts are plotted — tenure spans up to 72 months
    (~300 weekly cohorts), which is unreadable in one heatmap. build_cohort_table() still
    returns the full history; this is purely a plotting-window limit.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if retention_pivot is None:
        retention_pivot = _retention_by_contract()

    plot_data = retention_pivot.tail(recent_weeks)

    plt.figure(figsize=(8, max(4, len(plot_data) * 0.3)))
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Retention Rate (%)"},
    )
    plt.title(
        f"Cohort Retention Rate by Contract Type (last {len(plot_data)} weekly cohorts)"
    )
    plt.xlabel("Contract Type")
    plt.ylabel("Cohort Week (implied join week)")
    plt.tight_layout()

    output_path = output_path or os.path.join(OUTPUT_DIR, "cohort_heatmap.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    log.info("Saved cohort heatmap to %s", output_path)
    return output_path


def get_best_cohort(cohort_df: pd.DataFrame = None, min_size: int = 5) -> pd.Series:
    """Returns the cohort row with the highest retention rate.

    Cohorts smaller than min_size are excluded so a tiny, noisy cohort (e.g. 1 customer,
    100% or 0% retention) can't win by chance.
    """
    if cohort_df is None:
        cohort_df = build_cohort_table()
    if "retention_rate_pct" not in cohort_df.columns:
        cohort_df = calculate_retention_rate(cohort_df)

    eligible = cohort_df[cohort_df["cohort_size"] >= min_size]
    return eligible.loc[eligible["retention_rate_pct"].idxmax()]


def get_worst_cohort(cohort_df: pd.DataFrame = None, min_size: int = 5) -> pd.Series:
    """Returns the cohort row with the lowest retention rate (same min_size filter as
    get_best_cohort)."""
    if cohort_df is None:
        cohort_df = build_cohort_table()
    if "retention_rate_pct" not in cohort_df.columns:
        cohort_df = calculate_retention_rate(cohort_df)

    eligible = cohort_df[cohort_df["cohort_size"] >= min_size]
    return eligible.loc[eligible["retention_rate_pct"].idxmin()]


def save_cohort_report(output_dir: str = OUTPUT_DIR) -> dict:
    """Runs the full cohort analysis and saves the cohort table (CSV) and heatmap (PNG) to
    output_dir. Returns a summary dict including the saved paths and best/worst cohorts.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw = _load_customers()
    cohort_table = calculate_retention_rate(build_cohort_table(raw))
    csv_path = os.path.join(output_dir, "cohort_analysis.csv")
    cohort_table.to_csv(csv_path, index=False)
    log.info("Saved cohort table to %s", csv_path)

    heatmap_path = plot_cohort_heatmap(
        _retention_by_contract(raw),
        output_path=os.path.join(output_dir, "cohort_heatmap.png"),
    )

    best = get_best_cohort(cohort_table)
    worst = get_worst_cohort(cohort_table)
    log.info(
        "Best cohort: %s (%.2f%% retention) | Worst cohort: %s (%.2f%% retention)",
        best["cohort_week"],
        best["retention_rate_pct"],
        worst["cohort_week"],
        worst["retention_rate_pct"],
    )

    return {
        "csv_path": csv_path,
        "heatmap_path": heatmap_path,
        "total_cohorts": int(len(cohort_table)),
        "best_cohort_week": str(best["cohort_week"]),
        "best_cohort_retention_pct": float(best["retention_rate_pct"]),
        "worst_cohort_week": str(worst["cohort_week"]),
        "worst_cohort_retention_pct": float(worst["retention_rate_pct"]),
    }


if __name__ == "__main__":
    report = save_cohort_report()
    print("Cohort analysis report saved:")
    for key, value in report.items():
        print(f"  {key}: {value}")
