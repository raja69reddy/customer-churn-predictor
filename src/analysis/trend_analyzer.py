"""Churn trend analysis — monthly churn-rate trend, a seasonal check, and a simple linear
forecast.

Data-source note: real prediction history (churn_predictions + churn_predictions_history) only
spans ~1 calendar month (this project's own build history), too short for a genuine multi-month
trend. Instead, this module reuses src/analysis/cohort_analysis.py's "implied_join_date =
reference_date - tenure * 30.44 days" proxy (raw_customers.created_at is a single bulk-ingestion
timestamp, not real signup data) but buckets by MONTH instead of week, and reports the actual
(ground-truth) churn rate per implied-join-month — tenure spans 1-72 months, so this gives
genuine multi-month spread unlike the live prediction history. Same underlying proxy as
cohort_analysis.py, different lens: "how did the churn rate of customers who joined in month X
compare across months" rather than cohort_analysis.py's retention-rate framing.

Pure pandas/SQL — demo-mode safe when a pre-loaded df is passed in.

IMPORTANT caveat verified against real data: this synthetic dataset's `churn` column is
strongly tied to `tenure` by construction — churn rate is ~79% for tenure<=6 months, ~69% for
6-12, ~14% for 12-24, and 0% beyond 24 months (churned customers' tenure is frozen at whatever
point they left, so short tenure disproportionately means "already churned," not "new and
healthy"). Since analyze_churn_trend()'s most-recent months necessarily draw from the
lowest-tenure customers, its churn_rate values (~75-85%) reflect this generator artifact, not a
literal "75-85% of the business is churning right now" — do not present these numbers as the
company's real current churn rate (that's the ~26% dataset-wide baseline established since Day
2). Treat this module's output as a relative month-over-month trend/forecast shape, not an
absolute churn-rate reading.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("trend_analyzer")

OUTPUT_DIR = "data/processed"
AVG_DAYS_PER_MONTH = 30.44


def _load_customers(reference_date: datetime = None) -> pd.DataFrame:
    reference_date = reference_date or datetime.now()
    engine = get_engine()
    df = pd.read_sql("SELECT customer_id, tenure, churn FROM raw_customers", engine)
    df["implied_join_date"] = reference_date - pd.to_timedelta(
        df["tenure"] * AVG_DAYS_PER_MONTH, unit="D"
    )
    df["join_month"] = df["implied_join_date"].dt.to_period("M")
    return df


def analyze_churn_trend(periods: int = 6, df: pd.DataFrame = None) -> pd.DataFrame:
    """Actual churn rate per implied-join-month, for the most recent `periods` months."""
    if df is None:
        df = _load_customers()

    monthly = (
        df.groupby("join_month")
        .agg(
            customers=("customer_id", "count"),
            churned=("churn", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    monthly["churn_rate"] = round(100.0 * monthly["churned"] / monthly["customers"], 2)
    monthly = monthly.sort_values("join_month").tail(periods).reset_index(drop=True)
    monthly["join_month"] = monthly["join_month"].astype(str)

    log.info("Analyzed churn trend across %d month(s)", len(monthly))
    return monthly


def detect_seasonal_patterns(
    df: pd.DataFrame = None, variance_threshold: float = 5.0
) -> dict:
    """Checks whether churn rate varies meaningfully by calendar month (Jan, Feb, ...,
    regardless of year), using all available implied-join-month history — not just the recent
    `periods` window analyze_churn_trend() uses — so there's a full year's worth of calendar
    months to compare. A pattern is flagged as seasonal if the range between the highest and
    lowest calendar-month churn rate exceeds variance_threshold percentage points."""
    if df is None:
        df = _load_customers()

    df = df.copy()
    df["calendar_month_num"] = df["implied_join_date"].dt.month
    df["calendar_month"] = df["implied_join_date"].dt.month_name()

    by_month = (
        df.groupby(["calendar_month_num", "calendar_month"])
        .agg(
            customers=("customer_id", "count"),
            churned=("churn", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
        .sort_values("calendar_month_num")
    )
    by_month["churn_rate"] = round(
        100.0 * by_month["churned"] / by_month["customers"], 2
    )

    highest = by_month.loc[by_month["churn_rate"].idxmax()]
    lowest = by_month.loc[by_month["churn_rate"].idxmin()]
    spread = round(float(highest["churn_rate"] - lowest["churn_rate"]), 2)

    result = {
        "by_calendar_month": by_month[["calendar_month", "churn_rate", "customers"]],
        "highest_month": highest["calendar_month"],
        "highest_churn_rate": float(highest["churn_rate"]),
        "lowest_month": lowest["calendar_month"],
        "lowest_churn_rate": float(lowest["churn_rate"]),
        "spread_pct_points": spread,
        "is_seasonal": spread > variance_threshold,
    }
    log.info(
        "Seasonal check: spread=%.2f pts (threshold=%.1f) -> is_seasonal=%s",
        spread,
        variance_threshold,
        result["is_seasonal"],
    )
    return result


def forecast_churn_rate(months: int = 3, trend_df: pd.DataFrame = None) -> pd.DataFrame:
    """Simple linear-regression forecast of churn_rate for the next `months` periods, fit on
    analyze_churn_trend()'s recent history. Not a sophisticated time-series model — a single
    straight-line extrapolation, appropriate for a "simple forecast" over a short window.
    """
    if trend_df is None:
        trend_df = analyze_churn_trend()

    if len(trend_df) < 2:
        raise ValueError("Need at least 2 historical periods to fit a forecast.")

    x = np.arange(len(trend_df))
    y = trend_df["churn_rate"].values
    slope, intercept = np.polyfit(x, y, deg=1)

    future_x = np.arange(len(trend_df), len(trend_df) + months)
    forecast_values = np.clip(slope * future_x + intercept, 0, 100)

    last_period = pd.Period(trend_df["join_month"].iloc[-1], freq="M")
    future_periods = [str(last_period + i) for i in range(1, months + 1)]

    forecast_df = pd.DataFrame(
        {
            "join_month": future_periods,
            "churn_rate": np.round(forecast_values, 2),
            "is_forecast": True,
        }
    )
    log.info(
        "Forecasted %d month(s) — slope=%.3f pts/month, next=%.2f%%",
        months,
        slope,
        forecast_values[0],
    )
    return forecast_df


def get_trend_insights(
    trend_df: pd.DataFrame = None,
    forecast_df: pd.DataFrame = None,
    df: pd.DataFrame = None,
) -> str:
    """Plain-English summary of the trend direction, magnitude, and forecast.

    `df` (raw customer rows with implied_join_date, as built by _load_customers()) is passed
    through to the internal seasonal check — needed so a caller with a pre-loaded/demo-mode df
    doesn't have this function silently fall back to a live DB query for that one piece.
    """
    if trend_df is None:
        trend_df = analyze_churn_trend(df=df)
    if forecast_df is None:
        forecast_df = forecast_churn_rate(trend_df=trend_df)

    first_rate = trend_df["churn_rate"].iloc[0]
    last_rate = trend_df["churn_rate"].iloc[-1]
    delta = round(last_rate - first_rate, 2)

    if abs(delta) < 1.0:
        direction = "has stayed roughly flat"
    elif delta > 0:
        direction = f"has risen by {delta:+.2f} percentage points"
    else:
        direction = f"has fallen by {delta:+.2f} percentage points"

    seasonal = detect_seasonal_patterns(df=df)
    seasonal_note = (
        f"Calendar-month churn rates vary by {seasonal['spread_pct_points']:.1f} points "
        f"between the highest ({seasonal['highest_month']}, {seasonal['highest_churn_rate']:.1f}%) "
        f"and lowest ({seasonal['lowest_month']}, {seasonal['lowest_churn_rate']:.1f}%) months"
        + (
            " — a meaningful seasonal pattern."
            if seasonal["is_seasonal"]
            else ", not a strong seasonal signal."
        )
    )

    insight = (
        f"Over the last {len(trend_df)} month(s), the churn rate {direction} "
        f"({first_rate:.1f}% -> {last_rate:.1f}%). "
        f"The next {len(forecast_df)} month(s) are forecast to be "
        f"{', '.join(f'{r:.1f}%' for r in forecast_df['churn_rate'])}. "
        f"{seasonal_note}"
    )
    log.info("Generated trend insight text")
    return insight


def save_trend_report(output_dir: str = OUTPUT_DIR) -> dict:
    """Runs the full trend analysis (trend, seasonal check, forecast, insights) and saves a
    CSV + markdown report + a matplotlib PNG (actual solid, forecast dashed) to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    trend_df = analyze_churn_trend()
    forecast_df = forecast_churn_rate(trend_df=trend_df)
    seasonal = detect_seasonal_patterns()
    insight = get_trend_insights(trend_df, forecast_df)

    combined = pd.concat(
        [trend_df.assign(is_forecast=False), forecast_df], ignore_index=True
    )
    csv_path = os.path.join(output_dir, "churn_trend.csv")
    combined.to_csv(csv_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trend_df["join_month"], trend_df["churn_rate"], marker="o", label="Actual")
    forecast_x = [trend_df["join_month"].iloc[-1]] + list(forecast_df["join_month"])
    forecast_y = [trend_df["churn_rate"].iloc[-1]] + list(forecast_df["churn_rate"])
    ax.plot(
        forecast_x,
        forecast_y,
        marker="o",
        linestyle="--",
        color="orange",
        label="Forecast",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate Trend + Forecast")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "churn_trend.png")
    plt.savefig(png_path)
    plt.close(fig)

    report_path = os.path.join(output_dir, "trend_report.md")
    lines = [
        "# Churn Trend Report",
        "",
        "## Trend",
        trend_df.to_string(index=False),
        "",
        "## Forecast",
        forecast_df.to_string(index=False),
        "",
        "## Seasonal Check",
        f"- Spread: {seasonal['spread_pct_points']} pts | Seasonal: {seasonal['is_seasonal']}",
        "",
        "## Insight",
        insight,
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved trend report to %s, %s, %s", csv_path, png_path, report_path)
    return {
        "csv_path": csv_path,
        "png_path": png_path,
        "report_path": report_path,
        "insight": insight,
    }


if __name__ == "__main__":
    trend_df = analyze_churn_trend()
    print("=== Churn Trend (last 6 months) ===")
    print(trend_df.to_string(index=False))

    forecast_df = forecast_churn_rate(trend_df=trend_df)
    print("\n=== Forecast (next 3 months) ===")
    print(forecast_df.to_string(index=False))

    seasonal = detect_seasonal_patterns()
    print("\n=== Seasonal Patterns ===")
    print(seasonal["by_calendar_month"].to_string(index=False))
    print(
        f"Is seasonal: {seasonal['is_seasonal']} (spread={seasonal['spread_pct_points']} pts)"
    )

    print(f"\n=== Insight ===\n{get_trend_insights(trend_df, forecast_df)}")

    result = save_trend_report()
    print(f"\nSaved trend report to {result['report_path']}")
