"""EDA summary — prints and saves headline churn statistics from raw_customers."""

import os

import pandas as pd

from src.utils.db import get_engine

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "eda_summary.txt"
)

TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "12-24", "24-48", "48+"]


def _load() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM raw_customers", engine)


def _churn_rate(df: pd.DataFrame) -> float:
    return (df["churn"] == "Yes").mean() * 100


def _top_correlated_features(df: pd.DataFrame, n: int = 5) -> pd.Series:
    churn_binary = (df["churn"] == "Yes").astype(int)

    numeric_cols = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
    categorical_cols = [
        "gender",
        "partner",
        "dependents",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "paperless_billing",
        "payment_method",
    ]

    numeric_df = df[numeric_cols].copy()
    dummies = pd.get_dummies(df[categorical_cols], drop_first=False)
    features = pd.concat([numeric_df, dummies], axis=1)

    corr = features.corrwith(churn_binary).dropna()
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


def _contract_breakdown(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("contract")["churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )


def _tenure_group_breakdown(df: pd.DataFrame) -> pd.Series:
    tenure_group = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    )
    return df.groupby(tenure_group, observed=True)["churn"].apply(
        lambda s: (s == "Yes").mean() * 100
    )


def _monthly_charges_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("churn")["monthly_charges"]
        .describe()[["mean", "std", "min", "max"]]
        .round(2)
    )


def generate_summary() -> str:
    df = _load()

    lines = []
    lines.append("=" * 55)
    lines.append("  EDA Summary Report — raw_customers")
    lines.append("=" * 55)

    lines.append(f"\n[1] Overall churn rate: {_churn_rate(df):.2f}%")

    lines.append("\n[2] Top 5 features correlated with churn:")
    for feature, corr in _top_correlated_features(df).items():
        lines.append(f"    {feature:<35} {corr:+.4f}")

    lines.append("\n[3] Churn rate by contract type:")
    for segment, rate in _contract_breakdown(df).items():
        lines.append(f"    {segment:<20} {rate:.2f}%")

    lines.append("\n[4] Churn rate by tenure group:")
    for segment, rate in _tenure_group_breakdown(df).items():
        lines.append(f"    {segment:<10} {rate:.2f}%")

    lines.append("\n[5] Monthly charges statistics by churn:")
    stats = _monthly_charges_stats(df)
    for churn_label, row in stats.iterrows():
        lines.append(
            f"    Churn={churn_label:<4} mean=${row['mean']:.2f}  std=${row['std']:.2f}  "
            f"min=${row['min']:.2f}  max=${row['max']:.2f}"
        )

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


def run() -> None:
    summary = generate_summary()
    print(summary)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {os.path.normpath(OUTPUT_PATH)}")


if __name__ == "__main__":
    run()
