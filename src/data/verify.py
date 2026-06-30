"""Data quality verification — prints key stats from raw_customers table."""
import pandas as pd
from src.utils.db import get_engine


def verify() -> None:
    engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM raw_customers", conn)

    print(f"\n{'='*50}")
    print("  raw_customers — Data Verification Report")
    print(f"{'='*50}")

    print(f"\n[1] Total rows: {len(df):,}")

    churn_counts = df["churn"].value_counts()
    churn_pct = df["churn"].value_counts(normalize=True).mul(100).round(1)
    print("\n[2] Churn distribution:")
    for label in ["Yes", "No"]:
        print(f"    {label}: {churn_counts.get(label, 0):>5,}  ({churn_pct.get(label, 0):.1f}%)")

    print("\n[3] Gender distribution:")
    for val, cnt in df["gender"].value_counts().items():
        print(f"    {val}: {cnt:,}")

    print("\n[4] Contract type distribution:")
    for val, cnt in df["contract"].value_counts().items():
        print(f"    {val}: {cnt:,}")

    print("\n[5] Avg monthly charges by churn status:")
    avg_charges = df.groupby("churn")["monthly_charges"].mean().round(2)
    for label, avg in avg_charges.items():
        print(f"    Churn={label}: ${avg:.2f}/month")

    print("\n[6] Null value counts:")
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if null_cols.empty:
        print("    No null values found.")
    else:
        for col, n in null_cols.items():
            print(f"    {col}: {n}")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    verify()
