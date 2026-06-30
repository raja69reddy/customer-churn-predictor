"""Data loading pipeline — reads raw CSV and writes to raw_customers table."""
import pandas as pd
from sqlalchemy import text
from src.utils.db import get_engine


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    return df


def ingest(csv_path: str) -> None:
    engine = get_engine()
    df = load_csv(csv_path)
    df.to_sql("raw_customers", engine, if_exists="replace", index=False)
    print(f"Ingested {len(df)} rows into raw_customers.")


if __name__ == "__main__":
    ingest("data/raw/telco_churn.csv")
