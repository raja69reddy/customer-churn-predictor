"""Data ingestion pipeline — loads customers.csv into raw_customers table."""

import argparse
import logging
import time

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
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
    "monthly_charges",
    "total_charges",
    "churn",
]


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        log.error("CSV file not found: %s", path)
        raise
    except Exception as e:
        log.error("Failed to read CSV: %s", e)
        raise
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # strip whitespace from string columns
    str_cols = df.select_dtypes(include="str").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # convert total_charges to float (empty strings → NaN → 0)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(
        0.0
    )

    # validate uniqueness
    dupes = df["customer_id"].duplicated().sum()
    if dupes:
        log.warning("Found %d duplicate customer_id rows — dropping duplicates", dupes)
        df = df.drop_duplicates(subset="customer_id")

    # validate churn values
    invalid_churn = df[~df["churn"].isin(["Yes", "No"])]
    if not invalid_churn.empty:
        log.warning("Found %d rows with invalid churn values", len(invalid_churn))

    # validate numeric ranges
    bad_charges = (df["monthly_charges"] <= 0).sum()
    if bad_charges:
        log.warning("%d rows have non-positive monthly_charges", bad_charges)

    bad_tenure = (~df["tenure"].between(0, 72)).sum()
    if bad_tenure:
        log.warning("%d rows have tenure outside 0-72", bad_tenure)

    return df


def ingest(csv_path: str, mode: str = "full") -> int:
    log.info("Starting ingestion — mode=%s, file=%s", mode, csv_path)
    start = time.time()

    df = load_csv(csv_path)
    log.info("Loaded %d rows from CSV", len(df))

    df = validate(df)
    log.info("Validation complete — %d rows ready", len(df))

    engine = get_engine()
    try:
        with engine.begin() as conn:
            if mode == "full":
                conn.execute(text("TRUNCATE TABLE raw_customers"))
                log.info("Truncated raw_customers table")
                df.to_sql("raw_customers", conn, if_exists="append", index=False)
                inserted = len(df)
            elif mode == "incremental":
                existing = pd.read_sql("SELECT customer_id FROM raw_customers", conn)
                new_rows = df[~df["customer_id"].isin(existing["customer_id"])]
                if new_rows.empty:
                    log.info("No new rows to insert")
                    return 0
                new_rows.to_sql("raw_customers", conn, if_exists="append", index=False)
                inserted = len(new_rows)
            else:
                raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        log.error("DB insertion failed: %s", e)
        raise

    elapsed = time.time() - start
    log.info("Inserted %d rows into raw_customers in %.2fs", inserted, elapsed)
    print(f"Inserted {inserted} rows into raw_customers")
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest customer CSV into PostgreSQL")
    parser.add_argument("--path", default="data/raw/customers.csv")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full")
    args = parser.parse_args()
    ingest(args.path, args.mode)
