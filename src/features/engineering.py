"""Feature engineering — builds processed_customers from raw_customers."""
import pandas as pd
from src.utils.db import get_engine

SERVICE_COLS = [
    "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

CONTRACT_RISK = {"Month-to-month": 3, "One year": 2, "Two year": 1}
PAYMENT_RISK = {
    "Electronic check": 3, "Mailed check": 2,
    "Bank transfer (automatic)": 1, "Credit card (automatic)": 1,
}


def _tenure_group(tenure: int) -> str:
    if tenure <= 12:
        return "0-1 year"
    if tenure <= 24:
        return "1-2 years"
    if tenure <= 48:
        return "2-4 years"
    return "4+ years"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["customer_id"] = df["customerid"]
    out["tenure_group"] = df["tenure"].apply(_tenure_group)
    out["charge_per_month"] = df["monthly_charges"]
    out["services_count"] = df[SERVICE_COLS].apply(
        lambda row: sum(v not in ("No", "No phone service", "No internet service") for v in row),
        axis=1,
    )
    out["contract_risk_score"] = df["contract"].map(CONTRACT_RISK).fillna(2).astype(int)
    out["payment_risk_score"] = df["payment_method"].map(PAYMENT_RISK).fillna(2).astype(int)
    out["churn_label"] = (df["churn"].str.lower() == "yes").astype(int)
    out["churn_probability"] = None
    out["risk_segment"] = None
    return out


def run() -> None:
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw_customers", engine)
    processed = engineer(df)
    processed.to_sql("processed_customers", engine, if_exists="replace", index=False)
    print(f"Processed {len(processed)} rows into processed_customers.")


if __name__ == "__main__":
    run()
