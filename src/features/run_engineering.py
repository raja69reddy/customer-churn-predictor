"""Runs the full feature engineering pipeline against PostgreSQL raw_customers."""

from sqlalchemy import text

from src.features.engineering import FeatureEngineer
from src.utils.db import get_engine

PROCESSED_COLUMNS = [
    "customer_id",
    "tenure_group",
    "charge_per_month",
    "services_count",
    "contract_risk_score",
    "payment_risk_score",
    "churn_label",
]


def run() -> None:
    fe = FeatureEngineer()

    raw_df = fe.load_raw_data()
    print(f"Raw data shape: {raw_df.shape}")
    print(f"Raw feature names ({len(raw_df.columns)}): {list(raw_df.columns)}")

    df = fe.clean_data(raw_df)
    df = fe.create_tenure_group(df)
    df = fe.create_charge_per_month(df)
    df = fe.create_services_count(df)
    df = fe.create_contract_risk_score(df)
    df = fe.create_payment_risk_score(df)
    df = fe.create_churn_label(df)

    new_features = [c for c in df.columns if c not in raw_df.columns]

    print(f"\nEngineered data shape: {df.shape}")
    print(f"Engineered feature names ({len(df.columns)}): {list(df.columns)}")

    fe.fit_transform(df)
    fe.save_pipeline("models/preprocessor.pkl")
    print("\nSaved preprocessing pipeline to models/preprocessor.pkl")

    processed = df[PROCESSED_COLUMNS].copy()
    processed["tenure_group"] = processed["tenure_group"].astype(str)
    processed["churn_probability"] = None
    processed["risk_segment"] = None

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE processed_customers"))
        processed.to_sql("processed_customers", conn, if_exists="append", index=False)

    print(f"\nSaved {len(processed)} rows to processed_customers table")
    print(f"\nFeature engineering complete: {len(new_features)} features created")


if __name__ == "__main__":
    run()
