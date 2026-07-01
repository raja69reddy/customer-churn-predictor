"""Feature engineering — builds engineered features from raw_customers for modeling."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.db import get_engine

NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges"]

TENURE_BINS = [0, 12, 24, 48, np.inf]
TENURE_LABELS = ["New Customer", "Growing Customer", "Established Customer", "Loyal Customer"]

CONTRACT_RISK = {"Month-to-month": 3, "One year": 2, "Two year": 1}
PAYMENT_RISK = {
    "Electronic check": 3,
    "Mailed check": 2,
    "Bank transfer (automatic)": 1,
    "Credit card (automatic)": 1,
}

CATEGORICAL_COLS = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method",
]

SERVICE_COLS = [
    "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]


class FeatureEngineer:
    """Builds the processed_customers feature set from raw_customers."""

    def __init__(self):
        self.encoder = None
        self.scaler = None

    def load_raw_data(self) -> pd.DataFrame:
        engine = get_engine()
        return pd.read_sql("SELECT * FROM raw_customers", engine)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())

        df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("Unknown")

        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = self.encoder.fit_transform(df[CATEGORICAL_COLS])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(CATEGORICAL_COLS),
            index=df.index,
        )

        df = df.drop(columns=CATEGORICAL_COLS)
        return pd.concat([df, encoded_df], axis=1)

    def scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        self.scaler = StandardScaler()
        df[NUMERIC_COLS] = self.scaler.fit_transform(df[NUMERIC_COLS])

        return df

    def create_tenure_group(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
        )
        return df

    def create_charge_per_month(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        safe_tenure = df["tenure"].replace(0, 1)
        df["charge_per_month"] = df["total_charges"] / safe_tenure
        return df

    def create_services_count(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["services_count"] = (
            (df["phone_service"] == "Yes").astype(int)
            + (df["internet_service"] != "No").astype(int)
            + (df["online_security"] == "Yes").astype(int)
            + (df["online_backup"] == "Yes").astype(int)
            + (df["device_protection"] == "Yes").astype(int)
            + (df["tech_support"] == "Yes").astype(int)
            + (df["streaming_tv"] == "Yes").astype(int)
            + (df["streaming_movies"] == "Yes").astype(int)
        )
        return df

    def create_contract_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["contract_risk_score"] = df["contract"].map(CONTRACT_RISK).astype(int)
        return df

    def create_payment_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["payment_risk_score"] = df["payment_method"].map(PAYMENT_RISK).astype(int)
        return df

    def create_churn_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["churn_label"] = (df["churn"] == "Yes").astype(int)
        return df
