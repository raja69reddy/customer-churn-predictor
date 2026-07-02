"""Baseline model training — loads processed_customers and trains classifiers."""
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.db import get_engine

TARGET_COLUMN = "churn_label"
NON_FEATURE_COLUMNS = ["customer_id", "churn_label", "churn_probability", "risk_segment", "created_at"]


class ModelTrainer:
    """Loads processed features and trains baseline churn classifiers."""

    def __init__(self):
        self.df = None
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_processed_data(self) -> pd.DataFrame:
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM processed_customers", engine)
        df = pd.get_dummies(df, columns=["tenure_group"], prefix="tenure_group")

        self.df = df
        self.feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        return df

    def get_feature_columns(self) -> list:
        if self.feature_columns is None:
            raise ValueError("Call load_processed_data() before get_feature_columns().")
        return self.feature_columns

    def get_target_column(self) -> str:
        return TARGET_COLUMN

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        X = df[self.get_feature_columns()]
        y = df[self.get_target_column()]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_model(self, model, name: str) -> str:
        os.makedirs("models", exist_ok=True)
        path = os.path.join("models", f"{name}.pkl")
        joblib.dump(model, path)
        return path

    def load_model(self, name: str):
        path = os.path.join("models", f"{name}.pkl")
        return joblib.load(path)
