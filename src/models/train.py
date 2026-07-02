"""Baseline model training — loads processed_customers and trains classifiers."""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

    def _print_metrics(self, model_name: str, y_true, y_pred, y_proba) -> dict:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_proba),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }
        print(f"\n{model_name} metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")
        return metrics

    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        self._print_metrics("Logistic Regression", self.y_test, y_pred, y_proba)

        self.save_model(model, "logistic_regression")
        return model

    def train_decision_tree(self):
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        self._print_metrics("Decision Tree", self.y_test, y_pred, y_proba)

        self.save_model(model, "decision_tree")
        return model

    def train_random_forest(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        self._print_metrics("Random Forest", self.y_test, y_pred, y_proba)

        importance = pd.Series(model.feature_importances_, index=self.get_feature_columns())
        importance = importance.sort_values(ascending=False)
        top15 = importance.head(15)

        os.makedirs("data/processed", exist_ok=True)
        top15.rename("importance").rename_axis("feature").to_csv(
            "data/processed/rf_feature_importance.csv"
        )

        plt.figure(figsize=(8, max(4, len(top15) * 0.4)))
        plt.barh(top15.index[::-1], top15.values[::-1], color="steelblue")
        plt.title("Random Forest - Top 15 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("data/processed/rf_feature_importance.png")
        plt.close()

        self.save_model(model, "random_forest")
        return model

    def train_xgboost(self):
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        self._print_metrics("XGBoost", self.y_test, y_pred, y_proba)

        self.save_model(model, "xgboost")
        return model
