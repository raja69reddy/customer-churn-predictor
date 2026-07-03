"""Model evaluation — metrics, ROC curves, confusion matrices, and comparison reports."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


class ModelEvaluator:
    """Computes metrics and diagnostic plots for trained churn classifiers."""

    def evaluate_model(self, model, X_test, y_test) -> dict:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

    def plot_roc_curve(self, model, X_test, y_test, model_name, ax=None, save_path=None):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")

        if standalone:
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - {model_name}")
            ax.legend(loc="lower right")
            plt.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
            plt.close(fig)

        return auc_score

    def plot_confusion_matrix(self, model, X_test, y_test, model_name, ax=None, save_path=None):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")

        if standalone:
            plt.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
            plt.close(fig)

        return cm

    def compare_models(self, models_dict: dict, X_test, y_test) -> pd.DataFrame:
        rows = []
        for name, model in models_dict.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            rows.append({"model": name, **metrics})

        comparison = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
        print(comparison.to_string(index=False))
        return comparison

    def save_evaluation_report(self, results, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if isinstance(results, pd.DataFrame):
            results.to_csv(filename, index=False)
        else:
            pd.DataFrame(results).to_csv(filename, index=False)

    def cross_validate_model(self, model, X, y, cv: int = 5) -> dict:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_validate(
            model, X, y, cv=skf,
            scoring=["accuracy", "roc_auc", "f1"],
        )

        cv_results = {
            "accuracy": scores["test_accuracy"],
            "auc": scores["test_roc_auc"],
            "f1": scores["test_f1"],
        }

        print(f"Cross validation results ({cv}-fold stratified):")
        for metric, values in cv_results.items():
            print(f"    {metric}: mean={values.mean():.4f}  std={values.std():.4f}")

        return cv_results

    def plot_cv_results(self, cv_results: dict, model_name: str, save_path: str = "data/processed/cv_results.png"):
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = list(cv_results.keys())
        data = [cv_results[label] for label in labels]

        ax.boxplot(data, tick_labels=labels)
        ax.set_title(f"Cross Validation Scores - {model_name}")
        ax.set_ylabel("Score")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        return save_path
