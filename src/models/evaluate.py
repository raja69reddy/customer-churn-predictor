"""Model evaluation — metrics, ROC curves, confusion matrices, and comparison reports."""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402


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

    def plot_roc_curve(
        self, model, X_test, y_test, model_name, ax=None, save_path=None
    ):
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

    def plot_confusion_matrix(
        self, model, X_test, y_test, model_name, ax=None, save_path=None
    ):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
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

        comparison = (
            pd.DataFrame(rows)
            .sort_values("auc", ascending=False)
            .reset_index(drop=True)
        )
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
            model,
            X,
            y,
            cv=skf,
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

    def plot_cv_results(
        self,
        cv_results: dict,
        model_name: str,
        save_path: str = "data/processed/cv_results.png",
    ):
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

    def explain_model(self, model, X_test, save_dir: str = "data/processed/shap_plots"):
        os.makedirs(save_dir, exist_ok=True)

        X_test = X_test.astype(float)
        try:
            explainer = shap.TreeExplainer(
                model, feature_perturbation="tree_path_dependent"
            )
        except Exception:
            explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)

        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "shap_summary_beeswarm.png"), bbox_inches="tight"
        )
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_bar.png"), bbox_inches="tight")
        plt.close()

        return shap_values

    def get_top_features(self, shap_values, n: int = 15) -> pd.Series:
        values = (
            shap_values.values
            if hasattr(shap_values, "values")
            else np.asarray(shap_values)
        )
        if values.ndim == 3:
            values = values[:, :, -1]

        feature_names = getattr(shap_values, "feature_names", None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(values.shape[1])]

        mean_abs_shap = np.abs(values).mean(axis=0)
        ranking = (
            pd.Series(mean_abs_shap, index=feature_names)
            .sort_values(ascending=False)
            .head(n)
        )

        print(f"Top {len(ranking)} features by mean |SHAP value|:")
        for rank, (feature, value) in enumerate(ranking.items(), start=1):
            print(f"    {rank}. {feature}: {value:.4f}")

        return ranking

    def plot_precision_recall_curve(
        self,
        model,
        X_test,
        y_test,
        model_name: str = "model",
        save_path: str = "data/processed/precision_recall_curve.png",
    ):
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = f1_scores[:-1].argmax()
        best_threshold = thresholds[best_idx]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, label=f"{model_name}")
        ax.scatter(
            recall[best_idx],
            precision[best_idx],
            color="red",
            zorder=5,
            label=f"Optimal threshold = {best_threshold:.2f} (F1 = {f1_scores[best_idx]:.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {model_name}")
        ax.legend(loc="lower left")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)

        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "best_threshold": best_threshold,
        }
