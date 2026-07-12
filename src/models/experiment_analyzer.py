"""Analyzes MLflow experiment runs — best run lookup, history, comparison plots, and feature drift."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.models import mlflow_setup


def _all_runs() -> pd.DataFrame:
    mlflow_setup.configure_tracking()
    experiment = mlflow.get_experiment_by_name(mlflow_setup.EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{mlflow_setup.EXPERIMENT_NAME}' not found."
        )
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id])


def get_best_run() -> pd.Series:
    runs = _all_runs()
    if runs.empty:
        raise RuntimeError("No MLflow runs found.")
    return runs.loc[runs["metrics.auc"].idxmax()]


def get_run_history() -> pd.DataFrame:
    runs = _all_runs()
    return runs.sort_values("metrics.auc", ascending=False).reset_index(drop=True)


def plot_metric_comparison(
    metric: str = "auc", save_path: str = "data/processed/mlflow_metric_comparison.png"
) -> str:
    history = get_run_history()
    metric_col = f"metrics.{metric}"
    if metric_col not in history.columns:
        raise ValueError(f"Metric '{metric}' not found in run history.")

    labels = history["tags.mlflow.runName"].fillna(history["run_id"])
    values = history[metric_col]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, color="steelblue")
    plt.ylabel(metric.upper())
    plt.title(f"MLflow Run Comparison - {metric.upper()}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return save_path


def export_run_summary(path: str = "data/processed/mlflow_summary.csv") -> str:
    history = get_run_history()
    keep_cols = ["run_id"] + [
        c
        for c in history.columns
        if c.startswith("tags.mlflow.runName")
        or c.startswith("metrics.")
        or c.startswith("params.")
    ]
    summary = history[keep_cols]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary.to_csv(path, index=False)
    return path


def get_feature_drift(run_ids: list) -> pd.DataFrame:
    mlflow_setup.configure_tracking()

    from src.models.train import ModelTrainer

    trainer = ModelTrainer()
    trainer.load_processed_data()
    feature_names = trainer.get_feature_columns()

    importances = {}
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        # MLflow 3.x logs models as first-class "Logged Model" entities rather than plain
        # run-relative artifacts, so runs:/{run_id}/model no longer resolves. Resolving
        # mlflow.sklearn.load_model() directly against the bare models:/{model_id} URI also
        # fails on this local file store (empty artifact_path bug), so download the artifacts
        # to a local path first and load from there instead.
        model_id = run.outputs.model_outputs[0].model_id
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_id}"
        )
        model = mlflow.sklearn.load_model(local_path)
        name = run.data.tags.get("mlflow.runName", run_id)

        if hasattr(model, "feature_importances_"):
            importances[name] = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances[name] = np.abs(model.coef_[0])
        elif hasattr(model, "estimators_"):
            sub_importances = [
                est.feature_importances_
                for est in model.estimators_
                if hasattr(est, "feature_importances_")
            ]
            if sub_importances:
                importances[name] = np.mean(sub_importances, axis=0)

    if not importances:
        raise RuntimeError(
            "None of the given runs have inspectable feature importances."
        )

    drift_df = pd.DataFrame(importances, index=feature_names)
    print("Feature importance across runs/versions:")
    print(drift_df.to_string())
    return drift_df


if __name__ == "__main__":
    best = get_best_run()
    print(
        f"Best run: {best.get('tags.mlflow.runName')} (AUC={best.get('metrics.auc'):.4f})"
    )
