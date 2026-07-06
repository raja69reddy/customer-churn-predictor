"""MLflow Model Registry operations — register, promote, and compare churn model runs."""
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.models import mlflow_setup

REGISTERED_MODEL_NAME = "churn-predictor"


def register_best_model(registered_name: str = REGISTERED_MODEL_NAME):
    mlflow_setup.configure_tracking()

    experiment = mlflow.get_experiment_by_name(mlflow_setup.EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{mlflow_setup.EXPERIMENT_NAME}' not found.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.auc DESC"])
    if runs.empty:
        raise RuntimeError("No MLflow runs found — run training with MLflow tracking first.")

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, registered_name)
    print(
        f"Registered '{best_run['tags.mlflow.runName']}' (run_id={run_id}, "
        f"AUC={best_run['metrics.auc']:.4f}) as '{registered_name}' version {result.version}"
    )
    return result


def promote_to_production(model_name: str = REGISTERED_MODEL_NAME, version: str = None) -> str:
    mlflow_setup.configure_tracking()
    client = MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"No registered versions found for '{model_name}'.")
        version = max(versions, key=lambda v: int(v.version)).version

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production", archive_existing_versions=True,
    )
    print(f"Promoted {model_name} version {version} to Production")
    return version


def get_production_model(model_name: str = REGISTERED_MODEL_NAME):
    """Loads the Production model version with its native sklearn/xgboost/lightgbm interface
    intact (predict_proba etc.), rather than the generic mlflow.pyfunc wrapper."""
    mlflow_setup.configure_tracking()
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    return model


def list_model_versions(model_name: str = REGISTERED_MODEL_NAME):
    mlflow_setup.configure_tracking()
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    versions = sorted(versions, key=lambda v: int(v.version))

    print(f"Registered versions for '{model_name}':")
    for v in versions:
        print(f"    version={v.version}  stage={v.current_stage}  run_id={v.run_id}")
    return versions


def compare_runs(run_ids: list) -> pd.DataFrame:
    mlflow_setup.configure_tracking()
    client = MlflowClient()

    rows = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        row = {"run_id": run_id, "run_name": run.data.tags.get("mlflow.runName", "")}
        row.update(run.data.metrics)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    list_model_versions()
