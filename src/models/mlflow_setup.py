"""MLflow experiment tracking setup — shared tracking URI/experiment config for all scripts."""
import os

import mlflow

EXPERIMENT_NAME = "customer-churn-predictor"

_TRACKING_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "mlflow"))
_DB_PATH = os.path.join(_TRACKING_DIR, "mlflow.db").replace("\\", "/")
_ARTIFACT_PATH = os.path.join(_TRACKING_DIR, "mlruns").replace("\\", "/")

TRACKING_URI = f"sqlite:///{_DB_PATH}"
ARTIFACT_LOCATION = f"file:///{_ARTIFACT_PATH}"


def configure_tracking() -> None:
    os.makedirs(_TRACKING_DIR, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)


def setup_experiment(name: str = EXPERIMENT_NAME) -> str:
    configure_tracking()

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, artifact_location=ARTIFACT_LOCATION)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(name)
    return experiment_id


def get_experiment_id(name: str = EXPERIMENT_NAME) -> str:
    configure_tracking()

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return setup_experiment(name)
    return experiment.experiment_id


if __name__ == "__main__":
    experiment_id = setup_experiment()
    print(f"MLflow experiment '{EXPERIMENT_NAME}' ready (id={experiment_id})")
    print(f"Tracking URI: {TRACKING_URI}")
