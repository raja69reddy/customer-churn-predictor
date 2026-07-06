"""Registers the current best MLflow run as a Production model version."""
from mlflow.tracking import MlflowClient

from src.models import mlflow_registry, mlflow_setup
from src.models.train import ModelTrainer


def run() -> None:
    mlflow_setup.configure_tracking()
    client = MlflowClient()

    result = mlflow_registry.register_best_model()
    model_name = result.name
    version = result.version

    client.update_registered_model(
        name=model_name,
        description="Customer churn classifier for customer-churn-predictor. "
                     "Predicts churn probability from engineered customer features.",
    )
    client.update_model_version(
        name=model_name,
        version=version,
        description="Best-by-AUC run from the customer-churn-predictor MLflow experiment.",
    )
    client.set_model_version_tag(model_name, version, "project", "customer-churn-predictor")
    client.set_model_version_tag(model_name, version, "stage_reason", "highest AUC across all tracked runs")

    mlflow_registry.promote_to_production(model_name, version)

    details = client.get_model_version(model_name, version)
    print("\nRegistered model details:")
    print(f"    name: {details.name}")
    print(f"    version: {details.version}")
    print(f"    stage: {details.current_stage}")
    print(f"    run_id: {details.run_id}")
    print(f"    description: {details.description}")

    print("\nVerifying model loads from registry...")
    production_model = mlflow_registry.get_production_model(model_name)

    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)
    predictions = production_model.predict(trainer.X_test.head(5))
    print(f"Production model loaded successfully — sample predictions: {list(predictions)}")


if __name__ == "__main__":
    run()
