"""Batch-scores every customer in raw_customers using the MLflow Production model."""
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from src.models import mlflow_registry, mlflow_setup
from src.models.predict import ChurnPredictor
from src.utils.db import get_engine


def load_production_predictor() -> tuple[ChurnPredictor, str]:
    mlflow_setup.configure_tracking()
    client = MlflowClient()
    model_name = mlflow_registry.REGISTERED_MODEL_NAME

    versions = client.search_model_versions(f"name='{model_name}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        raise RuntimeError(f"No Production version found for '{model_name}' in MLflow registry.")
    version = production[0].version

    predictor = ChurnPredictor()
    predictor.model = mlflow_registry.get_production_model(model_name)
    predictor.model_name = f"{model_name}_v{version}"
    return predictor, version


def run() -> None:
    predictor, version = load_production_predictor()
    print(f"Using model version {version} from MLflow registry")

    engine = get_engine()
    customers = pd.read_sql("SELECT * FROM raw_customers", engine)
    print(f"Loaded {len(customers)} customers from raw_customers")

    scored = predictor.predict_batch(customers)
    print(f"Scored {len(scored)} customers")

    distribution = scored["risk_segment"].value_counts()
    high = int(distribution.get("High", 0))
    medium = int(distribution.get("Medium", 0))
    low = int(distribution.get("Low", 0))
    print(f"\nSummary: {high} high risk, {medium} medium risk, {low} low risk")

    top10 = scored.sort_values("churn_probability", ascending=False).head(10)
    print("\nGenerating GPT explanations for top 10 highest-risk customers...")
    for _, row in top10.iterrows():
        customer_dict = row.to_dict()
        prediction = {"churn_probability": row["churn_probability"], "risk_segment": row["risk_segment"]}
        explanation = predictor.explain_with_gpt(customer_dict, prediction)
        print(f"    {row['customer_id']} (p={row['churn_probability']:.3f}, {row['risk_segment']}): {explanation}")

    to_save = scored[["customer_id", "churn_probability", "risk_segment"]].copy()
    to_save["model_version"] = predictor.model_name

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE churn_predictions"))
        to_save.to_sql("churn_predictions", conn, if_exists="append", index=False)

    print(f"\nSaved {len(to_save)} predictions to churn_predictions table")

    mlflow_setup.setup_experiment()
    with mlflow.start_run(run_name="batch_scoring"):
        mlflow.log_param("model_version", predictor.model_name)
        mlflow.log_metric("customers_scored", len(scored))
        mlflow.log_metric("high_risk_count", high)
        mlflow.log_metric("medium_risk_count", medium)
        mlflow.log_metric("low_risk_count", low)
        mlflow.set_tag("run_type", "batch_scoring")
    print("Logged batch scoring run to MLflow")


if __name__ == "__main__":
    run()
