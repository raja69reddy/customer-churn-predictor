"""Retrains all 6 leaderboard models with full MLflow tracking."""

from src.models.train import ModelTrainer


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    run_ids = {}

    print("=== Logistic Regression ===")
    trainer.train_logistic_regression()

    print("\n=== Decision Tree ===")
    trainer.train_decision_tree()

    print("\n=== Random Forest (tuned) ===")
    trainer.tune_random_forest()

    print("\n=== XGBoost (tuned) ===")
    trainer.tune_xgboost()

    print("\n=== LightGBM ===")
    trainer.train_lightgbm()

    print("\n=== Ensemble ===")
    trainer.train_ensemble()

    import mlflow
    from src.models import mlflow_setup

    mlflow_setup.configure_tracking()
    runs = mlflow.search_runs(
        experiment_names=[mlflow_setup.EXPERIMENT_NAME],
        order_by=["start_time DESC"],
        max_results=6,
    )

    print("\nMLflow run IDs for all 6 retrained models:")
    for _, row in runs.iterrows():
        print(f"    {row['tags.mlflow.runName']:<25} run_id={row['run_id']}")


if __name__ == "__main__":
    run()
