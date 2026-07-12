"""Runs SHAP analysis on the current best model (per model_registry.is_active)."""

import pandas as pd

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.utils.db import get_engine


def run() -> None:
    engine = get_engine()
    registry = pd.read_sql(
        "SELECT * FROM model_registry WHERE is_active = TRUE", engine
    )
    if registry.empty:
        raise RuntimeError("No active model found in model_registry.")
    best_model_name = registry.iloc[0]["model_name"]
    print(f"Best model (is_active=True): {best_model_name}")

    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)
    model = trainer.load_model(best_model_name)

    evaluator = ModelEvaluator()
    print("\nCalculating SHAP values on test set...")
    shap_values = evaluator.explain_model(model, trainer.X_test)
    print("Saved SHAP summary and bar plots to data/processed/shap_plots/")

    print("\nTop 15 features by SHAP importance:")
    evaluator.get_top_features(shap_values, n=15)


if __name__ == "__main__":
    run()
