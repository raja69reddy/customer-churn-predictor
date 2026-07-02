"""Runs the full baseline model training + evaluation pipeline."""
from datetime import datetime

from sqlalchemy import text

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.utils.db import get_engine

MODEL_VERSION = "v1"


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)
    print(f"Loaded processed data: {df.shape[0]} rows, {len(trainer.get_feature_columns())} features")

    print("\nTraining baseline models...")
    models = {
        "logistic_regression": trainer.train_logistic_regression(),
        "decision_tree": trainer.train_decision_tree(),
        "random_forest": trainer.train_random_forest(),
        "xgboost": trainer.train_xgboost(),
    }

    evaluator = ModelEvaluator()
    print("\nModel Comparison:")
    comparison = evaluator.compare_models(models, trainer.X_test, trainer.y_test)
    comparison = comparison.rename(columns={
        "model": "Model", "accuracy": "Accuracy", "auc": "AUC",
        "f1": "F1", "precision": "Precision", "recall": "Recall",
    })
    print(comparison.to_string(index=False))

    best_row = comparison.iloc[0]
    print(f"\nBest model: {best_row['Model']} (AUC = {best_row['AUC']:.4f})")

    evaluator.save_evaluation_report(comparison, "data/processed/model_comparison.csv")
    print("Saved comparison to data/processed/model_comparison.csv")

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE model_registry SET is_active = FALSE"))
        for _, row in comparison.iterrows():
            is_active = row["Model"] == best_row["Model"]
            conn.execute(
                text(
                    "INSERT INTO model_registry "
                    "(model_name, model_version, accuracy, auc_score, f1_score, trained_at, is_active) "
                    "VALUES (:name, :version, :accuracy, :auc, :f1, :trained_at, :is_active)"
                ),
                {
                    "name": row["Model"],
                    "version": MODEL_VERSION,
                    "accuracy": float(row["Accuracy"]),
                    "auc": float(row["AUC"]),
                    "f1": float(row["F1"]),
                    "trained_at": datetime.now(),
                    "is_active": bool(is_active),
                },
            )
    print("Logged all results to model_registry table")


if __name__ == "__main__":
    run()
