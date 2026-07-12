"""Registers LightGBM/Ensemble results and flags the overall best model by AUC."""

from datetime import datetime

import pandas as pd
from sqlalchemy import text

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.utils.db import get_engine

NEW_MODELS = {
    "lightgbm": "v1",
    "ensemble": "v1",
}


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    evaluator = ModelEvaluator()
    engine = get_engine()

    with engine.begin() as conn:
        for model_name, version in NEW_MODELS.items():
            model = trainer.load_model(model_name)
            metrics = evaluator.evaluate_model(model, trainer.X_test, trainer.y_test)
            conn.execute(
                text(
                    "INSERT INTO model_registry "
                    "(model_name, model_version, accuracy, auc_score, f1_score, "
                    "trained_at, is_active) "
                    "VALUES (:name, :version, :accuracy, :auc, :f1, :trained_at, FALSE)"
                ),
                {
                    "name": model_name,
                    "version": version,
                    "accuracy": float(metrics["accuracy"]),
                    "auc": float(metrics["auc"]),
                    "f1": float(metrics["f1"]),
                    "trained_at": datetime.now(),
                },
            )

        registry = pd.read_sql("SELECT * FROM model_registry", conn)
        best_row = registry.loc[registry["auc_score"].idxmax()]
        best_id = int(best_row["id"])

        conn.execute(text("UPDATE model_registry SET is_active = FALSE"))
        conn.execute(
            text("UPDATE model_registry SET is_active = TRUE WHERE id = :best_id"),
            {"best_id": best_id},
        )

    print(f"Best model: {best_row['model_name']} with AUC: {best_row['auc_score']:.4f}")

    with open("models/best_model.txt", "w") as f:
        f.write(f"{best_row['model_name']}\n")
    print("Saved best model name to models/best_model.txt")


if __name__ == "__main__":
    run()
