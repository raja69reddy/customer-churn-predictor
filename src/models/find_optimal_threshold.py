"""Finds the classification threshold (0.1-0.9) that maximizes F1 score for the best model."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.train import ModelTrainer
from src.utils.db import get_engine

OUTPUT_PATH = "models/optimal_threshold.txt"


def run() -> None:
    engine = get_engine()
    registry = pd.read_sql("SELECT * FROM model_registry WHERE is_active = TRUE", engine)
    if registry.empty:
        raise RuntimeError("No active model found in model_registry.")
    best_model_name = registry.iloc[0]["model_name"]
    print(f"Best model (is_active=True): {best_model_name}")

    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)
    model = trainer.load_model(best_model_name)

    y_proba = model.predict_proba(trainer.X_test)[:, 1]

    thresholds = np.arange(0.1, 0.91, 0.1)
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(trainer.y_test, y_pred)
        results.append((round(threshold, 1), f1))
        print(f"    threshold={threshold:.1f}  F1={f1:.4f}")

    best_threshold, best_f1 = max(results, key=lambda r: r[1])
    print(f"\nOptimal threshold: {best_threshold} (F1 = {best_f1:.4f})")

    with open(OUTPUT_PATH, "w") as f:
        f.write(f"{best_threshold}\n")
    print(f"Saved optimal threshold to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
