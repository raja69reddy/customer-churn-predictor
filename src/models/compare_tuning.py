"""Compares tuned models against their baselines and updates model_registry."""

from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.utils.db import get_engine

PAIRS = [
    ("random_forest", "random_forest_tuned"),
    ("xgboost", "xgboost_tuned"),
]


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    evaluator = ModelEvaluator()
    rows = []
    tuned_metrics = {}

    for baseline_name, tuned_name in PAIRS:
        baseline_model = trainer.load_model(baseline_name)
        tuned_model = trainer.load_model(tuned_name)

        baseline_metrics = evaluator.evaluate_model(
            baseline_model, trainer.X_test, trainer.y_test
        )
        tuned = evaluator.evaluate_model(tuned_model, trainer.X_test, trainer.y_test)
        tuned_metrics[tuned_name] = tuned

        rows.append(
            {
                "model": baseline_name,
                "baseline_auc": baseline_metrics["auc"],
                "tuned_auc": tuned["auc"],
                "improvement": tuned["auc"] - baseline_metrics["auc"],
            }
        )

    comparison = pd.DataFrame(rows)
    print("Model | Baseline AUC | Tuned AUC | Improvement")
    for _, row in comparison.iterrows():
        print(
            f"{row['model']:<15} {row['baseline_auc']:.4f}        {row['tuned_auc']:.4f}      {row['improvement']:+.4f}"
        )

    comparison.to_csv("data/processed/tuning_comparison.csv", index=False)
    print("\nSaved comparison to data/processed/tuning_comparison.csv")

    x = np.arange(len(comparison))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, comparison["baseline_auc"], width, label="Baseline")
    ax.bar(x + width / 2, comparison["tuned_auc"], width, label="Tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["model"])
    ax.set_ylabel("AUC")
    ax.set_title("Baseline vs Tuned Model AUC")
    ax.set_ylim(0.85, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/processed/tuning_comparison.png")
    plt.close(fig)
    print("Saved bar chart to data/processed/tuning_comparison.png")

    engine = get_engine()
    with engine.begin() as conn:
        registry = pd.read_sql("SELECT * FROM model_registry", conn)

        for tuned_name, metrics in tuned_metrics.items():
            conn.execute(
                text(
                    "INSERT INTO model_registry "
                    "(model_name, model_version, accuracy, auc_score, f1_score, trained_at, is_active) "
                    "VALUES (:name, :version, :accuracy, :auc, :f1, :trained_at, FALSE)"
                ),
                {
                    "name": tuned_name,
                    "version": "v2",
                    "accuracy": float(metrics["accuracy"]),
                    "auc": float(metrics["auc"]),
                    "f1": float(metrics["f1"]),
                    "trained_at": datetime.now(),
                },
            )

        full_registry = pd.read_sql("SELECT * FROM model_registry", conn)
        best_id = int(full_registry.loc[full_registry["auc_score"].idxmax(), "id"])
        conn.execute(text("UPDATE model_registry SET is_active = FALSE"))
        conn.execute(
            text("UPDATE model_registry SET is_active = TRUE WHERE id = :best_id"),
            {"best_id": best_id},
        )

    final = pd.read_sql(
        "SELECT model_name, model_version, accuracy, auc_score, f1_score, is_active "
        "FROM model_registry ORDER BY auc_score DESC",
        engine,
    )
    print("\nUpdated model_registry:")
    print(final.to_string(index=False))


if __name__ == "__main__":
    run()
