"""Plots ROC curves for all 4 trained baseline models on a single chart."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models.evaluate import ModelEvaluator  # noqa: E402
from src.models.train import ModelTrainer  # noqa: E402

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
OUTPUT_PATH = "data/processed/roc_curves.png"


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    evaluator = ModelEvaluator()
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, color in zip(MODEL_NAMES, COLORS):
        model = trainer.load_model(name)
        auc_score = evaluator.plot_roc_curve(
            model, trainer.X_test, trainer.y_test, name, ax=ax
        )
        ax.lines[-1].set_color(color)
        print(f"{name}: AUC = {auc_score:.4f}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Baseline Models")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"\nSaved ROC curves plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
