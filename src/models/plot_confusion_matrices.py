"""Plots confusion matrices for all 4 trained baseline models in a 2x2 grid."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
OUTPUT_PATH = "data/processed/confusion_matrices.png"


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    evaluator = ModelEvaluator()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for name, ax in zip(MODEL_NAMES, axes.flatten()):
        model = trainer.load_model(name)
        cm = evaluator.plot_confusion_matrix(model, trainer.X_test, trainer.y_test, name, ax=ax)
        tn, fp, fn, tp = cm.ravel()
        print(f"{name}: TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    fig.suptitle("Confusion Matrices - All Baseline Models", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"\nSaved confusion matrices plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
