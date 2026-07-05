"""Compares LightGBM against XGBoost (tuned) on the same test set."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    xgboost_model = trainer.load_model("xgboost_tuned")
    lightgbm_model = trainer.load_model("lightgbm")

    evaluator = ModelEvaluator()
    xgb_metrics = evaluator.evaluate_model(xgboost_model, trainer.X_test, trainer.y_test)
    lgbm_metrics = evaluator.evaluate_model(lightgbm_model, trainer.X_test, trainer.y_test)

    comparison = pd.DataFrame({
        "Metric": ["accuracy", "auc", "f1", "precision", "recall"],
        "XGBoost": [xgb_metrics[m] for m in ["accuracy", "auc", "f1", "precision", "recall"]],
        "LightGBM": [lgbm_metrics[m] for m in ["accuracy", "auc", "f1", "precision", "recall"]],
    })

    print("Metric      | XGBoost  | LightGBM")
    for _, row in comparison.iterrows():
        print(f"{row['Metric']:<12}| {row['XGBoost']:.4f}   | {row['LightGBM']:.4f}")

    comparison.to_csv("data/processed/xgb_vs_lgbm.csv", index=False)
    print("\nSaved comparison to data/processed/xgb_vs_lgbm.csv")

    fig, ax = plt.subplots(figsize=(7, 7))
    evaluator.plot_roc_curve(xgboost_model, trainer.X_test, trainer.y_test, "XGBoost (tuned)", ax=ax)
    evaluator.plot_roc_curve(lightgbm_model, trainer.X_test, trainer.y_test, "LightGBM", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - XGBoost vs LightGBM")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("data/processed/xgb_vs_lgbm_roc.png")
    plt.close(fig)
    print("Saved ROC comparison plot to data/processed/xgb_vs_lgbm_roc.png")


if __name__ == "__main__":
    run()
