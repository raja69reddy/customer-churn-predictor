"""Builds the final leaderboard across all 6 models, sorted by AUC."""
from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest (tuned)": "random_forest_tuned",
    "XGBoost (tuned)": "xgboost_tuned",
    "LightGBM": "lightgbm",
    "Ensemble": "ensemble",
}


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    models = {label: trainer.load_model(name) for label, name in MODELS.items()}

    evaluator = ModelEvaluator()
    print("Final Model Leaderboard:")
    leaderboard = evaluator.compare_models(models, trainer.X_test, trainer.y_test)
    leaderboard = leaderboard.rename(columns={"model": "Model", "accuracy": "Accuracy", "auc": "AUC", "f1": "F1", "precision": "Precision", "recall": "Recall"})

    leaderboard.to_csv("data/processed/final_model_leaderboard.csv", index=False)
    print("\nSaved leaderboard to data/processed/final_model_leaderboard.csv")


if __name__ == "__main__":
    run()
