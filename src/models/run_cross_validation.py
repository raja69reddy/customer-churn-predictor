"""Runs 5-fold stratified cross validation for all 4 baseline models."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    ),
}


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    evaluator = ModelEvaluator()
    rows = []

    for name, model in MODELS.items():
        print(f"\n=== {name} ===")
        cv_results = evaluator.cross_validate_model(
            model, trainer.X_train, trainer.y_train, cv=5
        )
        rows.append(
            {
                "model": name,
                "accuracy_mean": cv_results["accuracy"].mean(),
                "accuracy_std": cv_results["accuracy"].std(),
                "auc_mean": cv_results["auc"].mean(),
                "auc_std": cv_results["auc"].std(),
                "f1_mean": cv_results["f1"].mean(),
                "f1_std": cv_results["f1"].std(),
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values("auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    print("\nCross Validation Comparison (mean +/- std):")
    for _, row in comparison.iterrows():
        print(
            f"    {row['model']:<20} "
            f"accuracy={row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.4f}  "
            f"auc={row['auc_mean']:.4f}+/-{row['auc_std']:.4f}  "
            f"f1={row['f1_mean']:.4f}+/-{row['f1_std']:.4f}"
        )

    comparison.to_csv("data/processed/cv_comparison.csv", index=False)
    print("\nSaved CV comparison to data/processed/cv_comparison.csv")


if __name__ == "__main__":
    run()
