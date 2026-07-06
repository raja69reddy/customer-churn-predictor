"""Trains a v2 challenger for the current best (XGBoost) model and promotes the winner."""
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.models import mlflow_registry, mlflow_setup
from src.models.train import ModelTrainer

V2_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
}


def run() -> None:
    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    search = RandomizedSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=1),
        param_distributions=V2_PARAM_DISTRIBUTIONS,
        n_iter=50,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(trainer.X_train, trainer.y_train)

    v2_model = search.best_estimator_
    print(f"v2 candidate best params: {search.best_params_}")
    print(f"v2 candidate best CV AUC: {search.best_score_:.4f}")

    y_pred = v2_model.predict(trainer.X_test)
    y_proba = v2_model.predict_proba(trainer.X_test)[:, 1]
    v2_metrics = trainer._print_metrics("XGBoost (tuned) v2", trainer.y_test, y_pred, y_proba)

    trainer.save_model(v2_model, "xgboost_tuned_v2")
    v2_run_id = trainer._log_mlflow_run("XGBoost (tuned) v2", search.best_params_, v2_metrics, v2_model)

    mlflow_setup.configure_tracking()
    client = MlflowClient()
    model_name = mlflow_registry.REGISTERED_MODEL_NAME

    versions = client.search_model_versions(f"name='{model_name}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        raise RuntimeError("No Production model found — run register_production_model.py first.")
    v1_version = production[0]
    v1_run = client.get_run(v1_version.run_id)
    v1_auc = v1_run.data.metrics["auc"]

    comparison = pd.DataFrame([
        {"version": f"v1 (production, {v1_run.data.tags.get('mlflow.runName', '')})", "auc": v1_auc},
        {"version": "v2 (challenger)", "auc": v2_metrics["auc"]},
    ])
    print("\nVersion comparison:")
    print(comparison.to_string(index=False))

    if v2_metrics["auc"] > v1_auc:
        import mlflow
        registered = mlflow.register_model(f"runs:/{v2_run_id}/model", model_name)
        mlflow_registry.promote_to_production(model_name, registered.version)
        print(f"\nv2 is better (AUC {v2_metrics['auc']:.4f} > {v1_auc:.4f}) — promoted to Production, v1 archived.")
    else:
        print(f"\nv1 remains better or equal (AUC {v1_auc:.4f} >= {v2_metrics['auc']:.4f}) — keeping v1 as Production.")


if __name__ == "__main__":
    run()
