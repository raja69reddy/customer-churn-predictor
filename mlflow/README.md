# MLflow Tracking — customer-churn-predictor

This project uses a local, file/SQLite-backed MLflow tracking server. All tracking data lives
under this `mlflow/` folder (`mlflow.db` for runs/metrics/params, `mlruns/` for model artifacts).
This folder is gitignored except for this README — tracking data is local-only and is not
committed to the repo.

## Starting the MLflow UI

From the project root, with the same tracking URI the scripts use:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root file:./mlflow/mlruns --port 5000
```

Then open http://localhost:5000 in a browser.

> **Environment note:** on this machine (Python 3.14 + MLflow 3.14.0), `mlflow ui` initially failed
> with `ImportError: cannot import name 'Traversable' from 'importlib.abc'` — MLflow's optional
> "assistant" feature still imports `Traversable` from its old Python location, which moved to
> `importlib.resources.abc` in Python 3.14. This was patched locally in the installed package
> (`site-packages/mlflow/assistant/skill_installer.py`) with a try/except fallback import; it does
> not affect this repo. If the UI fails to start elsewhere with the same error, apply the same
> fallback or use an MLflow version with this fixed upstream.

## Viewing experiments

All runs are logged under a single experiment named `customer-churn-predictor`
(see `EXPERIMENT_NAME` in `src/models/mlflow_setup.py`). In the UI, select that experiment from
the left sidebar to see every training run (Logistic Regression, Decision Tree, Random Forest,
XGBoost, LightGBM, Ensemble, and any tuning/challenger runs like "XGBoost (tuned) v2").

Each run has:
- **Params** — hyperparameters used (e.g. `n_estimators`, `max_depth`, `learning_rate`)
- **Metrics** — `accuracy`, `auc`, `f1`, `precision`, `recall`
- **Tags** — `model_type`, `dataset_size`, `date`
- **Artifacts** — the serialized model (`model/`) and a feature importance plot (`.png`)

## Comparing runs

In the UI: select multiple runs' checkboxes on the experiment page and click **Compare** to see
side-by-side params/metrics and parallel-coordinates/scatter plots.

From code: `src/models/experiment_analyzer.py` provides the same comparisons programmatically —
`get_run_history()` (all runs sorted by AUC), `plot_metric_comparison()` (bar chart), and
`export_run_summary()` (CSV export to `data/processed/mlflow_summary.csv`).
`src/models/mlflow_registry.py::compare_runs(run_ids)` compares a specific list of run IDs.

## Loading registered models

The best model is registered under the name `churn-predictor`
(`mlflow_registry.REGISTERED_MODEL_NAME`). To load the current Production version in code:

```python
from src.models.mlflow_registry import get_production_model
model = get_production_model()  # returns the native sklearn/xgboost/lightgbm estimator
```

Or directly via MLflow:

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/churn-predictor/Production")
```

## Experiment naming conventions

- One experiment per project: `customer-churn-predictor` — do not create a new experiment per day
  or per model; log every run (including tuning/challenger attempts) into this single experiment
  so all runs remain comparable.
- Run names follow the model's display name, e.g. `"XGBoost (tuned)"`, `"Random Forest (tuned)"`,
  `"XGBoost (tuned) v2"` for challenger/version-2 attempts. Avoid renaming an existing model's run
  name between versions — append `" v2"`, `" v3"`, etc. instead so history stays sortable/greppable.
- The registered model name (`churn-predictor`) is intentionally generic and stable across
  algorithm changes — whichever algorithm currently wins by AUC gets promoted under that same
  registered name, rather than each algorithm having its own registered model name.

## Model registry stages

MLflow's registry uses three stages per registered model version:

- **Staging** — a candidate version being evaluated, not yet serving predictions.
- **Production** — the version currently used by `batch_scorer.py` and other serving code
  (loaded via `models:/churn-predictor/Production`). Only one version should hold this stage;
  `mlflow_registry.promote_to_production()` archives the previous Production version automatically
  (`archive_existing_versions=True`).
- **Archived** — a previously-Production (or Staging) version kept for history/rollback, no longer
  actively served.

Note: MLflow has deprecated stage-based registry in favor of aliases/tags in newer releases, but
stages are still functional (with a `FutureWarning`) in the MLflow version pinned for this project
and are used here for simplicity.
