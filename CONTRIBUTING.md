# Contributing

Guide for setting up, running, and extending the customer-churn-predictor project locally.

## Local development environment

```bash
# 1. Clone the repo
git clone https://github.com/raja69reddy/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your PostgreSQL credentials and OpenAI key

# 5. Create the database and load the schema
psql -U postgres -c "CREATE DATABASE churn_db;"
psql -U postgres -d churn_db -f sql/schema.sql
```

Optional: `docker-compose up --build` starts Postgres, the FastAPI service, and the Streamlit
dashboard together (see the main [README](README.md#deployment)).

## Running the full pipeline

Run these in order for a from-scratch setup:

```bash
# 1. Generate (or refresh) the synthetic customer dataset
python mock_data/gen_customers.py

# 2. Load it into Postgres
python -m src.data.ingestion

# 3. Engineer features into processed_customers
python -m src.features.run_engineering

# 4. Train baseline models
python -m src.models.train

# 5. (Optional) tune, ensemble, and register with MLflow
python -m src.models.run_training_mlflow
python -m src.models.select_best_model

# 6. Score every customer and populate churn_predictions
python -m src.models.batch_scorer --mode full --output db

# 7. Backfill retention columns (run after any full batch-scoring pass)
python -m src.models.update_prediction_schema
```

## Training models locally

- `python -m src.models.train` trains the 4 baseline models (logistic regression, decision tree,
  random forest, XGBoost) and saves them to `models/*.pkl` (gitignored).
- `python -m src.models.run_training_mlflow` retrains all 6 leaderboard models (adds LightGBM and
  the soft-voting ensemble) with full MLflow experiment tracking (`mlflow/mlflow.db`).
- `python -m src.models.select_best_model` picks the best model by AUC and marks it active in the
  `model_registry` table; `src/models/mlflow_registry.py` handles promoting a model version to the
  MLflow **Production** stage, which is what `api/predictor.py` and `batch_scorer.py` load from.
- After training, re-run `python -m src.models.batch_scorer --mode full --output db` and
  `python -m src.models.update_prediction_schema` so predictions and retention columns reflect the
  new model.

## Running tests

```bash
# Full suite (requires a populated local churn_db + trained models)
pytest tests/ -v

# Only the tests that are safe to run without trained models / MLflow (matches CI)
pytest tests/test_ingestion.py -v
```

Some tests have real side effects worth knowing about:
- `tests/test_batch_pipeline.py` and `tests/test_ensemble.py::test_batch_scorer_processes_all_customers`
  actually run `batch_scorer.run()`, which truncates and rewrites `churn_predictions`. Re-run
  `python -m src.models.update_prediction_schema` afterward to restore the `retention_priority` /
  `recommended_action` / `estimated_revenue_at_risk` / `days_since_last_score` columns it doesn't set.
- `tests/test_mlflow.py`, `tests/test_models.py`, and `tests/test_ensemble.py` are marked
  `@pytest.mark.skip` in CI (they require trained models and a populated MLflow registry that a
  fresh checkout doesn't have) but should still be run locally after training.

## Code quality

```bash
black src/ api/ dashboard/ mock_data/ tests/
flake8 src/ api/ dashboard/ mock_data/ --max-line-length=100
```

`flake8` config lives in `.flake8` (max line length 100, `E203` ignored — that rule conflicts with
`black`'s slice formatting). Run `black` before `flake8` so formatting-only issues don't show up as
lint failures.

## Git commit message conventions

This project commits in small, single-purpose units. Follow the existing history
(`git log --oneline`) for style:

- **Day-based feature work**: `Day N: <what changed>`, e.g. `Day 13: added type hints to all key
  functions`. Used for the incremental, task-by-task build-out of the project.
- **Fixes**: `fix: <what was fixed>`, e.g. `fix: CI runs only test_ingestion.py - all others need
  local env`. Used for bug fixes and corrections outside the day-by-day plan.
- Keep each commit to one logical change (one file group / one concern) rather than bundling
  unrelated edits — makes `git log`/`git bisect` useful later.
- Write the message in the imperative/descriptive present tense ("added", "fixed", "updated" — past
  tense of the action taken, matching the existing history) and keep the subject line under ~70
  characters.

## Adding a new model

1. Add a `train_<model_name>()` (or similar) method to `ModelTrainer` in `src/models/train.py`,
   following the existing baseline methods (fit on `X_train`/`y_train`, save via `joblib.dump` to
   `models/<model_name>.pkl`, log to MLflow via `_log_mlflow_run()`).
2. Add the model name to `MODEL_NAMES` (or the relevant list) wherever the training/evaluation
   scripts iterate over known models (e.g. `src/models/run_training_mlflow.py`).
3. Evaluate it with `ModelEvaluator.evaluate_model()` (`src/models/evaluate.py`) and compare its
   AUC/F1 against the current leaderboard (`data/processed/final_model_leaderboard.csv`).
4. If it should become the active model, insert/update its row in `model_registry` (see
   `src/models/select_best_model.py` for the pattern) and promote it to MLflow **Production** via
   `src/models/mlflow_registry.py`.
5. Re-run `python -m src.models.batch_scorer --mode full --output db` and
   `python -m src.models.update_prediction_schema` so `churn_predictions` reflects the new model.
6. Add/extend tests in `tests/test_models.py` or `tests/test_ensemble.py` to cover the new model,
   and add type hints consistent with the rest of the codebase.
