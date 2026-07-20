# System Architecture — Customer Churn Predictor

## Overview

The system is a layered ML pipeline with two independent serving surfaces (a REST API and a
dashboard) that both read from the same trained model, so predictions never diverge between
"programmatic" and "interactive" use. Everything runs from a single PostgreSQL database as the
source of truth, with MLflow as a parallel model-artifact and experiment store.

## Data Flow (ASCII)

```
                                mock_data/gen_customers.py
                                          |
                                          v
                              data/raw/customers.csv
                                          |
                              src/data/ingestion.py
                                          |
                                          v
                          +-------------------------------+
                          |   PostgreSQL: raw_customers    |
                          +-------------------------------+
                                          |
                          src/features/engineering.py
                                          |
                                          v
                       +----------------------------------+
                       | PostgreSQL: processed_customers   |
                       |  models/preprocessor.pkl          |
                       +----------------------------------+
                                          |
              +---------------------------+---------------------------+
              |                                                       |
    src/models/train.py                                    src/models/run_training_mlflow.py
    (4 baseline models)                                     (6 models, full MLflow logging)
              |                                                       |
              v                                                       v
     PostgreSQL: model_registry  <----------------------->   MLflow tracking store
     (Postgres-side "active"          kept in sync by          (mlflow/mlflow.db +
      model flag)                  update_registry.py /         mlruns/, Model Registry
                                    manual promotion             "Production" stage)
                                          |
                                          v
                          src/models/batch_scorer.py
                                          |
                                          v
                       +----------------------------------+
                       | PostgreSQL: churn_predictions     |
                       | PostgreSQL: churn_predictions_    |
                       |             history (archive)     |
                       +----------------------------------+
                                          |
                  +-----------------------+-----------------------+
                  |                                                |
         api/main.py (FastAPI)                          dashboard/app.py (Streamlit)
         loads model from MLflow                         loads model_registry's
         "Production" stage                              is_active row (live mode) or
                  |                                       data/demo/*.csv (demo mode,
                  v                                       no DB required)
          REST clients / curl                                     |
                                                                    v
                                                          Browser (5 interactive pages)
```

## Component Descriptions

### 1. Data Ingestion Layer (`src/data/ingestion.py`, `mock_data/gen_customers.py`)
Generates a synthetic Telco-style dataset (7,043 rows, ~26% churn, realistic correlations
between contract type, tenure, and churn) and loads it into `raw_customers` via SQLAlchemy,
with `--mode full` (truncate + reload) or `--mode incremental` (append new customer IDs only).
Validates required columns, strips whitespace, coerces numeric types, and logs data-quality
warnings (duplicate IDs, invalid churn values, out-of-range tenure) without hard-failing on
them — the pipeline is designed to tolerate realistic messy data rather than demand perfection.

### 2. Feature Engineering Layer (`src/features/engineering.py`)
The `FeatureEngineer` class turns raw fields into modeling-ready features: tenure buckets,
`charge_per_month`, `services_count`, and two composite risk scores (contract type, payment
method) derived directly from the EDA findings. Also builds and persists a separate sklearn
`ColumnTransformer` preprocessing pipeline (imputer + scaler + one-hot encoder) to
`models/preprocessor.pkl`, independent of the summary columns written to
`processed_customers`.

### 3. ML Model Layer (`src/models/train.py`, `evaluate.py`, `predict.py`)
`ModelTrainer` trains and saves 6 models (Logistic Regression, Decision Tree, Random Forest,
XGBoost, LightGBM, and a soft-voting Ensemble of the three strongest), with GridSearchCV tuning
for Random Forest and XGBoost. `ModelEvaluator` computes accuracy/AUC/F1/precision/recall, ROC
curves, confusion matrices, and SHAP explanations. `ChurnPredictor` is the inference-time
wrapper used by both the dashboard and (indirectly, via `api/predictor.py`) the API — it runs
the same feature-engineering steps on a single customer dict and returns a probability, risk
segment, top SHAP factors, and a GPT (or template-fallback) explanation.

### 4. MLflow Tracking Layer (`src/models/mlflow_setup.py`, `mlflow_registry.py`)
Every training run logs parameters, metrics, and the model artifact itself to a local MLflow
tracking store (SQLite backend + file-based artifact store, gitignored). The best model is
promoted to the `Production` stage in the MLflow Model Registry — this is the source of truth
`api/predictor.py` reads from at startup, independent of the Postgres `model_registry` table's
`is_active` flag (which the dashboard reads instead). The two are kept in sync manually / via
`update_registry.py`, not automatically — a deliberate simplification for a solo project, noted
as a known limitation.

### 5. API Layer (`api/`)
A FastAPI service (`api/main.py`) exposing 17 endpoints across three groups: prediction
(`/predict`, `/predict/batch`, `/model/info`, `/model/performance`), customer lookup
(`/customer/{id}`, `/customer/{id}/predict`, `/customer/{id}/explanation`,
`/customers/high-risk`, `/customers/risk-summary`), and analytics (`/analytics/*`).
`api/predictor.py` wraps `ChurnPredictor` with MLflow model loading; `api/database.py`
centralizes the raw SQL data-access functions used by the DB-backed endpoints. Validation is
handled by Pydantic models (`api/models.py`), with a global exception handler returning
friendly 500s instead of raw tracebacks.

### 6. Dashboard Layer (`dashboard/`)
A 5-page Streamlit app — Overview, Predictions, Customer Lookup, Model Performance, Retention
Targeting — each independently re-declaring the same sidebar filters (Streamlit's per-widget
`session_state` keeps them in sync across page navigation without a shared module). Every page
tries a live PostgreSQL connection first and transparently falls back to `dashboard/demo_mode.py`
(static CSVs in `data/demo/`, committed to git) if none is available — this is what lets the
dashboard run on Streamlit Community Cloud with a lightweight `requirements_streamlit.txt` that
excludes the heavy ML stack entirely. Pages that need it (Customer Lookup, Model Performance)
guard their heavy imports (`xgboost`, `shap`, `matplotlib`, `sklearn`) behind `try/except
ImportError` so the app doesn't crash when those packages aren't installed.

## Technology Decisions and Rationale

| Decision | Rationale |
|---|---|
| **PostgreSQL over SQLite/flat files** | Real concurrent read/write from ingestion, training, batch scoring, the API, and the dashboard simultaneously; window functions and `DISTINCT ON` used extensively for the history/trend views. |
| **MLflow *and* a Postgres `model_registry` table** | MLflow for proper experiment tracking, artifact versioning, and stage promotion; a lightweight Postgres table for the dashboard because it's simpler to query with plain SQL than the MLflow tracking API for basic "what's active" checks. Accepted the sync-drift tradeoff rather than building a single unified registry — pragmatic for project scope. |
| **XGBoost as the production model** | Best AUC after tuning (0.9624) among 6 candidates, and tree-based — which matters because SHAP's fast, exact `TreeExplainer` only works on tree-based models, and explainability was a first-class requirement, not an afterthought. |
| **SHAP over simpler feature-importance methods** | Gives per-prediction, per-feature attribution (not just global importance), which is what powers the "why is this customer high-risk" explanation in both the API and dashboard. |
| **FastAPI over Flask/Django** | Native Pydantic request/response validation, automatic OpenAPI docs, and async support, with minimal boilerplate for a focused prediction-serving use case. |
| **Streamlit over a custom frontend** | Fast to build a genuinely interactive multipage app (filters, live charts, an ROI estimator) without writing separate frontend/backend code — appropriate for a solo project's dashboard needs. |
| **GPT explanations with a template fallback** | Produces a genuinely useful plain-English summary when an API key is configured, but the system is fully functional without one (falls back to a deterministic template built from the SHAP factors) — no hard dependency on a paid external service. |
| **Docker Compose (Postgres + FastAPI + Streamlit)** | One-command full-stack startup for local/internal deployment, separate from the lightweight Streamlit-Cloud-only deployment path (`requirements_streamlit.txt` + demo mode). |
| **SMOTE for class imbalance (training-time only)** | Balances the ~26/74 churn split during training without altering the real-world class distribution used for evaluation or scoring — avoids inflating apparent performance on the actual test set. |
