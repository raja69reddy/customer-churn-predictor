# Data Dictionary — Customer Churn Predictor

Reference documentation for every table and view in the `churn_db` PostgreSQL database. See
`sql/schema.sql` for the authoritative table definitions and `sql/views/*.sql` for the views.

## Tables

### `raw_customers`
The source-of-truth customer dataset, loaded by `src/data/ingestion.py` from
`data/raw/customers.csv`. One row per customer; `customer_id` is the primary key used to join
every other table in the schema.

| Column | Type | Description |
|---|---|---|
| `customer_id` | VARCHAR (PK) | Unique customer identifier, e.g. `CUST-00001`. |
| `gender` | VARCHAR(10) | `Male` / `Female`. |
| `senior_citizen` | INTEGER | `1` if the customer is a senior citizen, else `0`. |
| `partner` | VARCHAR(5) | `Yes` / `No` — has a partner. |
| `dependents` | VARCHAR(5) | `Yes` / `No` — has dependents. |
| `tenure` | INTEGER | Months the customer has been with the company (0-72 in this dataset). |
| `phone_service` | VARCHAR(5) | `Yes` / `No`. |
| `multiple_lines` | VARCHAR(20) | `Yes` / `No` / `No phone service`. |
| `internet_service` | VARCHAR(20) | `DSL` / `Fiber optic` / `No`. |
| `online_security` | VARCHAR(20) | `Yes` / `No` / `No internet service`. |
| `online_backup` | VARCHAR(20) | `Yes` / `No` / `No internet service`. |
| `device_protection` | VARCHAR(20) | `Yes` / `No` / `No internet service`. |
| `tech_support` | VARCHAR(20) | `Yes` / `No` / `No internet service`. |
| `streaming_tv` | VARCHAR(20) | `Yes` / `No` / `No internet service`. |
| `streaming_movies` | VARCHAR(20) | `Yes` / `No` / `No internet service`. |
| `contract` | VARCHAR(20) | `Month-to-month` / `One year` / `Two year`. |
| `paperless_billing` | VARCHAR(5) | `Yes` / `No`. |
| `payment_method` | VARCHAR(30) | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`. |
| `monthly_charges` | FLOAT | Current monthly bill amount ($). |
| `total_charges` | FLOAT | Cumulative amount billed to date ($). |
| `churn` | VARCHAR(5) | Ground-truth label: `Yes` / `No`. Never changes after ingestion. |
| `created_at` | TIMESTAMP | Row insert time (defaults to `NOW()`). |

### `processed_customers`
Feature-engineered output of `src/features/engineering.py`, one row per customer. Holds only
the summary engineered columns (not a full one-hot feature matrix — the sklearn
`ColumnTransformer` pipeline used at model-training time is saved separately to
`models/preprocessor.pkl`).

| Column | Type | Description |
|---|---|---|
| `customer_id` | VARCHAR (PK) | Joins to `raw_customers.customer_id`. |
| `tenure_group` | VARCHAR(20) | Bucketed tenure: `New Customer`, `Growing Customer`, `Established Customer`, `Loyal Customer`. |
| `charge_per_month` | FLOAT | `total_charges / tenure` (tenure floored at 1 to avoid divide-by-zero). |
| `services_count` | INTEGER | Count of active add-on/phone/internet services (0-8). |
| `contract_risk_score` | INTEGER | 1-3, higher = riskier contract type (Month-to-month=3, One year=2, Two year=1). |
| `payment_risk_score` | INTEGER | 1-3, higher = riskier payment method (Electronic check=3 down to autopay=1). |
| `churn_label` | INTEGER | Binary version of `raw_customers.churn` (`Yes`→1, `No`→0). |
| `churn_probability` | FLOAT | Legacy column from early pipeline versions — **not** the current source of truth for predictions; see `churn_predictions` instead. |
| `risk_segment` | VARCHAR(20) | Legacy column, same caveat as above. |
| `created_at` | TIMESTAMP | Row insert time. |

### `churn_predictions`
The current, single-latest prediction per customer. Overwritten on every full/incremental
batch-scoring run (`src/models/batch_scorer.py`) — it does **not** hold history; see
`churn_predictions_history` for that.

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL (PK) | Surrogate key. |
| `customer_id` | VARCHAR | Joins to `raw_customers.customer_id`. |
| `churn_probability` | FLOAT | Model output, 0.0-1.0. |
| `risk_segment` | VARCHAR(20) | `High` (≥0.7) / `Medium` (≥0.4) / `Low` (<0.4) by default thresholds. |
| `model_version` | VARCHAR(20) | Identifies which model produced this score (e.g. `churn-predictor_v1`). |
| `predicted_at` | TIMESTAMP | When this row was scored. |
| `retention_priority` | VARCHAR(10) | `Tier1`/`Tier2`/`Tier3` — High-risk customers only, by `monthly_charges` tier. NULL for Medium/Low. Backfilled by `src/models/update_prediction_schema.py`. |
| `recommended_action` | TEXT | Free-text retention action, backfilled alongside `retention_priority`. |
| `estimated_revenue_at_risk` | FLOAT | `monthly_charges * churn_probability`. |
| `days_since_last_score` | INTEGER | Days since `predicted_at`. |

**Important:** `retention_priority`/`recommended_action`/`estimated_revenue_at_risk`/
`days_since_last_score` get wiped back to NULL every time `batch_scorer.py` runs in full mode
or `run_training.py`/`select_best_model.py` etc. rewrite `model_registry` — always re-run
`python -m src.models.update_prediction_schema` afterward to repopulate them.

### `churn_predictions_history`
Archive table. Right before `churn_predictions` overwrites a customer's row (batch or
incremental scoring), the outgoing row is copied here first — this is what makes score-drift
and risk-segment-transition analysis possible, since `churn_predictions` itself only ever
holds the single latest score per customer.

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL (PK) | Surrogate key. |
| `customer_id` | VARCHAR | Joins to `raw_customers.customer_id`. |
| `churn_probability` | FLOAT | The score at the time it was archived. |
| `risk_segment` | VARCHAR(20) | The segment at the time it was archived. |
| `model_version` | VARCHAR(20) | Which model produced the archived score. |
| `predicted_at` | TIMESTAMP | When the archived score was originally produced. |
| `archived_at` | TIMESTAMP | When this row was archived (i.e., right before the next overwrite). |

To get each customer's most recent prior score: `DISTINCT ON (customer_id) ... ORDER BY
customer_id, archived_at DESC`.

### `model_registry`
One row per trained model per training run — reruns of `run_training.py` /
`run_training_mlflow.py` / `select_best_model.py` etc. **append** new rows rather than
upserting, so the same `model_name` can appear multiple times with different `trained_at`
values. `is_active` marks the single model currently used by `ChurnPredictor.load_best_model()`
(used by the dashboard); the FastAPI layer instead reads the separate MLflow Model Registry
`Production` stage — the two are usually, but not automatically, kept in sync.

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL (PK) | Surrogate key. |
| `model_name` | VARCHAR(100) | e.g. `xgboost_tuned`, `lightgbm`, `ensemble`. |
| `model_version` | VARCHAR(20) | `v1` (baseline) or `v2` (tuned), informal versioning. |
| `accuracy` | FLOAT | Test-set accuracy. |
| `auc_score` | FLOAT | Test-set ROC-AUC — the primary model-selection metric. |
| `f1_score` | FLOAT | Test-set F1. |
| `trained_at` | TIMESTAMP | When this run completed. |
| `is_active` | BOOLEAN | `TRUE` for exactly one row — the model the dashboard loads. Recompute with `python -m src.models.update_registry` after any retrain. |

## SQL Views

| View | Purpose |
|---|---|
| `vw_churn_overview` | Churn rate, avg tenure, avg monthly charges — broken down by contract, gender, senior-citizen status, tenure group, charge tier, and services count, plus one `overall` row. |
| `vw_churn_by_features` | Similar breakdown to `vw_churn_overview` but focused on internet service, payment method, and partner/dependents combinations. |
| `vw_feature_importance` | Churn rate broken down by the *engineered* features in `processed_customers` (tenure group, contract/payment risk score, services count, charge tier). |
| `vw_high_risk` | Every customer scored against a simple 4-flag rule (contract risk, payment risk, low services count, low tenure) into High/Medium/Low — an earlier, rule-based risk segmentation independent of the ML model. |
| `vw_churn_predictions` | The current `churn_predictions` joined with `raw_customers` profile fields, sorted by probability descending. Powers the API's `/customers/high-risk` and the dashboard's Predictions page. |
| `vw_risk_segments` | Per-segment counts, averages, and revenue-at-risk, plus a synthetic `High+Medium (At Risk)` combined row. Powers `/customers/risk-summary`, `/analytics/revenue-at-risk`, and the dashboard Overview KPIs. |
| `vw_model_performance` | All `model_registry` rows ranked by AUC, with the active model highlighted and an `auc_rank`. Powers `/model/performance` and the dashboard Model Performance page. |
| `vw_prediction_summary` | Multi-section view (`segment_summary`, `top_20_highest_risk`, `improving_scores`, `worsening_scores`) combining current scores with `churn_predictions_history` for drift analysis. Different sections populate different columns — filter on the `section` column first. |

`sql/queries/prediction_analysis.sql` also contains standalone drift-analysis queries that
aren't wrapped in a view.

## Sample Queries

**Overall churn rate:**
```sql
SELECT ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM raw_customers;
```

**Top 10 highest-risk customers right now:**
```sql
SELECT customer_id, churn_probability, risk_segment
FROM churn_predictions
ORDER BY churn_probability DESC
LIMIT 10;
```

**Currently active production model:**
```sql
SELECT model_name, model_version, auc_score, trained_at
FROM model_registry
WHERE is_active = TRUE;
```

**Monthly revenue at risk (High + Medium segments):**
```sql
SELECT * FROM vw_risk_segments WHERE risk_segment = 'High+Medium (At Risk)';
```

**Customers whose churn probability got worse since their last scoring run:**
```sql
SELECT customer_id, segment, churn_probability, prev_probability
FROM vw_prediction_summary
WHERE section = 'worsening_scores'
ORDER BY (churn_probability - prev_probability) DESC;
```

**Churn rate by contract type, with sample size:**
```sql
SELECT segment, total_customers, churn_rate_pct
FROM vw_churn_overview
WHERE dimension = 'contract'
ORDER BY churn_rate_pct DESC;
```

**A specific customer's full prediction + profile:**
```sql
SELECT * FROM vw_churn_predictions WHERE customer_id = 'CUST-05036';
```

**Model leaderboard, best first:**
```sql
SELECT model_name, model_version, auc_score, f1_score, highlight
FROM vw_model_performance
ORDER BY auc_rank;
```
