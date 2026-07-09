-- =============================================================
-- Customer Churn Predictor — Database Schema
-- =============================================================

-- 1. Raw customer data (Telco dataset)
CREATE TABLE IF NOT EXISTS raw_customers (
    customer_id        VARCHAR PRIMARY KEY,
    gender             VARCHAR(10),
    senior_citizen     INTEGER,
    partner            VARCHAR(5),
    dependents         VARCHAR(5),
    tenure             INTEGER,
    phone_service      VARCHAR(5),
    multiple_lines     VARCHAR(20),
    internet_service   VARCHAR(20),
    online_security    VARCHAR(20),
    online_backup      VARCHAR(20),
    device_protection  VARCHAR(20),
    tech_support       VARCHAR(20),
    streaming_tv       VARCHAR(20),
    streaming_movies   VARCHAR(20),
    contract           VARCHAR(20),
    paperless_billing  VARCHAR(5),
    payment_method     VARCHAR(30),
    monthly_charges    FLOAT,
    total_charges      FLOAT,
    churn              VARCHAR(5),
    created_at         TIMESTAMP DEFAULT NOW()
);

-- 2. Feature-engineered / processed customer data
CREATE TABLE IF NOT EXISTS processed_customers (
    customer_id          VARCHAR PRIMARY KEY,
    tenure_group         VARCHAR(20),
    charge_per_month     FLOAT,
    services_count       INTEGER,
    contract_risk_score  INTEGER,
    payment_risk_score   INTEGER,
    churn_label          INTEGER,
    churn_probability    FLOAT,
    risk_segment         VARCHAR(20),
    created_at           TIMESTAMP DEFAULT NOW()
);

-- 3. Model prediction log
CREATE TABLE IF NOT EXISTS churn_predictions (
    id                         SERIAL PRIMARY KEY,
    customer_id                VARCHAR,
    churn_probability          FLOAT,
    risk_segment                VARCHAR(20),
    model_version               VARCHAR(20),
    predicted_at                 TIMESTAMP DEFAULT NOW(),
    -- Day 9: retention targeting + monitoring columns
    -- (backfilled by src/models/update_prediction_schema.py)
    retention_priority           VARCHAR(10),
    recommended_action           TEXT,
    estimated_revenue_at_risk    FLOAT,
    days_since_last_score        INTEGER
);

-- 4. Model registry (version tracking)
CREATE TABLE IF NOT EXISTS model_registry (
    id            SERIAL PRIMARY KEY,
    model_name    VARCHAR(100),
    model_version VARCHAR(20),
    accuracy      FLOAT,
    auc_score     FLOAT,
    f1_score      FLOAT,
    trained_at    TIMESTAMP DEFAULT NOW(),
    is_active     BOOLEAN DEFAULT FALSE
);

-- 5. Prediction history — archived copy of a churn_predictions row taken right before it is
--    overwritten by a new batch/incremental scoring run, so score drift and risk-segment
--    transitions can be analyzed over time (churn_predictions itself only ever holds each
--    customer's single latest prediction).
CREATE TABLE IF NOT EXISTS churn_predictions_history (
    id                SERIAL PRIMARY KEY,
    customer_id       VARCHAR,
    churn_probability FLOAT,
    risk_segment      VARCHAR(20),
    model_version     VARCHAR(20),
    predicted_at      TIMESTAMP,
    archived_at       TIMESTAMP DEFAULT NOW()
);
