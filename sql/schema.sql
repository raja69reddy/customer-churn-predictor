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
    id                SERIAL PRIMARY KEY,
    customer_id       VARCHAR,
    churn_probability FLOAT,
    risk_segment      VARCHAR(20),
    model_version     VARCHAR(20),
    predicted_at      TIMESTAMP DEFAULT NOW()
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
