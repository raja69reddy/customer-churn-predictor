-- vw_feature_importance: churn rate broken down by engineered features from processed_customers
CREATE OR REPLACE VIEW vw_feature_importance AS

-- 1. Churn by tenure group
SELECT
    'tenure_group'                     AS dimension,
    tenure_group                       AS segment,
    COUNT(*)                           AS total_customers,
    SUM(churn_label)                   AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2) AS churn_rate_pct
FROM processed_customers
GROUP BY tenure_group

UNION ALL

-- 2. Churn by contract risk score
SELECT
    'contract_risk_score'              AS dimension,
    contract_risk_score::TEXT          AS segment,
    COUNT(*)                           AS total_customers,
    SUM(churn_label)                   AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2) AS churn_rate_pct
FROM processed_customers
GROUP BY contract_risk_score

UNION ALL

-- 3. Churn by payment risk score
SELECT
    'payment_risk_score'               AS dimension,
    payment_risk_score::TEXT           AS segment,
    COUNT(*)                           AS total_customers,
    SUM(churn_label)                   AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2) AS churn_rate_pct
FROM processed_customers
GROUP BY payment_risk_score

UNION ALL

-- 4. Churn by services count
SELECT
    'services_count'                   AS dimension,
    services_count::TEXT               AS segment,
    COUNT(*)                           AS total_customers,
    SUM(churn_label)                   AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2) AS churn_rate_pct
FROM processed_customers
GROUP BY services_count

UNION ALL

-- 5. Churn by charge_per_month tier
SELECT
    'charge_per_month_tier'            AS dimension,
    CASE
        WHEN charge_per_month < 35 THEN 'low'
        WHEN charge_per_month < 70 THEN 'medium'
        ELSE 'high'
    END                                AS segment,
    COUNT(*)                           AS total_customers,
    SUM(churn_label)                   AS churned,
    ROUND(100.0 * SUM(churn_label) / COUNT(*), 2) AS churn_rate_pct
FROM processed_customers
GROUP BY
    CASE
        WHEN charge_per_month < 35 THEN 'low'
        WHEN charge_per_month < 70 THEN 'medium'
        ELSE 'high'
    END;
