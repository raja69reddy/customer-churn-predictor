-- vw_churn_by_features: churn rate broken down by key customer/account features
CREATE OR REPLACE VIEW vw_churn_by_features AS

-- 1. Churn by contract type
SELECT
    'contract'                         AS dimension,
    contract                           AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct
FROM raw_customers
GROUP BY contract

UNION ALL

-- 2. Churn by internet service type
SELECT
    'internet_service'                 AS dimension,
    internet_service                   AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct
FROM raw_customers
GROUP BY internet_service

UNION ALL

-- 3. Churn by payment method
SELECT
    'payment_method'                   AS dimension,
    payment_method                     AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct
FROM raw_customers
GROUP BY payment_method

UNION ALL

-- 4. Churn by senior citizen status
SELECT
    'senior_citizen'                   AS dimension,
    CASE WHEN senior_citizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct
FROM raw_customers
GROUP BY senior_citizen

UNION ALL

-- 5. Churn by partner and dependents combination
SELECT
    'partner_dependents'               AS dimension,
    CASE
        WHEN partner = 'Yes' AND dependents = 'Yes' THEN 'Partner+Dependents'
        WHEN partner = 'Yes' AND dependents = 'No'  THEN 'Partner Only'
        WHEN partner = 'No'  AND dependents = 'Yes' THEN 'Dependents Only'
        ELSE 'Neither'
    END                                AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct
FROM raw_customers
GROUP BY
    CASE
        WHEN partner = 'Yes' AND dependents = 'Yes' THEN 'Partner+Dependents'
        WHEN partner = 'Yes' AND dependents = 'No'  THEN 'Partner Only'
        WHEN partner = 'No'  AND dependents = 'Yes' THEN 'Dependents Only'
        ELSE 'Neither'
    END;
