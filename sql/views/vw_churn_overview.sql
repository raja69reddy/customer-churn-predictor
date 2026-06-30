-- vw_churn_overview: high-level churn statistics across key dimensions
CREATE OR REPLACE VIEW vw_churn_overview AS

-- 1. Overall churn rate
SELECT
    'overall'                          AS dimension,
    'all'                              AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct,
    ROUND(AVG(tenure)::NUMERIC, 2)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges)::NUMERIC, 2) AS avg_monthly_charges
FROM raw_customers

UNION ALL

-- 2. Churn by contract type
SELECT
    'contract'                         AS dimension,
    contract                           AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct,
    ROUND(AVG(tenure)::NUMERIC, 2)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges)::NUMERIC, 2) AS avg_monthly_charges
FROM raw_customers
GROUP BY contract

UNION ALL

-- 3. Churn by gender
SELECT
    'gender'                           AS dimension,
    gender                             AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct,
    ROUND(AVG(tenure)::NUMERIC, 2)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges)::NUMERIC, 2) AS avg_monthly_charges
FROM raw_customers
GROUP BY gender

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
    )                                  AS churn_rate_pct,
    ROUND(AVG(tenure)::NUMERIC, 2)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges)::NUMERIC, 2) AS avg_monthly_charges
FROM raw_customers
GROUP BY senior_citizen;
