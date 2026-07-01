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
GROUP BY senior_citizen

UNION ALL

-- 5. Churn by tenure group
SELECT
    'tenure_group'                     AS dimension,
    CASE
        WHEN tenure <= 12 THEN '0-12'
        WHEN tenure <= 24 THEN '12-24'
        WHEN tenure <= 48 THEN '24-48'
        ELSE '48+'
    END                                AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct,
    ROUND(AVG(tenure)::NUMERIC, 2)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges)::NUMERIC, 2) AS avg_monthly_charges
FROM raw_customers
GROUP BY
    CASE
        WHEN tenure <= 12 THEN '0-12'
        WHEN tenure <= 24 THEN '12-24'
        WHEN tenure <= 48 THEN '24-48'
        ELSE '48+'
    END

UNION ALL

-- 6. Churn by monthly charge tier
SELECT
    'monthly_charge_tier'              AS dimension,
    CASE
        WHEN monthly_charges < 35 THEN 'low'
        WHEN monthly_charges < 70 THEN 'medium'
        ELSE 'high'
    END                                AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct,
    ROUND(AVG(tenure)::NUMERIC, 2)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges)::NUMERIC, 2) AS avg_monthly_charges
FROM raw_customers
GROUP BY
    CASE
        WHEN monthly_charges < 35 THEN 'low'
        WHEN monthly_charges < 70 THEN 'medium'
        ELSE 'high'
    END

UNION ALL

-- 7. Churn by number of services subscribed
SELECT
    'services_count'                   AS dimension,
    services_count::TEXT               AS segment,
    COUNT(*)                           AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)          AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                  AS churn_rate_pct,
    ROUND(AVG(tenure)::NUMERIC, 2)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges)::NUMERIC, 2) AS avg_monthly_charges
FROM (
    SELECT
        *,
        (phone_service = 'Yes')::INT
        + (internet_service <> 'No')::INT
        + (online_security = 'Yes')::INT
        + (online_backup = 'Yes')::INT
        + (device_protection = 'Yes')::INT
        + (tech_support = 'Yes')::INT
        + (streaming_tv = 'Yes')::INT
        + (streaming_movies = 'Yes')::INT AS services_count
    FROM raw_customers
) rc_services
GROUP BY services_count;
