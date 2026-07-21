-- retention_roi.sql — campaign cost/ROI analysis for retention outreach. Each block is
-- independent and can be run on its own.
--
-- Assumptions (matching the dashboard's Campaign ROI Estimator defaults on the Retention
-- Targeting page, dashboard/pages/5_retention.py, so the two stay consistent):
--   campaign_cost_per_customer = $15
--   expected_retention_success_rate = 30%
-- "Revenue recovered" is annualized (monthly_charges * 12) since a successful retention
-- outreach is assumed to keep the customer for at least a year, not just one billing cycle.

-- =============================================================
-- 1. ROI of retention campaigns by tier (Tier1/Tier2/Tier3, High-risk only —
--    retention_priority is only populated for High risk customers)
-- =============================================================
WITH tier_stats AS (
    SELECT
        retention_priority,
        COUNT(*)                          AS customers_targeted,
        AVG(rc.monthly_charges)           AS avg_monthly_charges
    FROM churn_predictions cp
    JOIN raw_customers rc ON rc.customer_id = cp.customer_id
    WHERE retention_priority IS NOT NULL
    GROUP BY retention_priority
)
SELECT
    retention_priority,
    customers_targeted,
    ROUND((customers_targeted * 15.0)::NUMERIC, 2)                                   AS campaign_cost,
    ROUND((customers_targeted * 0.30 * avg_monthly_charges * 12)::NUMERIC, 2)        AS revenue_recovered,
    ROUND(
        (
            (customers_targeted * 0.30 * avg_monthly_charges * 12 - customers_targeted * 15.0)
            / NULLIF(customers_targeted * 15.0, 0) * 100
        )::NUMERIC,
        1
    )                                                                                 AS roi_pct
FROM tier_stats
ORDER BY retention_priority;

-- =============================================================
-- 2. Cost per customer actually saved (campaign cost / expected customers retained)
-- =============================================================
WITH tier_stats AS (
    SELECT
        retention_priority,
        COUNT(*)                AS customers_targeted,
        COUNT(*) * 0.30          AS expected_customers_retained
    FROM churn_predictions
    WHERE retention_priority IS NOT NULL
    GROUP BY retention_priority
)
SELECT
    retention_priority,
    customers_targeted,
    ROUND(expected_customers_retained::NUMERIC, 1)                                          AS expected_customers_retained,
    ROUND((customers_targeted * 15.0 / NULLIF(expected_customers_retained, 0))::NUMERIC, 2) AS cost_per_customer_saved
FROM tier_stats
ORDER BY retention_priority;

-- =============================================================
-- 3. Revenue recovered by tier (annualized, at 30% success rate)
-- =============================================================
SELECT
    cp.retention_priority,
    COUNT(*)                                                                      AS customers_targeted,
    ROUND(AVG(rc.monthly_charges)::NUMERIC, 2)                                    AS avg_monthly_charges,
    ROUND((COUNT(*) * 0.30 * AVG(rc.monthly_charges) * 12)::NUMERIC, 2)           AS annual_revenue_recovered
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
WHERE cp.retention_priority IS NOT NULL
GROUP BY cp.retention_priority
ORDER BY annual_revenue_recovered DESC;

-- =============================================================
-- 4. Break-even analysis by tier — the retention success rate needed for
--    revenue recovered (annualized) to equal campaign cost
-- =============================================================
WITH tier_stats AS (
    SELECT
        retention_priority,
        COUNT(*)                          AS customers_targeted,
        AVG(rc.monthly_charges)           AS avg_monthly_charges
    FROM churn_predictions cp
    JOIN raw_customers rc ON rc.customer_id = cp.customer_id
    WHERE retention_priority IS NOT NULL
    GROUP BY retention_priority
)
SELECT
    retention_priority,
    customers_targeted,
    ROUND((customers_targeted * 15.0)::NUMERIC, 2)                                            AS campaign_cost,
    ROUND((15.0 / NULLIF(avg_monthly_charges * 12, 0) * 100)::NUMERIC, 2)                      AS break_even_success_rate_pct,
    ROUND((customers_targeted * (15.0 / NULLIF(avg_monthly_charges * 12, 0)))::NUMERIC, 1)     AS break_even_customers_to_retain
FROM tier_stats
ORDER BY retention_priority;

-- =============================================================
-- 5. Expected savings by risk segment (High/Medium/Low), at 30% success rate,
--    annualized — broader view than tier (tier only covers High risk)
-- =============================================================
SELECT
    cp.risk_segment,
    COUNT(*)                                                                       AS customer_count,
    ROUND((COUNT(*) * 15.0)::NUMERIC, 2)                                          AS campaign_cost,
    ROUND((COUNT(*) * 0.30 * AVG(rc.monthly_charges) * 12)::NUMERIC, 2)           AS expected_annual_savings
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
WHERE cp.risk_segment IN ('High', 'Medium')
GROUP BY cp.risk_segment
ORDER BY expected_annual_savings DESC;
