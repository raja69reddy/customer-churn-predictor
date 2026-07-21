-- customer_lifetime_value.sql — CLV estimation and CLV-at-risk analysis, built on top of
-- churn_predictions. Each block is independent and can be run on its own.
--
-- CLV formula: CLV = monthly_charges * expected_tenure_months, where
--   expected_tenure_months = LEAST(60, 1 / churn_probability)
-- This is the standard "expected lifetime under a geometric churn process" estimate: if a
-- customer churns with probability p each period, their expected number of remaining
-- periods is 1/p. Capped at 60 months (5 years) so near-zero probabilities (Low-risk,
-- long-tenure customers) don't produce absurdly large numbers — this cap only affects the
-- Low risk segment in practice, since High/Medium risk customers have high enough
-- churn_probability that 1/p is already well under 60.

-- =============================================================
-- 1. Per-customer CLV
-- =============================================================
SELECT
    cp.customer_id,
    rc.monthly_charges,
    cp.churn_probability,
    cp.risk_segment,
    ROUND(LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0))::NUMERIC, 1) AS expected_tenure_months,
    ROUND(
        (rc.monthly_charges * LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0)))::NUMERIC,
        2
    ) AS clv
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
ORDER BY clv DESC;

-- =============================================================
-- 2. CLV by risk segment (average and total)
-- =============================================================
SELECT
    cp.risk_segment,
    COUNT(*) AS customer_count,
    ROUND(AVG(rc.monthly_charges * LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0)))::NUMERIC, 2) AS avg_clv,
    ROUND(SUM(rc.monthly_charges * LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0)))::NUMERIC, 2) AS total_clv
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
GROUP BY cp.risk_segment
ORDER BY total_clv DESC;

-- =============================================================
-- 3. CLV at risk — total future revenue exposed by High risk customers
-- =============================================================
SELECT
    COUNT(*)                                                                                             AS high_risk_customers,
    ROUND(SUM(rc.monthly_charges * LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0)))::NUMERIC, 2)         AS total_clv_at_risk,
    ROUND(AVG(rc.monthly_charges * LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0)))::NUMERIC, 2)         AS avg_clv_at_risk
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
WHERE cp.risk_segment = 'High';

-- =============================================================
-- 4. Top 10 highest-CLV customers currently at risk (High or Medium)
-- =============================================================
SELECT
    cp.customer_id,
    rc.contract,
    rc.tenure,
    rc.monthly_charges,
    cp.churn_probability,
    cp.risk_segment,
    ROUND(
        (rc.monthly_charges * LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0)))::NUMERIC,
        2
    ) AS clv
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
WHERE cp.risk_segment IN ('High', 'Medium')
ORDER BY clv DESC
LIMIT 10;

-- =============================================================
-- 5. Revenue saved by segment if churn were fully prevented
--    (i.e., the CLV that segment currently represents — same shape as query 2's total_clv,
--     framed as the upside of a 100%-effective retention program)
-- =============================================================
SELECT
    cp.risk_segment,
    COUNT(*)                                                                                     AS customer_count,
    ROUND(SUM(rc.monthly_charges * LEAST(60, 1.0 / NULLIF(cp.churn_probability, 0)))::NUMERIC, 2) AS revenue_saved_if_prevented
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
WHERE cp.risk_segment IN ('High', 'Medium')
GROUP BY cp.risk_segment
ORDER BY revenue_saved_if_prevented DESC;
