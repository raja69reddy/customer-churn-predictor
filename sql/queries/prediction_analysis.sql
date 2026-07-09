-- prediction_analysis.sql — ad-hoc analysis queries against churn_predictions /
-- churn_predictions_history. Each block is independent and can be run on its own.
--
-- Note on "contact info": the Telco dataset behind raw_customers has no name/phone/email
-- fields. payment_method + phone_service are the closest available proxies for "how this
-- customer currently interacts with billing/service" and are included below in that spirit.
--
-- Drift/transition queries compare each customer's current churn_predictions row against
-- their most recent churn_predictions_history row (archived automatically by
-- batch_scorer.py/incremental_scorer.py right before a prediction is overwritten). If a
-- customer has never been re-scored, they simply won't appear in the drift results.

-- =============================================================
-- 1. Prediction score distribution by decile
-- =============================================================
WITH deciled AS (
    SELECT
        churn_probability,
        NTILE(10) OVER (ORDER BY churn_probability) AS decile
    FROM churn_predictions
)
SELECT
    decile,
    COUNT(*)                                    AS customer_count,
    ROUND(MIN(churn_probability)::NUMERIC, 4)   AS min_score,
    ROUND(MAX(churn_probability)::NUMERIC, 4)   AS max_score,
    ROUND(AVG(churn_probability)::NUMERIC, 4)   AS avg_score
FROM deciled
GROUP BY decile
ORDER BY decile;

-- =============================================================
-- 2. High risk customer details with contact info
-- =============================================================
SELECT
    cp.customer_id,
    rc.gender,
    rc.senior_citizen,
    rc.phone_service,
    rc.payment_method,
    rc.contract,
    rc.tenure,
    rc.monthly_charges,
    cp.churn_probability,
    cp.risk_segment
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
WHERE cp.risk_segment = 'High'
ORDER BY cp.churn_probability DESC;

-- =============================================================
-- 3. Score change from last prediction (drift)
-- =============================================================
WITH latest_history AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        churn_probability AS prev_probability,
        risk_segment       AS prev_segment,
        archived_at
    FROM churn_predictions_history
    ORDER BY customer_id, archived_at DESC
)
SELECT
    cp.customer_id,
    lh.prev_probability,
    cp.churn_probability                                              AS current_probability,
    ROUND((cp.churn_probability - lh.prev_probability)::NUMERIC, 4)   AS score_change,
    lh.prev_segment,
    cp.risk_segment                                                    AS current_segment
FROM churn_predictions cp
JOIN latest_history lh ON lh.customer_id = cp.customer_id
ORDER BY ABS(cp.churn_probability - lh.prev_probability) DESC;

-- =============================================================
-- 4. Customers who moved from Medium to High risk
-- =============================================================
WITH latest_history AS (
    SELECT DISTINCT ON (customer_id)
        customer_id, risk_segment AS prev_segment, archived_at
    FROM churn_predictions_history
    ORDER BY customer_id, archived_at DESC
)
SELECT
    cp.customer_id,
    cp.churn_probability,
    lh.prev_segment,
    cp.risk_segment AS current_segment
FROM churn_predictions cp
JOIN latest_history lh ON lh.customer_id = cp.customer_id
WHERE lh.prev_segment = 'Medium' AND cp.risk_segment = 'High'
ORDER BY cp.churn_probability DESC;

-- =============================================================
-- 5. Customers who moved from High to Medium risk
-- =============================================================
WITH latest_history AS (
    SELECT DISTINCT ON (customer_id)
        customer_id, risk_segment AS prev_segment, archived_at
    FROM churn_predictions_history
    ORDER BY customer_id, archived_at DESC
)
SELECT
    cp.customer_id,
    cp.churn_probability,
    lh.prev_segment,
    cp.risk_segment AS current_segment
FROM churn_predictions cp
JOIN latest_history lh ON lh.customer_id = cp.customer_id
WHERE lh.prev_segment = 'High' AND cp.risk_segment = 'Medium'
ORDER BY cp.churn_probability DESC;

-- =============================================================
-- 6. Revenue at risk by risk segment
-- =============================================================
SELECT
    cp.risk_segment,
    COUNT(*)                                        AS customer_count,
    ROUND(SUM(rc.monthly_charges)::NUMERIC, 2)      AS monthly_revenue_at_risk,
    ROUND(AVG(rc.monthly_charges)::NUMERIC, 2)      AS avg_monthly_charges
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
GROUP BY cp.risk_segment
ORDER BY monthly_revenue_at_risk DESC;
