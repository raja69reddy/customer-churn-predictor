-- vw_prediction_summary: segment-level summary (counts, avg probability, revenue at risk,
-- trend vs previous snapshot) plus customer-level sections (top 20 highest risk, and
-- customers whose score improved/worsened vs their most recent churn_predictions_history
-- entry). Different "section" values have different populated columns — see comments below.
CREATE OR REPLACE VIEW vw_prediction_summary AS

WITH latest_history AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        churn_probability AS prev_probability,
        risk_segment       AS prev_segment
    FROM churn_predictions_history
    ORDER BY customer_id, archived_at DESC
),
prev_segment_counts AS (
    SELECT prev_segment AS risk_segment, COUNT(*) AS prev_count
    FROM latest_history
    GROUP BY prev_segment
)

-- SECTION 1: total customers / avg churn probability / revenue at risk / trend, per segment
SELECT
    'segment_summary'                                  AS section,
    cp.risk_segment                                     AS segment,
    NULL::VARCHAR                                        AS customer_id,
    COUNT(*)::BIGINT                                     AS customer_count,
    ROUND(AVG(cp.churn_probability)::NUMERIC, 4)        AS avg_churn_probability,
    ROUND(SUM(rc.monthly_charges)::NUMERIC, 2)          AS revenue_at_risk,
    (COUNT(*) - COALESCE(MAX(psc.prev_count), 0))::BIGINT AS trend_vs_previous,
    NULL::FLOAT                                          AS churn_probability,
    NULL::FLOAT                                          AS prev_probability,
    NULL::VARCHAR                                        AS score_direction
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
LEFT JOIN prev_segment_counts psc ON psc.risk_segment = cp.risk_segment
GROUP BY cp.risk_segment

UNION ALL

-- SECTION 2: top 20 highest risk customers
(
    SELECT
        'top_20_highest_risk' AS section,
        cp.risk_segment        AS segment,
        cp.customer_id,
        NULL::BIGINT            AS customer_count,
        NULL::NUMERIC           AS avg_churn_probability,
        NULL::NUMERIC           AS revenue_at_risk,
        NULL::BIGINT            AS trend_vs_previous,
        cp.churn_probability::FLOAT AS churn_probability,
        NULL::FLOAT             AS prev_probability,
        NULL::VARCHAR           AS score_direction
    FROM churn_predictions cp
    ORDER BY cp.churn_probability DESC
    LIMIT 20
)

UNION ALL

-- SECTION 3: customers whose churn probability improved (decreased) since their last prediction
SELECT
    'improving_scores'        AS section,
    cp.risk_segment            AS segment,
    cp.customer_id,
    NULL::BIGINT                AS customer_count,
    NULL::NUMERIC                AS avg_churn_probability,
    NULL::NUMERIC                AS revenue_at_risk,
    NULL::BIGINT                 AS trend_vs_previous,
    cp.churn_probability::FLOAT AS churn_probability,
    lh.prev_probability::FLOAT   AS prev_probability,
    'improving'                  AS score_direction
FROM churn_predictions cp
JOIN latest_history lh ON lh.customer_id = cp.customer_id
WHERE cp.churn_probability < lh.prev_probability

UNION ALL

-- SECTION 4: customers whose churn probability worsened (increased) since their last prediction
SELECT
    'worsening_scores'        AS section,
    cp.risk_segment            AS segment,
    cp.customer_id,
    NULL::BIGINT                AS customer_count,
    NULL::NUMERIC                AS avg_churn_probability,
    NULL::NUMERIC                AS revenue_at_risk,
    NULL::BIGINT                 AS trend_vs_previous,
    cp.churn_probability::FLOAT AS churn_probability,
    lh.prev_probability::FLOAT   AS prev_probability,
    'worsening'                  AS score_direction
FROM churn_predictions cp
JOIN latest_history lh ON lh.customer_id = cp.customer_id
WHERE cp.churn_probability > lh.prev_probability;
