-- vw_risk_segments: risk segment summary stats plus combined revenue-at-risk row
CREATE OR REPLACE VIEW vw_risk_segments AS
WITH segment_stats AS (
    SELECT
        cp.risk_segment,
        COUNT(*)                                                      AS customer_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)            AS pct_of_total,
        ROUND(AVG(rc.monthly_charges)::NUMERIC, 2)                    AS avg_monthly_charges,
        ROUND(AVG(rc.tenure)::NUMERIC, 2)                             AS avg_tenure,
        MODE() WITHIN GROUP (ORDER BY rc.contract)                    AS most_common_contract,
        ROUND(SUM(rc.monthly_charges)::NUMERIC, 2)                    AS total_monthly_revenue
    FROM churn_predictions cp
    JOIN raw_customers rc ON rc.customer_id = cp.customer_id
    GROUP BY cp.risk_segment
)
SELECT
    risk_segment,
    customer_count,
    pct_of_total,
    avg_monthly_charges,
    avg_tenure,
    most_common_contract,
    total_monthly_revenue,
    CASE WHEN risk_segment IN ('High', 'Medium') THEN total_monthly_revenue ELSE NULL END AS revenue_at_risk
FROM segment_stats

UNION ALL

SELECT
    'High+Medium (At Risk)'                                    AS risk_segment,
    SUM(customer_count)                                        AS customer_count,
    ROUND(SUM(pct_of_total), 2)                                AS pct_of_total,
    ROUND(SUM(avg_monthly_charges * customer_count) / NULLIF(SUM(customer_count), 0), 2) AS avg_monthly_charges,
    ROUND(SUM(avg_tenure * customer_count) / NULLIF(SUM(customer_count), 0), 2)          AS avg_tenure,
    NULL                                                        AS most_common_contract,
    SUM(total_monthly_revenue)                                 AS total_monthly_revenue,
    SUM(total_monthly_revenue)                                 AS revenue_at_risk
FROM segment_stats
WHERE risk_segment IN ('High', 'Medium');
