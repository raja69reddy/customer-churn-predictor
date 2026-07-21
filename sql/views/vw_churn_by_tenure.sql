-- vw_churn_by_tenure: churn rate, avg charges, customer counts, and revenue at risk
-- broken down by tenure group (0-12, 12-24, 24-48, 48+ months).
--
-- churn_rate_pct / avg_monthly_charges / customer_count come from raw_customers (ground
-- truth). revenue_at_risk is monthly_charges summed over the High+Medium risk customers in
-- that tenure group, joined from the latest churn_predictions — it is 0 (not NULL) for a
-- tenure group with no High/Medium customers.
CREATE OR REPLACE VIEW vw_churn_by_tenure AS
WITH tenure_buckets AS (
    SELECT
        rc.customer_id,
        rc.churn,
        rc.monthly_charges,
        CASE
            WHEN rc.tenure <= 12 THEN '0-12'
            WHEN rc.tenure <= 24 THEN '12-24'
            WHEN rc.tenure <= 48 THEN '24-48'
            ELSE '48+'
        END AS tenure_group,
        cp.risk_segment
    FROM raw_customers rc
    LEFT JOIN churn_predictions cp ON cp.customer_id = rc.customer_id
)
SELECT
    tenure_group,
    COUNT(*)                                                        AS customer_count,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                                                AS churn_rate_pct,
    ROUND(AVG(monthly_charges)::NUMERIC, 2)                         AS avg_monthly_charges,
    ROUND(
        SUM(CASE WHEN risk_segment IN ('High', 'Medium') THEN monthly_charges ELSE 0 END)::NUMERIC,
        2
    )                                                                AS revenue_at_risk
FROM tenure_buckets
GROUP BY tenure_group
ORDER BY
    CASE tenure_group
        WHEN '0-12' THEN 1
        WHEN '12-24' THEN 2
        WHEN '24-48' THEN 3
        ELSE 4
    END;
