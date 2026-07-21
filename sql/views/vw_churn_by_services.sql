-- vw_churn_by_services: churn rate/distribution/avg charges by services_count, plus the
-- most common add-on service combinations among churners.
--
-- Two sections with different populated columns — filter on `section` first:
--   'services_count_summary'    -> segment = services_count (0-8), churn_rate_pct populated
--   'top_combinations_churners' -> segment = a service-combo string, pct_of_churners populated
--                                   (top 10 combos by count, pct is share of ALL churners,
--                                   not just the top 10)
CREATE OR REPLACE VIEW vw_churn_by_services AS

WITH services AS (
    SELECT
        rc.*,
        (rc.phone_service = 'Yes')::INT
        + (rc.internet_service <> 'No')::INT
        + (rc.online_security = 'Yes')::INT
        + (rc.online_backup = 'Yes')::INT
        + (rc.device_protection = 'Yes')::INT
        + (rc.tech_support = 'Yes')::INT
        + (rc.streaming_tv = 'Yes')::INT
        + (rc.streaming_movies = 'Yes')::INT AS services_count
    FROM raw_customers rc
)

-- SECTION 1: churn rate, customer counts, and avg charges by services_count
SELECT
    'services_count_summary'                                     AS section,
    services_count::TEXT                                         AS segment,
    COUNT(*)                                                     AS customer_count,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                                             AS churn_rate_pct,
    ROUND(AVG(monthly_charges)::NUMERIC, 2)                      AS avg_monthly_charges,
    NULL::NUMERIC                                                AS pct_of_churners
FROM services
GROUP BY services_count

UNION ALL

-- SECTION 2: top 10 most common add-on service combinations among churners
(
    SELECT
        'top_combinations_churners'                              AS section,
        CONCAT(
            'security=', online_security, ',',
            'backup=', online_backup, ',',
            'device_protect=', device_protection, ',',
            'tech_support=', tech_support, ',',
            'streaming_tv=', streaming_tv, ',',
            'streaming_movies=', streaming_movies
        )                                                         AS segment,
        COUNT(*)                                                  AS customer_count,
        NULL::NUMERIC                                             AS churn_rate_pct,
        ROUND(AVG(monthly_charges)::NUMERIC, 2)                   AS avg_monthly_charges,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)        AS pct_of_churners
    FROM services
    WHERE churn = 'Yes'
    GROUP BY
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies
    ORDER BY COUNT(*) DESC
    LIMIT 10
);
