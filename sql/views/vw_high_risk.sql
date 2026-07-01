-- vw_high_risk: customers scored against engineered risk features, with a High/Medium/Low segment
CREATE OR REPLACE VIEW vw_high_risk AS
SELECT
    rc.customer_id,
    rc.gender,
    rc.senior_citizen,
    rc.tenure,
    rc.contract,
    rc.internet_service,
    rc.online_security,
    rc.tech_support,
    rc.monthly_charges,
    rc.total_charges,
    rc.churn,
    rc.payment_method,
    pc.contract_risk_score,
    pc.payment_risk_score,
    pc.services_count,
    (
        (pc.contract_risk_score >= 3)::INT
        + (pc.payment_risk_score >= 2)::INT
        + (pc.services_count <= 2)::INT
        + (rc.tenure <= 12)::INT
    )                                              AS risk_flags_count,
    CASE
        WHEN (
            (pc.contract_risk_score >= 3)::INT
            + (pc.payment_risk_score >= 2)::INT
            + (pc.services_count <= 2)::INT
            + (rc.tenure <= 12)::INT
        ) >= 3 THEN 'High'
        WHEN (
            (pc.contract_risk_score >= 3)::INT
            + (pc.payment_risk_score >= 2)::INT
            + (pc.services_count <= 2)::INT
            + (rc.tenure <= 12)::INT
        ) = 2 THEN 'Medium'
        ELSE 'Low'
    END                                            AS risk_segment
FROM raw_customers rc
JOIN processed_customers pc ON rc.customer_id = pc.customer_id;
