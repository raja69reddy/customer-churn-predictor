-- vw_high_risk: customers matching high-churn-risk profile
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
    rc.payment_method
FROM raw_customers rc
WHERE
    rc.contract        = 'Month-to-month'
    AND rc.tenure      < 12
    AND rc.monthly_charges > (SELECT AVG(monthly_charges) FROM raw_customers)
    AND rc.online_security IN ('No', 'No internet service')
    AND rc.tech_support    IN ('No', 'No internet service');
