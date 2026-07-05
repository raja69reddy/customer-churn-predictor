-- vw_churn_predictions: latest churn predictions joined with customer details
CREATE OR REPLACE VIEW vw_churn_predictions AS
SELECT
    cp.customer_id,
    rc.gender,
    rc.senior_citizen,
    rc.tenure,
    rc.contract,
    rc.internet_service,
    rc.payment_method,
    rc.monthly_charges,
    rc.total_charges,
    rc.churn                               AS actual_churn,
    cp.churn_probability,
    cp.risk_segment,
    cp.model_version,
    cp.predicted_at,
    EXTRACT(DAY FROM NOW() - cp.predicted_at)::INT AS days_since_prediction
FROM churn_predictions cp
JOIN raw_customers rc ON rc.customer_id = cp.customer_id
ORDER BY cp.churn_probability DESC;
