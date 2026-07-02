-- vw_model_performance: all trained model versions ranked by AUC, with the active/best model highlighted
CREATE OR REPLACE VIEW vw_model_performance AS
SELECT
    model_name,
    model_version,
    ROUND(accuracy::NUMERIC, 4)   AS accuracy,
    ROUND(auc_score::NUMERIC, 4)  AS auc_score,
    ROUND(f1_score::NUMERIC, 4)   AS f1_score,
    trained_at,
    is_active,
    CASE WHEN is_active THEN 'Best Model' ELSE '' END AS highlight,
    RANK() OVER (ORDER BY auc_score DESC) AS auc_rank
FROM model_registry
ORDER BY auc_score DESC;
