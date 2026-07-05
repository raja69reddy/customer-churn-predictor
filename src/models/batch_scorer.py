"""Batch-scores every customer in raw_customers for churn risk and logs results to PostgreSQL."""
import pandas as pd
from sqlalchemy import text

from src.models.predict import ChurnPredictor
from src.utils.db import get_engine


def run() -> None:
    predictor = ChurnPredictor()
    predictor.load_best_model()
    print(f"Using best model: {predictor.model_name}")

    engine = get_engine()
    customers = pd.read_sql("SELECT * FROM raw_customers", engine)
    print(f"Loaded {len(customers)} customers from raw_customers")

    scored = predictor.predict_batch(customers)
    print(f"Scored {len(scored)} customers")

    distribution = scored["risk_segment"].value_counts()
    high = int(distribution.get("High", 0))
    medium = int(distribution.get("Medium", 0))
    low = int(distribution.get("Low", 0))
    print(f"\nSummary: {high} high risk, {medium} medium risk, {low} low risk")

    top10 = scored.sort_values("churn_probability", ascending=False).head(10)
    print("\nGenerating GPT explanations for top 10 highest-risk customers...")
    for _, row in top10.iterrows():
        customer_dict = row.to_dict()
        prediction = {"churn_probability": row["churn_probability"], "risk_segment": row["risk_segment"]}
        explanation = predictor.explain_with_gpt(customer_dict, prediction)
        print(f"    {row['customer_id']} (p={row['churn_probability']:.3f}, {row['risk_segment']}): {explanation}")

    to_save = scored[["customer_id", "churn_probability", "risk_segment"]].copy()
    to_save["model_version"] = predictor.model_name

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE churn_predictions"))
        to_save.to_sql("churn_predictions", conn, if_exists="append", index=False)

    print(f"\nSaved {len(to_save)} predictions to churn_predictions table")


if __name__ == "__main__":
    run()
