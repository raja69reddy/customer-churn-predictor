"""Ensures model_registry.is_active correctly flags the best model by AUC score."""
import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine


def run() -> None:
    engine = get_engine()

    with engine.begin() as conn:
        registry = pd.read_sql("SELECT * FROM model_registry", conn)
        if registry.empty:
            raise RuntimeError("model_registry is empty — run src.models.run_training first.")

        best_id = int(registry.loc[registry["auc_score"].idxmax(), "id"])

        conn.execute(text("UPDATE model_registry SET is_active = FALSE"))
        conn.execute(text("UPDATE model_registry SET is_active = TRUE WHERE id = :best_id"), {"best_id": best_id})

    final = pd.read_sql(
        "SELECT model_name, model_version, accuracy, auc_score, f1_score, trained_at, is_active "
        "FROM model_registry ORDER BY auc_score DESC",
        engine,
    )
    print("Final model_registry:")
    print(final.to_string(index=False))


if __name__ == "__main__":
    run()
