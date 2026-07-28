"""Shared pipeline-run history tracking — every pipeline module (daily/weekly/model) logs its
runs here via record_pipeline_run(), so dashboard/pages/6_monitoring.py can show a unified
run-history table without querying MLflow directly.
"""

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("pipeline_tracking")


def record_pipeline_run(
    pipeline_name: str,
    status: str,
    rows_processed: int = 0,
    duration_seconds: float = 0.0,
    details: str = None,
) -> None:
    """Inserts one row into pipeline_runs. status should be 'success' or 'failed'."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO pipeline_runs "
                "(pipeline_name, status, rows_processed, duration_seconds, details) "
                "VALUES (:name, :status, :rows, :duration, :details)"
            ),
            {
                "name": pipeline_name,
                "status": status,
                "rows": rows_processed,
                "duration": duration_seconds,
                "details": details,
            },
        )
    log.info(
        "Recorded pipeline run: %s status=%s rows=%d duration=%.2fs",
        pipeline_name,
        status,
        rows_processed,
        duration_seconds,
    )


def get_pipeline_run_history(
    pipeline_name: str = None, limit: int = 20
) -> pd.DataFrame:
    """Returns the most recent pipeline_runs rows, optionally filtered to one pipeline_name."""
    engine = get_engine()
    query = "SELECT * FROM pipeline_runs"
    params = {"limit": limit}
    if pipeline_name:
        query += " WHERE pipeline_name = :name"
        params["name"] = pipeline_name
    query += " ORDER BY run_at DESC LIMIT :limit"

    return pd.read_sql(text(query), engine, params=params)
