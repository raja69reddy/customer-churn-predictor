"""Demo-mode data helpers — let the dashboard run without a live PostgreSQL connection.

Streamlit Community Cloud deployments won't have access to the local Postgres instance
used for development, so every page falls back to the static CSVs in ``data/demo/``
(committed to git, unlike ``data/processed/`` and ``data/raw/``) whenever a live DB
connection can't be established.
"""

import os

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.utils.db import get_engine

DEMO_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo"
)

DEMO_FILES = {
    "predictions": "demo_predictions.csv",
    "model_metrics": "demo_model_metrics.csv",
    "risk_summary": "demo_risk_summary.csv",
    "churn_overview": "demo_churn_overview.csv",
}


@st.cache_resource(ttl=300)
def is_demo_mode() -> bool:
    """Returns True if a live PostgreSQL connection cannot be established."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return False
    except Exception:
        return True


def load_demo_data(table_name: str) -> pd.DataFrame:
    """Loads a demo CSV by logical name (predictions, model_metrics, risk_summary,
    churn_overview) from data/demo/."""
    filename = DEMO_FILES.get(table_name, f"demo_{table_name}.csv")
    path = os.path.join(DEMO_DATA_DIR, filename)
    return pd.read_csv(path)


def get_demo_ga4_data() -> pd.DataFrame:
    """Returns a sample GA4-style web analytics dataframe (sessions/conversions by day).

    Not wired into any dashboard page today — provided for future web-analytics
    integration so the demo-mode surface area matches what a production deployment
    would eventually need.
    """
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq="D")
    sessions = [820 + (i * 7) % 260 for i in range(30)]
    conversions = [max(1, round(s * 0.032)) for s in sessions]
    return pd.DataFrame(
        {
            "date": dates,
            "sessions": sessions,
            "conversions": conversions,
            "bounce_rate_pct": [round(38 + (i % 12) * 0.9, 1) for i in range(30)],
        }
    )


def get_demo_predictions() -> pd.DataFrame:
    """Returns the 100 sample customer predictions used in demo mode."""
    return load_demo_data("predictions")


def get_demo_model_metrics() -> pd.DataFrame:
    """Returns sample model comparison metrics (mirrors vw_model_performance)."""
    return load_demo_data("model_metrics")


def get_demo_risk_summary() -> pd.DataFrame:
    """Returns sample per-segment risk summary counts (mirrors vw_risk_segments)."""
    return load_demo_data("risk_summary")


def get_demo_churn_overview() -> pd.DataFrame:
    """Returns sample churn-by-dimension overview stats (mirrors vw_churn_overview)."""
    return load_demo_data("churn_overview")
