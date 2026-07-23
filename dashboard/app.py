"""Streamlit dashboard — main entry point."""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import demo_mode  # noqa: E402
from src.analysis.revenue_analysis import (  # noqa: E402
    calculate_clv_by_segment,
    calculate_revenue_at_risk,
    estimate_retention_savings,
)
from src.utils.db import get_engine  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
)

CACHE_TTL_SECONDS = 300


def render_sidebar_filters() -> dict:
    st.sidebar.title("🔮 Customer Churn Predictor")
    st.sidebar.caption(
        "End-to-end churn prediction: explore churn drivers, review individual customer "
        "risk, inspect model performance, and target retention outreach."
    )

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown(
        "- Overview\n"
        "- Predictions\n"
        "- Customer Lookup\n"
        "- Model Performance\n"
        "- Retention Targeting\n\n"
        "(use the page list above the navigation, or the sidebar page switcher)"
    )

    st.sidebar.markdown("### Filters")
    risk_segment = st.sidebar.selectbox(
        "Risk Segment", ["All", "High", "Medium", "Low"], key="risk_segment"
    )
    contract = st.sidebar.selectbox(
        "Contract Type",
        ["All", "Month-to-month", "One year", "Two year"],
        key="contract",
    )
    tenure_range = st.sidebar.slider(
        "Tenure Range (months)", 0, 72, (0, 72), key="tenure_range"
    )

    render_cache_controls()

    return {
        "risk_segment": risk_segment,
        "contract": contract,
        "tenure_range": tenure_range,
    }


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_app"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(
        f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


def render_mode_badge() -> None:
    if demo_mode.is_demo_mode():
        st.warning(
            "🟡 **Demo Mode** — showing sample data (no live database connection)."
        )
    else:
        st.success("🟢 **Live Mode** — connected to the database.")


def _demo_project_stats() -> dict:
    predictions = demo_mode.get_demo_predictions()
    total_scored = len(predictions)
    high_risk_count = int((predictions["risk_segment"] == "High").sum())
    churn_rate = (predictions["churn"] == "Yes").mean() * 100

    metrics = demo_mode.get_demo_model_metrics()
    model_auc = float(metrics["auc_score"].max()) if not metrics.empty else None

    return {
        "total_scored": total_scored,
        "churn_rate": churn_rate,
        "high_risk_count": high_risk_count,
        "model_auc": model_auc,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_project_stats() -> dict:
    if demo_mode.is_demo_mode():
        return _demo_project_stats()

    engine = get_engine()

    predictions = pd.read_sql(
        "SELECT churn_probability, risk_segment FROM churn_predictions", engine
    )
    total_scored = len(predictions)
    high_risk_count = int((predictions["risk_segment"] == "High").sum())

    raw = pd.read_sql("SELECT churn FROM raw_customers", engine)
    churn_rate = (raw["churn"] == "Yes").mean() * 100

    model = pd.read_sql(
        "SELECT auc_score FROM model_registry WHERE is_active = TRUE LIMIT 1", engine
    )
    model_auc = float(model.iloc[0]["auc_score"]) if not model.empty else None

    return {
        "total_scored": total_scored,
        "churn_rate": churn_rate,
        "high_risk_count": high_risk_count,
        "model_auc": model_auc,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_revenue_metrics() -> dict:
    """Revenue-at-risk / CLV / ROI metrics, via src/analysis/revenue_analysis.py.

    In demo mode, the demo predictions CSV is passed in directly as the `df` argument each
    function already accepts — it has the same columns _load_predictions() would return from
    a live query, so no separate demo-mode logic needs to be duplicated here.
    """
    demo_df = demo_mode.get_demo_predictions() if demo_mode.is_demo_mode() else None

    return {
        "revenue_at_risk": calculate_revenue_at_risk(demo_df),
        "clv_by_segment": calculate_clv_by_segment(demo_df),
        "roi": estimate_retention_savings(df=demo_df),
    }


def render_revenue_metrics() -> None:
    st.divider()
    st.subheader("💰 Revenue at Risk")

    try:
        metrics = load_revenue_metrics()
    except Exception:
        st.error(
            "⚠️ Could not load revenue metrics. Please check your connection and try again."
        )
        return

    roi = metrics["roi"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue at Risk", f"${metrics['revenue_at_risk']:,.2f}/mo")
    col2.metric("High Risk Customer Count", f"{roi['customers_targeted']:,}")
    col3.metric("Estimated Savings if Retained", f"${roi['revenue_recovered']:,.2f}/yr")
    st.caption(
        f"Savings assume a ${roi['campaign_cost_per_customer']:.0f}/customer retention "
        f"campaign at a {roi['success_rate']:.0%} success rate, annualized."
    )

    st.markdown("#### Avg CLV by Segment")
    st.dataframe(metrics["clv_by_segment"], use_container_width=True, hide_index=True)


def main() -> None:
    render_mode_badge()

    filters = render_sidebar_filters()
    st.session_state["filters"] = filters

    st.title("🔮 Customer Churn Predictor")
    st.markdown(
        "Welcome! This dashboard summarizes churn risk across the customer base, lets you "
        "look up individual customers, compare model performance, and build retention "
        "outreach lists. Use the page navigation in the sidebar to explore."
    )

    try:
        stats = load_project_stats()
    except Exception:
        st.error(
            "⚠️ Could not connect to the database. Please check your connection "
            "(is PostgreSQL running and is .env configured?) and try again."
        )
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers Scored", f"{stats['total_scored']:,}")
    col2.metric("Overall Churn Rate", f"{stats['churn_rate']:.1f}%")
    col3.metric("High Risk Customers", f"{stats['high_risk_count']:,}")
    col4.metric(
        "Model AUC Score",
        f"{stats['model_auc']:.4f}" if stats["model_auc"] is not None else "N/A",
    )

    render_revenue_metrics()


if __name__ == "__main__":
    main()
