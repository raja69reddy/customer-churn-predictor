"""Streamlit dashboard — main entry point."""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.db import get_engine

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
)


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
    risk_segment = st.sidebar.selectbox("Risk Segment", ["All", "High", "Medium", "Low"])
    contract = st.sidebar.selectbox(
        "Contract Type", ["All", "Month-to-month", "One year", "Two year"]
    )
    tenure_range = st.sidebar.slider("Tenure Range (months)", 0, 72, (0, 72))

    return {"risk_segment": risk_segment, "contract": contract, "tenure_range": tenure_range}


def load_project_stats() -> dict:
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


def main() -> None:
    filters = render_sidebar_filters()
    st.session_state["filters"] = filters

    st.title("🔮 Customer Churn Predictor")
    st.markdown(
        "Welcome! This dashboard summarizes churn risk across the customer base, lets you "
        "look up individual customers, compare model performance, and build retention "
        "outreach lists. Use the page navigation in the sidebar to explore."
    )

    stats = load_project_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers Scored", f"{stats['total_scored']:,}")
    col2.metric("Overall Churn Rate", f"{stats['churn_rate']:.1f}%")
    col3.metric("High Risk Customers", f"{stats['high_risk_count']:,}")
    col4.metric(
        "Model AUC Score",
        f"{stats['model_auc']:.4f}" if stats["model_auc"] is not None else "N/A",
    )


if __name__ == "__main__":
    main()
