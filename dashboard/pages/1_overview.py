"""Churn Overview page — headline KPIs and churn distribution charts."""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dashboard.components.charts import churn_distribution_pie
from dashboard.components.metrics import (
    display_kpi_row,
    format_number,
    format_percentage,
)
from src.utils.db import get_engine

st.set_page_config(page_title="Churn Overview", page_icon="🔮", layout="wide")

CACHE_TTL_SECONDS = 300


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_overview"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(
        f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


def render_sidebar_filters() -> dict:
    st.sidebar.title("Filters")
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


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_filtered_customers(filters: dict) -> pd.DataFrame:
    engine = get_engine()
    query = """
        SELECT rc.customer_id, rc.churn, rc.contract, rc.tenure, rc.monthly_charges,
               cp.risk_segment, cp.churn_probability
        FROM raw_customers rc
        LEFT JOIN churn_predictions cp ON cp.customer_id = rc.customer_id
        WHERE rc.tenure BETWEEN %(tenure_min)s AND %(tenure_max)s
    """
    params = {
        "tenure_min": filters["tenure_range"][0],
        "tenure_max": filters["tenure_range"][1],
    }

    if filters["risk_segment"] != "All":
        query += " AND cp.risk_segment = %(risk_segment)s"
        params["risk_segment"] = filters["risk_segment"]
    if filters["contract"] != "All":
        query += " AND rc.contract = %(contract)s"
        params["contract"] = filters["contract"]

    return pd.read_sql(query, engine, params=params)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_churn_overview() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM vw_churn_overview", engine)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_model_accuracy() -> float:
    engine = get_engine()
    result = pd.read_sql(
        "SELECT accuracy FROM model_registry WHERE is_active = TRUE LIMIT 1", engine
    )
    return float(result.iloc[0]["accuracy"]) if not result.empty else None


def main() -> None:
    st.title("Churn Overview")

    filters = render_sidebar_filters()
    try:
        with st.spinner("Loading overview..."):
            customers = load_filtered_customers(filters)
            overview = load_churn_overview()
            model_accuracy = load_model_accuracy()
    except Exception:
        st.error(
            "⚠️ Could not connect to the database. Please check your connection and try again."
        )
        return

    if customers.empty:
        st.warning(
            "No data available for the selected filters. Try widening your filter selection."
        )
        return

    churn_rate = (customers["churn"] == "Yes").mean() if len(customers) else 0
    high_risk_count = int((customers["risk_segment"] == "High").sum())
    revenue_at_risk = customers.loc[
        customers["risk_segment"].isin(["High", "Medium"]), "monthly_charges"
    ].sum()

    display_kpi_row(
        [
            {
                "title": "Overall Churn Rate",
                "value": format_percentage(churn_rate),
                "color": "#d62728",
            },
            {
                "title": "Total High Risk Customers",
                "value": format_number(high_risk_count),
                "color": "#ff7f0e",
            },
            {
                "title": "Revenue at Risk",
                "value": f"${revenue_at_risk:,.2f}",
                "color": "#9467bd",
            },
            {
                "title": "Model Accuracy",
                "value": (
                    format_percentage(model_accuracy)
                    if model_accuracy is not None
                    else "N/A"
                ),
                "color": "#2ca02c",
            },
        ]
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(churn_distribution_pie(customers), use_container_width=True)

    contract_data = overview[overview["dimension"] == "contract"]
    with col2:
        import plotly.express as px

        fig = px.bar(
            contract_data,
            x="segment",
            y="churn_rate_pct",
            title="Churn Rate by Contract Type",
            color_discrete_sequence=["#1f77b4"],
            text="churn_rate_pct",
        )
        fig.update_layout(xaxis_title="Contract Type", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    tenure_data = overview[overview["dimension"] == "tenure_group"]
    if not tenure_data.empty:
        import plotly.express as px

        fig2 = px.bar(
            tenure_data,
            x="segment",
            y="churn_rate_pct",
            title="Churn Rate by Tenure Group",
            color_discrete_sequence=["#1f77b4"],
            text="churn_rate_pct",
        )
        fig2.update_layout(xaxis_title="Tenure Group", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)


main()
