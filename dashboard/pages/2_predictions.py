"""Churn Predictions page — risk segment breakdown and searchable high-risk customer table."""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dashboard.components.charts import (  # noqa: E402
    probability_histogram,
    risk_segment_bar,
)
from dashboard.components.metrics import (  # noqa: E402
    display_kpi_row,
    format_number,
    format_percentage,
)
from src.utils.db import get_engine  # noqa: E402

st.set_page_config(page_title="Churn Predictions", page_icon="🔮", layout="wide")

CACHE_TTL_SECONDS = 300


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_predictions"):
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
def load_predictions(filters: dict) -> pd.DataFrame:
    engine = get_engine()
    query = (
        "SELECT * FROM vw_churn_predictions "
        "WHERE tenure BETWEEN %(tenure_min)s AND %(tenure_max)s"
    )
    params = {
        "tenure_min": filters["tenure_range"][0],
        "tenure_max": filters["tenure_range"][1],
    }

    if filters["risk_segment"] != "All":
        query += " AND risk_segment = %(risk_segment)s"
        params["risk_segment"] = filters["risk_segment"]
    if filters["contract"] != "All":
        query += " AND contract = %(contract)s"
        params["contract"] = filters["contract"]

    return pd.read_sql(query, engine, params=params)


def main() -> None:
    st.title("Churn Predictions")

    filters = render_sidebar_filters()
    try:
        with st.spinner("Loading predictions..."):
            predictions = load_predictions(filters)
    except Exception:
        st.error(
            "⚠️ Could not connect to the database. Please check your connection and try again."
        )
        return

    if predictions.empty:
        st.warning(
            "No data available for the selected filters. Try widening your filter selection."
        )
        return

    counts = predictions["risk_segment"].value_counts()
    avg_probability = predictions["churn_probability"].mean() if len(predictions) else 0

    display_kpi_row(
        [
            {
                "title": "High Risk Count",
                "value": format_number(int(counts.get("High", 0))),
                "color": "#d62728",
            },
            {
                "title": "Medium Risk Count",
                "value": format_number(int(counts.get("Medium", 0))),
                "color": "#ff7f0e",
            },
            {
                "title": "Low Risk Count",
                "value": format_number(int(counts.get("Low", 0))),
                "color": "#2ca02c",
            },
            {
                "title": "Avg Churn Probability",
                "value": format_percentage(avg_probability),
                "color": "#1f77b4",
            },
        ]
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(risk_segment_bar(predictions), use_container_width=True)
    with col2:
        st.plotly_chart(probability_histogram(predictions), use_container_width=True)

    st.divider()
    st.subheader("High Risk Customers")

    search_id = st.text_input("Search by Customer ID")

    high_risk = predictions[predictions["risk_segment"] == "High"][
        [
            "customer_id",
            "churn_probability",
            "risk_segment",
            "contract",
            "tenure",
            "monthly_charges",
            "predicted_at",
        ]
    ].rename(columns={"churn_probability": "probability", "risk_segment": "segment"})
    high_risk = high_risk.sort_values("probability", ascending=False)

    if search_id:
        high_risk = high_risk[
            high_risk["customer_id"].str.contains(search_id, case=False, na=False)
        ]

    st.dataframe(high_risk, use_container_width=True, hide_index=True)

    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download All Predictions as CSV",
        data=csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


main()
