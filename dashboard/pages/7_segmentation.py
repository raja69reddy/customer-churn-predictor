"""Customer Segmentation page — Value x Risk 2D matrix, segment KPIs, and a downloadable
segment list.

src/analysis/customer_segmentation.py is pure pandas/SQL (no heavy ML deps), so unlike the
SHAP/what-if pages this whole page works the same way in Live and Demo mode — no
try/except ImportError gate needed, matching the dashboard/app.py::revenue_analysis pattern.
"""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dashboard import demo_mode  # noqa: E402
from dashboard.components.charts import segment_matrix_heatmap  # noqa: E402
from dashboard.components.metrics import display_kpi_row, format_number  # noqa: E402
from src.analysis.customer_segmentation import _load_customers  # noqa: E402
from src.analysis.customer_segmentation import create_2d_segments  # noqa: E402
from src.analysis.customer_segmentation import segment_by_risk  # noqa: E402
from src.analysis.customer_segmentation import segment_by_tenure  # noqa: E402
from src.analysis.customer_segmentation import segment_by_value  # noqa: E402

st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")

CACHE_TTL_SECONDS = 300


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_segmentation"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(
        f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_segmented_customers() -> pd.DataFrame:
    if demo_mode.is_demo_mode():
        df = demo_mode.get_demo_predictions()[
            [
                "customer_id",
                "churn_probability",
                "risk_segment",
                "tenure",
                "monthly_charges",
            ]
        ].copy()
    else:
        df = _load_customers()

    df = segment_by_value(df)
    df = segment_by_risk(df)
    df = segment_by_tenure(df)
    return df


def render_segment_characteristics(df: pd.DataFrame) -> None:
    st.markdown("#### Segment Characteristics")
    table = (
        df.groupby(["value_segment", "risk_segment"], observed=True)
        .agg(
            customers=("customer_id", "count"),
            avg_tenure=("tenure", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
            total_monthly_revenue=("monthly_charges", "sum"),
        )
        .reset_index()
    )
    table = table[table["customers"] > 0].sort_values(
        ["value_segment", "risk_segment"], ascending=[False, False]
    )
    table["avg_tenure"] = table["avg_tenure"].round(1)
    table["avg_monthly_charges"] = table["avg_monthly_charges"].round(2)
    table["avg_churn_probability"] = table["avg_churn_probability"].map(
        lambda p: f"{p:.1%}"
    )
    table["total_monthly_revenue"] = table["total_monthly_revenue"].round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    is_demo = demo_mode.is_demo_mode()
    if is_demo:
        st.warning(
            "🟡 **Demo Mode** — showing sample data (no live database connection)."
        )
    else:
        st.success("🟢 **Live Mode** — connected to the database.")

    render_cache_controls()

    st.title("Customer Segmentation")

    try:
        with st.spinner("Loading segmentation..."):
            df = load_segmented_customers()
    except Exception:
        st.error(
            "⚠️ Could not connect to the database. Please check your connection and try again."
        )
        return

    if df.empty:
        st.warning("No data available — no customers found.")
        return

    high_value_high_risk = int(
        ((df["value_segment"] == "High") & (df["risk_segment"] == "High")).sum()
    )
    display_kpi_row(
        [
            {
                "title": "Total Customers",
                "value": format_number(len(df)),
                "color": "#1f77b4",
            },
            {
                "title": "Total Monthly Revenue",
                "value": f"${df['monthly_charges'].sum():,.2f}",
                "color": "#2ca02c",
            },
            {
                "title": "High Value Customers",
                "value": format_number(int((df["value_segment"] == "High").sum())),
                "color": "#9467bd",
            },
            {
                "title": "High Value + High Risk (urgent)",
                "value": format_number(high_value_high_risk),
                "color": "#d62728",
            },
        ]
    )

    st.divider()
    matrix = create_2d_segments(df["value_segment"], df["risk_segment"])
    st.plotly_chart(segment_matrix_heatmap(matrix), use_container_width=True)

    st.divider()
    render_segment_characteristics(df)

    st.divider()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Segmentation List as CSV",
        data=csv_bytes,
        file_name="customer_segments.csv",
        mime="text/csv",
    )


main()
