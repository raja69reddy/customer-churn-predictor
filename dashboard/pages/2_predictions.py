"""Churn Predictions page — risk segment breakdown and searchable high-risk customer table."""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dashboard import demo_mode  # noqa: E402
from dashboard.components.charts import (  # noqa: E402
    RISK_COLOR_MAP,
    probability_gauge,
    probability_histogram,
    risk_segment_bar,
)
from dashboard.components.metrics import (  # noqa: E402
    display_kpi_row,
    format_number,
    format_percentage,
    get_risk_color,
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


def _filter_demo_predictions(filters: dict) -> pd.DataFrame:
    df = demo_mode.get_demo_predictions()
    df = df[
        (df["tenure"] >= filters["tenure_range"][0])
        & (df["tenure"] <= filters["tenure_range"][1])
    ]
    if filters["risk_segment"] != "All":
        df = df[df["risk_segment"] == filters["risk_segment"]]
    if filters["contract"] != "All":
        df = df[df["contract"] == filters["contract"]]
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_predictions(filters: dict) -> pd.DataFrame:
    if demo_mode.is_demo_mode():
        return _filter_demo_predictions(filters)

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


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_risk_segment_trend() -> pd.DataFrame:
    """Returns risk segment counts per day (long format: date, risk_segment, count).

    Live mode aggregates the real churn_predictions_history archive by day (see Day 9).
    Demo mode has no real history, so it generates an illustrative 14-day trend around
    the current demo segment counts.
    """
    if demo_mode.is_demo_mode():
        demo = demo_mode.get_demo_predictions()
        base_counts = demo["risk_segment"].value_counts().to_dict()
        dates = pd.date_range(
            end=pd.Timestamp.today().normalize(), periods=14, freq="D"
        )
        rows = []
        for i, d in enumerate(dates):
            wobble = ((i * 37) % 11 - 5) / 20
            for segment in ["High", "Medium", "Low"]:
                base = base_counts.get(segment, 0)
                rows.append(
                    {
                        "date": d,
                        "risk_segment": segment,
                        "count": max(0, round(base * (1 + wobble))),
                    }
                )
        return pd.DataFrame(rows)

    engine = get_engine()
    query = """
        WITH daily_latest AS (
            SELECT DISTINCT ON (customer_id, archived_at::date)
                customer_id, archived_at::date AS d, risk_segment
            FROM churn_predictions_history
            ORDER BY customer_id, archived_at::date, archived_at DESC
        )
        SELECT d AS date, risk_segment, COUNT(*) AS count
        FROM daily_latest
        GROUP BY d, risk_segment
        ORDER BY d, risk_segment
    """
    return pd.read_sql(query, engine)


def _probability_shade(value: float) -> str:
    """Linear white->red interpolation, avoiding a matplotlib dependency for
    Styler.background_gradient (not available in the lightweight cloud deploy)."""
    value = max(0.0, min(1.0, value))
    start, end = (255, 245, 240), (165, 15, 21)  # light red -> dark red
    r, g, b = (int(start[i] + (end[i] - start[i]) * value) for i in range(3))
    text_color = "white" if value > 0.55 else "black"
    return f"background-color: rgb({r},{g},{b}); color: {text_color};"


def _style_high_risk_table(df: pd.DataFrame):
    def _segment_style(val):
        color = RISK_COLOR_MAP.get(val, "#7f7f7f")
        return f"background-color: {color}; color: white; font-weight: 600; text-align: center;"

    return (
        df.style.map(_segment_style, subset=["segment"])
        .map(lambda v: _probability_shade(v), subset=["probability"])
        .format({"probability": "{:.1%}", "monthly_charges": "${:.2f}"})
    )


def main() -> None:
    if demo_mode.is_demo_mode():
        st.warning(
            "🟡 **Demo Mode** — showing sample data (no live database connection)."
        )
    else:
        st.success("🟢 **Live Mode** — connected to the database.")

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
    st.subheader("Top Risk Customer")
    top_risk = predictions.sort_values("churn_probability", ascending=False).iloc[0]
    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        st.plotly_chart(
            probability_gauge(
                top_risk["churn_probability"],
                get_risk_color(top_risk["risk_segment"]),
                title=f"{top_risk['customer_id']}",
            ),
            use_container_width=True,
        )
    with gcol2:
        st.markdown(f"**Customer ID:** {top_risk['customer_id']}")
        st.markdown(f"**Risk Segment:** {top_risk['risk_segment']}")
        st.markdown(f"**Contract:** {top_risk['contract']}")
        st.markdown(f"**Tenure:** {top_risk['tenure']} months")
        st.markdown(f"**Monthly Charges:** ${top_risk['monthly_charges']:.2f}")

    st.divider()
    st.subheader("Trends & Distributions")
    trend = load_risk_segment_trend()
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        if not trend.empty:
            trend_fig = px.area(
                trend,
                x="date",
                y="count",
                color="risk_segment",
                color_discrete_map=RISK_COLOR_MAP,
                category_orders={"risk_segment": ["Low", "Medium", "High"]},
                title="Risk Segment Trend Over Time",
            )
            trend_fig.update_layout(xaxis_title="Date", yaxis_title="Customer Count")
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("No historical scoring runs available yet to build a trend.")
    with tcol2:
        box_fig = px.box(
            predictions,
            x="contract",
            y="churn_probability",
            color="contract",
            title="Churn Probability Distribution by Contract Type",
            points="outliers",
        )
        box_fig.update_layout(
            xaxis_title="Contract Type",
            yaxis_title="Churn Probability",
            showlegend=False,
        )
        st.plotly_chart(box_fig, use_container_width=True)

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

    if high_risk.empty:
        st.info("No high risk customers match the current search/filters.")
    else:
        st.dataframe(
            _style_high_risk_table(high_risk), use_container_width=True, hide_index=True
        )

    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download All Predictions as CSV",
        data=csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


main()
