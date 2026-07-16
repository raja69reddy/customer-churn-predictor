"""Churn Overview page — headline KPIs, trend charts, and churn distribution charts."""

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
from dashboard.components.charts import churn_distribution_pie  # noqa: E402
from dashboard.components.metrics import (  # noqa: E402
    display_kpi_row,
    format_number,
    format_percentage,
)
from src.utils.db import get_engine  # noqa: E402

st.set_page_config(page_title="Churn Overview", page_icon="🔮", layout="wide")

CACHE_TTL_SECONDS = 300

TENURE_BINS = [0, 12, 24, 48, 1000]
TENURE_LABELS = [
    "New (0-12mo)",
    "Growing (12-24mo)",
    "Established (24-48mo)",
    "Loyal (48mo+)",
]


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


def _filter_demo_customers(filters: dict) -> pd.DataFrame:
    df = demo_mode.get_demo_predictions()
    df = df[
        (df["tenure"] >= filters["tenure_range"][0])
        & (df["tenure"] <= filters["tenure_range"][1])
    ]
    if filters["risk_segment"] != "All":
        df = df[df["risk_segment"] == filters["risk_segment"]]
    if filters["contract"] != "All":
        df = df[df["contract"] == filters["contract"]]
    return df[
        [
            "customer_id",
            "churn",
            "contract",
            "tenure",
            "monthly_charges",
            "risk_segment",
            "churn_probability",
        ]
    ]


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_filtered_customers(filters: dict) -> pd.DataFrame:
    if demo_mode.is_demo_mode():
        return _filter_demo_customers(filters)

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
    if demo_mode.is_demo_mode():
        return demo_mode.get_demo_churn_overview()
    engine = get_engine()
    return pd.read_sql("SELECT * FROM vw_churn_overview", engine)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_model_accuracy() -> tuple:
    """Returns (active model accuracy, delta vs the runner-up model)."""
    if demo_mode.is_demo_mode():
        metrics = demo_mode.get_demo_model_metrics()
        if metrics.empty:
            return None, None
        ranked = metrics.sort_values("auc_score", ascending=False).reset_index(
            drop=True
        )
        best_acc = float(ranked.loc[0, "accuracy"])
        runner_up_acc = float(ranked.loc[1, "accuracy"]) if len(ranked) > 1 else None
        return best_acc, (
            best_acc - runner_up_acc if runner_up_acc is not None else None
        )

    engine = get_engine()
    ranked = pd.read_sql(
        "SELECT accuracy, is_active FROM model_registry ORDER BY auc_score DESC", engine
    )
    if ranked.empty:
        return None, None
    active = ranked[ranked["is_active"]]
    if active.empty:
        return None, None
    best_acc = float(active.iloc[0]["accuracy"])
    others = ranked[~ranked["is_active"]]
    runner_up_acc = float(others.iloc[0]["accuracy"]) if not others.empty else None
    return best_acc, (best_acc - runner_up_acc if runner_up_acc is not None else None)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_previous_period_snapshot(customer_ids: tuple) -> pd.DataFrame:
    """Returns each customer's most recently *archived* (pre-current) risk_segment.

    Live mode reads churn_predictions_history (real archive of every prior scoring run —
    see Day 9). Demo mode has no real history, so it uses a deterministic alternating-row
    split of the demo sample as an illustrative stand-in "previous period".
    """
    if demo_mode.is_demo_mode():
        demo = demo_mode.get_demo_predictions()
        prev = demo[demo["customer_id"].isin(customer_ids)].iloc[::2]
        return prev[["customer_id", "risk_segment"]]

    if not customer_ids:
        return pd.DataFrame(columns=["customer_id", "risk_segment"])

    from sqlalchemy import bindparam, text

    engine = get_engine()
    query = text("""
        SELECT DISTINCT ON (customer_id) customer_id, risk_segment
        FROM churn_predictions_history
        WHERE customer_id IN :ids
        ORDER BY customer_id, archived_at DESC
        """).bindparams(bindparam("ids", expanding=True))
    return pd.read_sql(query, engine, params={"ids": list(customer_ids)})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_churn_trend() -> pd.DataFrame:
    """Returns avg churn probability / high-risk count per day.

    Live mode aggregates the real churn_predictions_history archive (one row per
    scoring run) by day. Demo mode has no real history, so it generates a smooth
    illustrative 14-day trend around the current demo average — clearly shown only
    under the Demo Mode banner.
    """
    if demo_mode.is_demo_mode():
        demo = demo_mode.get_demo_predictions()
        base_prob = float(demo["churn_probability"].mean())
        base_high = int((demo["risk_segment"] == "High").sum())
        dates = pd.date_range(
            end=pd.Timestamp.today().normalize(), periods=14, freq="D"
        )
        wobble = [((i * 37) % 11 - 5) / 100 for i in range(14)]
        return pd.DataFrame(
            {
                "date": dates,
                "avg_probability": [max(0.0, min(1.0, base_prob + w)) for w in wobble],
                "high_risk_count": [max(0, base_high + int(w * 40)) for w in wobble],
            }
        )

    engine = get_engine()
    query = """
        WITH daily_latest AS (
            SELECT DISTINCT ON (customer_id, archived_at::date)
                customer_id, archived_at::date AS d, churn_probability, risk_segment
            FROM churn_predictions_history
            ORDER BY customer_id, archived_at::date, archived_at DESC
        )
        SELECT d AS date, AVG(churn_probability) AS avg_probability,
               SUM(CASE WHEN risk_segment = 'High' THEN 1 ELSE 0 END) AS high_risk_count
        FROM daily_latest
        GROUP BY d
        ORDER BY d
    """
    return pd.read_sql(query, engine)


def main() -> None:
    is_demo = demo_mode.is_demo_mode()
    if is_demo:
        st.warning(
            "🟡 **Demo Mode** — showing sample data (no live database connection)."
        )
    else:
        st.success("🟢 **Live Mode** — connected to the database.")

    st.title("Churn Overview")

    filters = render_sidebar_filters()
    try:
        with st.spinner("Loading overview..."):
            customers = load_filtered_customers(filters)
            overview = load_churn_overview()
            model_accuracy, accuracy_delta = load_model_accuracy()
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

    # --- period-over-period comparison, using the previously archived scoring run ---
    prev_snapshot = load_previous_period_snapshot(tuple(customers["customer_id"]))
    prev_merged = customers.merge(
        prev_snapshot, on="customer_id", how="inner", suffixes=("", "_prev")
    )
    if not prev_merged.empty:
        prev_churn_rate = (prev_merged["churn"] == "Yes").mean()
        prev_high_risk_count = int((prev_merged["risk_segment_prev"] == "High").sum())
        prev_revenue_at_risk = prev_merged.loc[
            prev_merged["risk_segment_prev"].isin(["High", "Medium"]), "monthly_charges"
        ].sum()
        churn_rate_delta = churn_rate - prev_churn_rate
        high_risk_delta = high_risk_count - prev_high_risk_count
        revenue_delta = revenue_at_risk - prev_revenue_at_risk
    else:
        churn_rate_delta = high_risk_delta = revenue_delta = None

    display_kpi_row(
        [
            {
                "title": "Overall Churn Rate",
                "value": format_percentage(churn_rate),
                "delta": (
                    f"{churn_rate_delta * 100:+.1f} pp vs previous run"
                    if churn_rate_delta is not None
                    else "vs previous run: n/a"
                ),
                "color": "#d62728",
            },
            {
                "title": "Total High Risk Customers",
                "value": format_number(high_risk_count),
                "delta": (
                    f"{high_risk_delta:+,} vs previous run"
                    if high_risk_delta is not None
                    else "vs previous run: n/a"
                ),
                "color": "#ff7f0e",
            },
            {
                "title": "Revenue at Risk",
                "value": f"${revenue_at_risk:,.2f}",
                "delta": (
                    f"${revenue_delta:+,.2f} vs previous run"
                    if revenue_delta is not None
                    else "vs previous run: n/a"
                ),
                "color": "#9467bd",
            },
            {
                "title": "Model Accuracy",
                "value": (
                    format_percentage(model_accuracy)
                    if model_accuracy is not None
                    else "N/A"
                ),
                "delta": (
                    f"{accuracy_delta * 100:+.2f} pp vs runner-up model"
                    if accuracy_delta is not None
                    else "vs runner-up: n/a"
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
        fig = px.bar(
            contract_data.sort_values("churn_rate_pct"),
            x="segment",
            y="churn_rate_pct",
            title="Churn Rate by Contract Type",
            color="churn_rate_pct",
            color_continuous_scale="Reds",
            text="churn_rate_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            xaxis_title="Contract Type",
            yaxis_title="Churn Rate (%)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Churn Trend")
    trend = load_churn_trend()
    if not trend.empty:
        col3, col4 = st.columns(2)
        with col3:
            trend_fig = px.line(
                trend,
                x="date",
                y="avg_probability",
                markers=True,
                title="Average Churn Probability Over Time",
                color_discrete_sequence=["#d62728"],
            )
            trend_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Avg Churn Probability",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(trend_fig, use_container_width=True)
        with col4:
            high_risk_fig = px.line(
                trend,
                x="date",
                y="high_risk_count",
                markers=True,
                title="High Risk Customer Count Over Time",
                color_discrete_sequence=["#ff7f0e"],
            )
            high_risk_fig.update_layout(
                xaxis_title="Date", yaxis_title="High Risk Customers"
            )
            st.plotly_chart(high_risk_fig, use_container_width=True)

        if not is_demo:
            st.caption(
                "Churn probability is deterministic for a fixed model — this trend only moves "
                "when the active model, thresholds, or underlying data change between scoring runs."
            )
    else:
        st.info("No historical scoring runs available yet to build a trend.")

    st.divider()
    st.subheader("Churn Rate by Tenure Group & Contract")
    heatmap_source = customers.copy()
    heatmap_source["tenure_group"] = pd.cut(
        heatmap_source["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    )
    pivot = (
        heatmap_source.assign(is_churn=(heatmap_source["churn"] == "Yes").astype(int))
        .pivot_table(
            index="tenure_group",
            columns="contract",
            values="is_churn",
            aggfunc="mean",
            observed=False,
        )
        .reindex(TENURE_LABELS)
    )
    if not pivot.empty:
        heatmap_fig = px.imshow(
            pivot * 100,
            text_auto=".1f",
            color_continuous_scale="Reds",
            aspect="auto",
            labels=dict(x="Contract Type", y="Tenure Group", color="Churn Rate (%)"),
            title="Churn Rate (%) — Tenure Group x Contract Type",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

    st.divider()
    csv_bytes = customers.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Overview Data as CSV",
        data=csv_bytes,
        file_name="churn_overview_data.csv",
        mime="text/csv",
    )


main()
