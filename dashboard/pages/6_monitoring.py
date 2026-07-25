"""Model Monitoring page — PSI-based drift detection for the active model's predictions and
input features, plus a prediction-score-over-time view.

The PSI gauge, feature drift table, and data drift alerts need a live trained model plus the
full ML stack (xgboost/sklearn, via src/monitoring/model_monitor.py and data_drift_detector.py)
that aren't available in demo mode / requirements_streamlit.txt — those sections show an
explanatory message instead. The "prediction score distribution over time" section is pure
SQL/pandas (churn_predictions + churn_predictions_history, both populated by every prior day's
batch scoring runs) and works the same way in demo mode, using the single-snapshot demo CSV.
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dashboard import demo_mode  # noqa: E402
from src.utils.db import get_engine  # noqa: E402

try:
    from src.monitoring.data_drift_detector import (
        detect_charges_drift,
        detect_churn_rate_drift,
        detect_tenure_drift,
    )
    from src.monitoring.model_monitor import alert_on_drift, generate_monitoring_report

    LIVE_MONITORING_AVAILABLE = True
except ImportError:
    LIVE_MONITORING_AVAILABLE = False

st.set_page_config(page_title="Model Monitoring", page_icon="📡", layout="wide")

CACHE_TTL_SECONDS = 300
PSI_GAUGE_MAX = 0.5
STATUS_COLORS = {"Stable": "#2ca02c", "Moderate": "#ff7f0e", "Significant": "#d62728"}


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_monitoring"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(
        f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_predictions_over_time() -> pd.DataFrame:
    """Daily average churn probability, combining current + archived predictions —
    the historical archive (churn_predictions_history) is what gives this real day-to-day
    variance, since churn_predictions itself only reflects the latest scoring run."""
    if demo_mode.is_demo_mode():
        df = demo_mode.get_demo_predictions()[["predicted_at", "churn_probability"]]
        df["day"] = pd.to_datetime(df["predicted_at"]).dt.date
        return (
            df.groupby("day")
            .agg(
                avg_probability=("churn_probability", "mean"),
                n=("churn_probability", "size"),
            )
            .reset_index()
        )

    engine = get_engine()
    return pd.read_sql(
        """
        SELECT day, AVG(churn_probability) AS avg_probability, COUNT(*) AS n
        FROM (
            SELECT predicted_at::date AS day, churn_probability FROM churn_predictions
            UNION ALL
            SELECT predicted_at::date AS day, churn_probability FROM churn_predictions_history
        ) combined
        GROUP BY day
        ORDER BY day
        """,
        engine,
    )


def psi_gauge(psi: float, status: str, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=psi,
            number={"valueformat": ".4f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, PSI_GAUGE_MAX]},
                "bar": {"color": STATUS_COLORS.get(status, "#7f7f7f")},
                "steps": [
                    {"range": [0, 0.1], "color": "#e6f4ea"},
                    {"range": [0.1, 0.2], "color": "#fdf0e0"},
                    {"range": [0.2, PSI_GAUGE_MAX], "color": "#fbe4e2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.8,
                    "value": 0.2,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_live_monitoring_sections() -> None:
    with st.spinner("Running drift detection..."):
        report = generate_monitoring_report()
        alerts = alert_on_drift(report=report)
        tenure_drift = detect_tenure_drift()
        charges_drift = detect_charges_drift()
        churn_rate_drift = detect_churn_rate_drift()

    st.subheader("📊 PSI Drift Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            psi_gauge(
                report["prediction_drift"]["psi"],
                report["prediction_drift"]["status"],
                "Prediction Score PSI",
            ),
            use_container_width=True,
        )
    with col2:
        top_feature = report["feature_drift"].iloc[0]
        st.plotly_chart(
            psi_gauge(
                top_feature["psi"],
                top_feature["status"],
                f"Highest Feature PSI ({top_feature['feature']})",
            ),
            use_container_width=True,
        )
    st.caption(
        f"Model: `{report['model_name']}` — Overall status: **{report['overall_status']}**"
    )

    st.divider()
    st.subheader("🧬 Feature Drift Table")
    display = report["feature_drift"].copy()
    display["status"] = display["status"].map(
        lambda s: f"{'🟢' if s == 'Stable' else '🟠' if s == 'Moderate' else '🔴'} {s}"
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🚨 Data Drift Alerts")
    business_checks = [tenure_drift, charges_drift, churn_rate_drift]
    any_alert = bool(alerts) or any(c["status"] != "Stable" for c in business_checks)

    if not any_alert:
        st.success(
            "✅ No drift alerts — all monitored features and prediction scores are stable."
        )
    else:
        for alert in alerts:
            st.warning(f"⚠️ {alert['message']}")
        for check in business_checks:
            if check["status"] != "Stable":
                st.warning(
                    f"⚠️ {check['feature']} drift PSI={check['psi']} ({check['status']})"
                )

    with st.expander("Business feature drift details (tenure, charges, churn rate)"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "feature": c["feature"],
                        "psi": c["psi"],
                        "status": c["status"],
                    }
                    for c in business_checks
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    is_demo = demo_mode.is_demo_mode()
    if is_demo:
        st.warning(
            "🟡 **Demo Mode** — showing sample data (no live database connection)."
        )
    else:
        st.success("🟢 **Live Mode** — connected to the database.")

    render_cache_controls()

    st.title("Model Monitoring")

    if is_demo or not LIVE_MONITORING_AVAILABLE:
        st.info(
            "ℹ️ PSI drift detection requires a live trained model and the full ML dependencies "
            "(xgboost/scikit-learn), which aren't available in demo mode. Run this dashboard "
            "locally with the full `requirements.txt` and a populated database to see live "
            "PSI gauges, feature drift, and data drift alerts."
        )
    else:
        try:
            render_live_monitoring_sections()
        except Exception as e:
            st.error(f"⚠️ Could not run drift detection: {e}")

    st.divider()
    st.subheader("📈 Prediction Score Distribution Over Time")
    try:
        with st.spinner("Loading prediction history..."):
            history = load_predictions_over_time()
    except Exception:
        st.error(
            "⚠️ Could not connect to the database. Please check your connection and try again."
        )
        return

    if history.empty:
        st.warning("No data available — no scored predictions found.")
        return

    if is_demo:
        st.caption(
            "Demo mode only ships a single sample snapshot — this shows that snapshot's "
            "average probability rather than a real day-over-day trend."
        )

    if len(history) == 1:
        # A line chart is meaningless (and renders oddly) with a single x value — show it as
        # a metric instead, which is what demo mode's single-snapshot data actually is.
        row = history.iloc[0]
        st.metric(
            f"Average Churn Probability ({row['day']})",
            f"{row['avg_probability']:.1%}",
        )
    else:
        fig = px.line(
            history,
            x="day",
            y="avg_probability",
            markers=True,
            title="Average Churn Probability by Day",
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Average Churn Probability")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


main()
