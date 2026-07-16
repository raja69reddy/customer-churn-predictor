"""Customer Lookup page — look up a single customer and run a live churn prediction."""

import os
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dashboard import demo_mode  # noqa: E402
from dashboard.components.charts import probability_gauge  # noqa: E402
from dashboard.components.metrics import get_risk_color  # noqa: E402
from src.utils.db import get_engine  # noqa: E402

try:
    from src.models.predict import RETENTION_ACTIONS, ChurnPredictor
except ImportError:
    # Heavy ML deps (xgboost/shap/openai/...) aren't in requirements_streamlit.txt —
    # demo mode never needs a live ChurnPredictor, so fall back to a static copy of
    # RETENTION_ACTIONS and treat the live-scoring path as unavailable.
    ChurnPredictor = None
    RETENTION_ACTIONS = {
        "High": [
            "Offer a loyalty discount or incentive to switch to a longer-term contract",
            "Assign a customer success rep for proactive outreach this week",
            "Offer a free service add-on (e.g. online security) to increase switching cost",
        ],
        "Medium": [
            "Send a satisfaction survey to identify pain points",
            "Highlight underused services or bundle benefits",
            "Offer a modest loyalty discount tied to contract renewal",
        ],
        "Low": [
            "No immediate action needed — monitor at the next billing cycle",
        ],
    }

st.set_page_config(page_title="Customer Lookup", page_icon="🔮", layout="wide")

CACHE_TTL_SECONDS = 300

ACTION_ICONS = ["🎯", "📞", "🎁", "📋", "💬", "⭐"]


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_customer_lookup"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(
        f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_customer(customer_id: str) -> pd.DataFrame:
    if demo_mode.is_demo_mode():
        demo = demo_mode.get_demo_predictions()
        return demo[demo["customer_id"] == customer_id]

    engine = get_engine()
    return pd.read_sql(
        "SELECT * FROM raw_customers WHERE customer_id = %(cid)s",
        engine,
        params={"cid": customer_id},
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_similar_high_risk(
    customer_id: str, contract: str, tenure: int, limit: int = 5
) -> pd.DataFrame:
    """Returns other High-risk customers with the same contract type, ranked by tenure proximity."""
    if demo_mode.is_demo_mode():
        demo = demo_mode.get_demo_predictions()
        pool = demo[
            (demo["risk_segment"] == "High")
            & (demo["contract"] == contract)
            & (demo["customer_id"] != customer_id)
        ].copy()
    else:
        engine = get_engine()
        pool = pd.read_sql(
            "SELECT * FROM vw_churn_predictions WHERE risk_segment = 'High' "
            "AND contract = %(contract)s AND customer_id != %(cid)s",
            engine,
            params={"contract": contract, "cid": customer_id},
        )

    if pool.empty:
        return pool

    pool["tenure_diff"] = (pool["tenure"] - tenure).abs()
    cols = [
        c
        for c in [
            "customer_id",
            "tenure",
            "monthly_charges",
            "churn_probability",
            "risk_segment",
        ]
        if c in pool.columns
    ]
    return pool.sort_values("tenure_diff").head(limit)[cols]


def _demo_prediction_result(row: pd.Series) -> tuple[dict, list, str]:
    """Builds a (result, factors, explanation) tuple from a precomputed demo row,
    mirroring ChurnPredictor.predict_single / explain_prediction / explain_with_gpt."""
    result = {
        "churn_probability": float(row["churn_probability"]),
        "risk_segment": row["risk_segment"],
    }
    factors = [
        {
            "feature": row[f"factor_{i}"],
            "shap_value": float(row[f"factor_{i}_shap"]),
            "direction": (
                "increases risk" if row[f"factor_{i}_shap"] > 0 else "decreases risk"
            ),
        }
        for i in (1, 2, 3)
    ]
    names = [f["feature"].replace("_", " ") for f in factors]
    explanation = (
        "[demo mode — precomputed sample explanation] "
        f"This customer has a {result['risk_segment'].lower()} churn risk "
        f"({result['churn_probability']:.0%} probability), driven mainly by {names[0]}, "
        f"{names[1]}, and {names[2]}."
    )
    return result, factors, explanation


def render_profile_card(customer: pd.Series) -> None:
    st.subheader("Customer Profile")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**👤 Demographics**")
            st.markdown(f"Customer ID: `{customer['customer_id']}`")
            st.markdown(f"Gender: {customer['gender']}")
            st.markdown(
                f"Senior Citizen: {'Yes' if customer['senior_citizen'] else 'No'}"
            )
            st.markdown(f"Partner: {customer['partner']}")
            st.markdown(f"Dependents: {customer['dependents']}")
        with col2:
            st.markdown("**📄 Account**")
            st.markdown(f"Tenure: {customer['tenure']} months")
            st.markdown(f"Contract: {customer['contract']}")
            st.markdown(f"Payment Method: {customer['payment_method']}")
            st.markdown(f"Internet Service: {customer['internet_service']}")
            st.markdown(f"Phone Service: {customer['phone_service']}")
        with col3:
            st.markdown("**💰 Billing & Services**")
            st.markdown(f"Monthly Charges: ${customer['monthly_charges']:.2f}")
            st.markdown(f"Total Charges: ${customer['total_charges']:.2f}")
            st.markdown(f"Online Security: {customer['online_security']}")
            st.markdown(f"Tech Support: {customer['tech_support']}")
            st.markdown(f"Actual Churn (historical): {customer['churn']}")


def render_animated_probability(probability: float, color: str) -> None:
    st.markdown("**Churn Probability**")
    bar = st.progress(0)
    target = int(round(probability * 100))
    step = max(1, target // 25)
    for pct in range(0, target + 1, step):
        bar.progress(min(pct, target), text=f"{min(pct, target)}%")
        time.sleep(0.01)
    bar.progress(target, text=f"{target}%")
    st.markdown(
        f"<span style='background-color:{color}; color:white; padding:6px 16px; "
        f"border-radius:16px; font-weight:600;'>{target}% churn probability</span>",
        unsafe_allow_html=True,
    )


def render_risk_factor_badges(factors: list) -> None:
    st.markdown(f"#### Top {len(factors)} Risk Factors")
    badges_html = ""
    for factor in factors:
        increases = factor["direction"] == "increases risk"
        bg = "#fbe4e2" if increases else "#e6f4ea"
        fg = "#d62728" if increases else "#2ca02c"
        arrow = "▲" if increases else "▼"
        badges_html += (
            f"<span style='display:inline-block; background-color:{bg}; color:{fg}; "
            f"padding:6px 12px; margin:4px 6px 4px 0; border-radius:14px; font-size:0.85rem; "
            f"font-weight:600;'>{arrow} {factor['feature'].replace('_', ' ')} "
            f"(SHAP={factor['shap_value']:+.3f})</span>"
        )
    st.markdown(badges_html, unsafe_allow_html=True)


def render_action_cards(actions: list) -> None:
    st.markdown("#### Recommended Retention Actions")
    for i, action in enumerate(actions):
        icon = ACTION_ICONS[i % len(ACTION_ICONS)]
        with st.container(border=True):
            st.markdown(f"{icon} &nbsp; {action}")


def render_similar_customers(customer_id: str, contract: str, tenure: int) -> None:
    st.markdown("#### Similar High Risk Customers")
    st.caption(
        f"Other High risk customers on a {contract} contract, closest in tenure."
    )
    similar = load_similar_high_risk(customer_id, contract, tenure)
    if similar.empty:
        st.info("No other High risk customers found with a matching profile.")
        return
    display = similar.rename(
        columns={"churn_probability": "probability", "risk_segment": "segment"}
    )
    if "probability" in display.columns:
        display["probability"] = display["probability"].map(lambda p: f"{p:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    is_demo = demo_mode.is_demo_mode()
    if is_demo:
        st.warning(
            "🟡 **Demo Mode** — showing sample data (no live database connection)."
        )
    else:
        st.success("🟢 **Live Mode** — connected to the database.")

    render_cache_controls()

    st.title("Customer Lookup")

    placeholder = "e.g. CUST-DEMO-001" if is_demo else "e.g. CUST-00001"
    customer_id = st.text_input("Customer ID", placeholder=placeholder)
    predict_clicked = st.button("Predict Churn", type="primary")

    if not predict_clicked:
        return

    if not customer_id:
        st.warning("Enter a customer ID first.")
        return

    try:
        matches = load_customer(customer_id.strip())
    except Exception:
        st.error(
            "⚠️ Could not connect to the database. Please check your connection and try again."
        )
        return

    if matches.empty:
        st.warning(f"No data available — no customer found with ID '{customer_id}'.")
        return

    customer = matches.iloc[0]
    render_profile_card(customer)

    st.divider()

    if is_demo or ChurnPredictor is None:
        with st.spinner("Loading prediction..."):
            result, factors, explanation = _demo_prediction_result(customer)
    else:
        try:
            with st.spinner("Running prediction..."):
                predictor = ChurnPredictor()
                predictor.load_best_model()

                customer_dict = customer.to_dict()
                result = predictor.predict_single(customer_dict)
                factors = predictor.explain_prediction(customer_dict, top_n=5)
                explanation = predictor.explain_with_gpt(customer_dict, result)
        except Exception as e:
            st.error(f"⚠️ Could not generate a prediction for this customer: {e}")
            return

    color = get_risk_color(result["risk_segment"])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            probability_gauge(result["churn_probability"], color),
            use_container_width=True,
        )
    with col2:
        st.markdown(
            f"<span style='background-color:{color}; color:white; padding:6px 16px; "
            f"border-radius:16px; font-weight:600;'>{result['risk_segment']} Risk</span>",
            unsafe_allow_html=True,
        )
        render_animated_probability(result["churn_probability"], color)
        render_risk_factor_badges(factors)

    st.divider()
    st.markdown("#### AI Explanation")
    st.info(explanation)

    render_action_cards(
        RETENTION_ACTIONS.get(result["risk_segment"], RETENTION_ACTIONS["Low"])
    )

    if result["risk_segment"] == "High":
        st.divider()
        render_similar_customers(
            customer["customer_id"], customer["contract"], customer["tenure"]
        )


main()
