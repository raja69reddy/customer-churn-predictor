"""Customer Lookup page — look up a single customer and run a live churn prediction."""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.components.metrics import get_risk_color
from src.models.predict import RETENTION_ACTIONS, ChurnPredictor
from src.utils.db import get_engine

st.set_page_config(page_title="Customer Lookup", page_icon="🔮", layout="wide")


def load_customer(customer_id: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        "SELECT * FROM raw_customers WHERE customer_id = %(cid)s",
        engine,
        params={"cid": customer_id},
    )


def probability_gauge(probability: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#e6f4ea"},
                {"range": [40, 70], "color": "#fdf0e0"},
                {"range": [70, 100], "color": "#fbe4e2"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_profile_card(customer: pd.Series) -> None:
    st.subheader("Customer Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Customer ID:** {customer['customer_id']}")
        st.markdown(f"**Gender:** {customer['gender']}")
        st.markdown(f"**Senior Citizen:** {'Yes' if customer['senior_citizen'] else 'No'}")
        st.markdown(f"**Partner:** {customer['partner']}")
        st.markdown(f"**Dependents:** {customer['dependents']}")
    with col2:
        st.markdown(f"**Tenure:** {customer['tenure']} months")
        st.markdown(f"**Contract:** {customer['contract']}")
        st.markdown(f"**Payment Method:** {customer['payment_method']}")
        st.markdown(f"**Internet Service:** {customer['internet_service']}")
        st.markdown(f"**Phone Service:** {customer['phone_service']}")
    with col3:
        st.markdown(f"**Monthly Charges:** ${customer['monthly_charges']:.2f}")
        st.markdown(f"**Total Charges:** ${customer['total_charges']:.2f}")
        st.markdown(f"**Online Security:** {customer['online_security']}")
        st.markdown(f"**Tech Support:** {customer['tech_support']}")
        st.markdown(f"**Actual Churn (historical):** {customer['churn']}")


def main() -> None:
    st.title("Customer Lookup")

    customer_id = st.text_input("Customer ID", placeholder="e.g. CUST-00001")
    predict_clicked = st.button("Predict Churn", type="primary")

    if not predict_clicked:
        return

    if not customer_id:
        st.warning("Enter a customer ID first.")
        return

    matches = load_customer(customer_id.strip())
    if matches.empty:
        st.error(f"No customer found with ID '{customer_id}'.")
        return

    customer = matches.iloc[0]
    render_profile_card(customer)

    st.divider()

    predictor = ChurnPredictor()
    predictor.load_best_model()

    customer_dict = customer.to_dict()
    result = predictor.predict_single(customer_dict)
    color = get_risk_color(result["risk_segment"])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(probability_gauge(result["churn_probability"], color), use_container_width=True)
    with col2:
        st.markdown(
            f"<span style='background-color:{color}; color:white; padding:6px 16px; "
            f"border-radius:16px; font-weight:600;'>{result['risk_segment']} Risk</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Churn Probability:** {result['churn_probability']:.1%}")

        st.markdown("#### Top 5 Risk Factors")
        factors = predictor.explain_prediction(customer_dict, top_n=5)
        for i, factor in enumerate(factors, start=1):
            st.markdown(
                f"{i}. **{factor['feature'].replace('_', ' ')}** — {factor['direction']} "
                f"(SHAP={factor['shap_value']:+.4f})"
            )

    st.divider()
    st.markdown("#### AI Explanation")
    explanation = predictor.explain_with_gpt(customer_dict, result)
    st.info(explanation)

    st.markdown("#### Recommended Retention Actions")
    actions = RETENTION_ACTIONS.get(result["risk_segment"], RETENTION_ACTIONS["Low"])
    for action in actions:
        st.markdown(f"- {action}")


main()
