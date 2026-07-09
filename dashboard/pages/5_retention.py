"""Retention Targeting page — priority tiers and recommended outreach actions."""
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.components.metrics import display_kpi_row, format_number

st.set_page_config(page_title="Retention Targeting", page_icon="🔮", layout="wide")

RETENTION_TARGETS_PATH = "data/processed/retention_targets.csv"
CACHE_TTL_SECONDS = 300

TIER_ACTIONS = {
    "Tier 1": "Personal call + discount offer",
    "Tier 2": "Email campaign + loyalty reward",
    "Tier 3": "Automated email sequence",
}


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_retention"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_retention_targets() -> pd.DataFrame:
    return pd.read_csv(RETENTION_TARGETS_PATH)


def main() -> None:
    render_cache_controls()

    st.title("Retention Targeting")

    with st.spinner("Loading retention targets..."):
        targets = load_retention_targets()
    tier_counts = targets["priority_tier"].value_counts()
    total_revenue_at_risk = targets["monthly_charges"].sum()

    display_kpi_row([
        {"title": "Tier 1 Customers", "value": format_number(int(tier_counts.get("Tier 1", 0))), "color": "#d62728"},
        {"title": "Tier 2 Customers", "value": format_number(int(tier_counts.get("Tier 2", 0))), "color": "#ff7f0e"},
        {"title": "Tier 3 Customers", "value": format_number(int(tier_counts.get("Tier 3", 0))), "color": "#7f7f7f"},
        {"title": "Total Revenue at Risk", "value": f"${total_revenue_at_risk:,.2f}", "color": "#9467bd"},
    ])

    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        pie_data = tier_counts.reset_index()
        pie_data.columns = ["priority_tier", "count"]
        fig = px.pie(
            pie_data, names="priority_tier", values="count",
            title="Tier Distribution", hole=0.35,
            color="priority_tier",
            color_discrete_map={"Tier 1": "#d62728", "Tier 2": "#ff7f0e", "Tier 3": "#7f7f7f"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Recommended Actions by Tier")
        for tier, action in TIER_ACTIONS.items():
            count = int(tier_counts.get(tier, 0))
            st.markdown(f"**{tier}** ({count} customers): {action}")

    st.divider()
    st.subheader("Tier 1 Priority Customers")
    tier1 = targets[targets["priority_tier"] == "Tier 1"].sort_values("churn_probability", ascending=False)
    st.dataframe(tier1, use_container_width=True, hide_index=True)

    csv_bytes = targets.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Retention Targeting List as CSV",
        data=csv_bytes,
        file_name="retention_targets.csv",
        mime="text/csv",
    )


main()
