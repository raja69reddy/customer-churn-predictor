"""Retention Targeting page — priority tiers, revenue at risk, and campaign ROI estimation."""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dashboard.components.metrics import display_kpi_row, format_number  # noqa: E402

st.set_page_config(page_title="Retention Targeting", page_icon="🔮", layout="wide")

RETENTION_TARGETS_PATH = "data/processed/retention_targets.csv"
CACHE_TTL_SECONDS = 300

TIER_COLORS = {"Tier 1": "#d62728", "Tier 2": "#ff7f0e", "Tier 3": "#7f7f7f"}
TIER_ACTIONS = {
    "Tier 1": ("📞", "Personal call + discount offer"),
    "Tier 2": ("📧", "Email campaign + loyalty reward"),
    "Tier 3": ("🤖", "Automated email sequence"),
}


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_retention"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(
        f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_retention_targets() -> pd.DataFrame:
    return pd.read_csv(RETENTION_TARGETS_PATH)


def render_action_cards(tier_counts: pd.Series) -> None:
    st.markdown("#### Recommended Actions by Tier")
    for tier, (icon, action) in TIER_ACTIONS.items():
        count = int(tier_counts.get(tier, 0))
        with st.container(border=True):
            st.markdown(
                f"<span style='background-color:{TIER_COLORS[tier]}; color:white; "
                f"padding:4px 12px; border-radius:12px; font-weight:600;'>{tier}</span>"
                f"&nbsp;&nbsp;{icon} {action} &nbsp; "
                f"<span style='color:gray;'>({count} customers)</span>",
                unsafe_allow_html=True,
            )


def render_export_by_tier(targets: pd.DataFrame) -> None:
    st.markdown("#### Export Targeting List")
    tier_choice = st.selectbox(
        "Select a tier to export",
        ["All Tiers"] + sorted(targets["priority_tier"].unique()),
    )
    export_df = (
        targets
        if tier_choice == "All Tiers"
        else targets[targets["priority_tier"] == tier_choice]
    )
    st.download_button(
        f"Download {tier_choice} Targeting List as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"retention_targets_{tier_choice.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


def render_roi_estimator(targets: pd.DataFrame) -> None:
    st.subheader("📊 Campaign ROI Estimator")
    st.caption(
        "Estimate the return on a retention campaign targeting a tier, "
        "based on an assumed cost per customer and expected retention success rate."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        tier_choice = st.selectbox(
            "Target tier", sorted(targets["priority_tier"].unique()), key="roi_tier"
        )
    with col2:
        cost_per_customer = st.number_input(
            "Campaign cost per customer ($)", min_value=0.0, value=15.0, step=1.0
        )
    with col3:
        success_rate = (
            st.slider("Expected retention success rate (%)", 0, 100, 30) / 100
        )

    tier_df = targets[targets["priority_tier"] == tier_choice]
    num_targeted = len(tier_df)
    campaign_cost = num_targeted * cost_per_customer
    customers_retained = num_targeted * success_rate
    annual_revenue_saved = customers_retained * tier_df["monthly_charges"].mean() * 12
    roi_pct = (
        ((annual_revenue_saved - campaign_cost) / campaign_cost * 100)
        if campaign_cost > 0
        else 0
    )

    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    rcol1.metric("Customers Targeted", format_number(num_targeted))
    rcol2.metric("Campaign Cost", f"${campaign_cost:,.2f}")
    rcol3.metric("Est. Annual Revenue Saved", f"${annual_revenue_saved:,.2f}")
    rcol4.metric("Estimated ROI", f"{roi_pct:+.0f}%")

    if campaign_cost > 0:
        st.caption(
            f"Assumes ~{customers_retained:.0f} of {num_targeted} targeted customers are "
            f"retained at a {success_rate:.0%} success rate, each worth "
            f"~${tier_df['monthly_charges'].mean():.2f}/month (annualized)."
        )


def main() -> None:
    render_cache_controls()

    st.title("Retention Targeting")

    try:
        with st.spinner("Loading retention targets..."):
            targets = load_retention_targets()
    except FileNotFoundError:
        st.error(
            "⚠️ No retention targeting list found. Run `python -m src.models.retention_targets` "
            "first to generate it."
        )
        return

    if targets.empty:
        st.warning("No data available — the retention targeting list is empty.")
        return

    tier_counts = targets["priority_tier"].value_counts()
    total_revenue_at_risk = targets["monthly_charges"].sum()

    display_kpi_row(
        [
            {
                "title": "Tier 1 Customers",
                "value": format_number(int(tier_counts.get("Tier 1", 0))),
                "color": "#d62728",
            },
            {
                "title": "Tier 2 Customers",
                "value": format_number(int(tier_counts.get("Tier 2", 0))),
                "color": "#ff7f0e",
            },
            {
                "title": "Tier 3 Customers",
                "value": format_number(int(tier_counts.get("Tier 3", 0))),
                "color": "#7f7f7f",
            },
            {
                "title": "Total Revenue at Risk",
                "value": f"${total_revenue_at_risk:,.2f}",
                "color": "#9467bd",
            },
        ]
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        pie_data = tier_counts.reset_index()
        pie_data.columns = ["priority_tier", "count"]
        fig = px.pie(
            pie_data,
            names="priority_tier",
            values="count",
            title="Tier Distribution",
            hole=0.35,
            color="priority_tier",
            color_discrete_map=TIER_COLORS,
            category_orders={"priority_tier": ["Tier 1", "Tier 2", "Tier 3"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        revenue_by_tier = (
            targets.groupby("priority_tier")["monthly_charges"]
            .sum()
            .reindex(["Tier 1", "Tier 2", "Tier 3"])
            .dropna()
            .reset_index()
        )
        revenue_by_tier.columns = ["priority_tier", "monthly_revenue_at_risk"]
        rev_fig = px.bar(
            revenue_by_tier,
            x="priority_tier",
            y="monthly_revenue_at_risk",
            color="priority_tier",
            color_discrete_map=TIER_COLORS,
            title="Monthly Revenue at Risk by Tier",
            text="monthly_revenue_at_risk",
        )
        rev_fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        rev_fig.update_layout(
            xaxis_title="Priority Tier",
            yaxis_title="Monthly Revenue at Risk ($)",
            showlegend=False,
        )
        st.plotly_chart(rev_fig, use_container_width=True)

    st.divider()
    render_action_cards(tier_counts)

    st.divider()
    render_roi_estimator(targets)

    st.divider()
    st.subheader("Tier 1 Priority Customers")
    tier1 = targets[targets["priority_tier"] == "Tier 1"].sort_values(
        "churn_probability", ascending=False
    )
    st.dataframe(tier1, use_container_width=True, hide_index=True)

    render_export_by_tier(targets)


main()
