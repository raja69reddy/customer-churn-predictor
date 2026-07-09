"""Reusable KPI card components and formatting helpers for the Streamlit dashboard."""
import streamlit as st

RISK_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
DEFAULT_COLOR = "#1f77b4"


def format_percentage(p: float) -> str:
    """Formats a fraction (0.234) as a percentage string ('23.4%')."""
    return f"{p * 100:.1f}%"


def format_number(n) -> str:
    """Formats a number (7043) with thousands separators ('7,043')."""
    return f"{n:,}"


def get_risk_color(segment: str) -> str:
    """Returns the display color for a risk segment: High=red, Medium=orange, Low=green."""
    return RISK_COLORS.get(segment, "#7f7f7f")


def display_kpi_card(title: str, value, delta: str = None, color: str = DEFAULT_COLOR) -> None:
    """Renders a single colored KPI card with a title, big value, and optional delta caption."""
    delta_html = (
        f"<div style='font-size:0.85rem; color:gray; margin-top:2px;'>{delta}</div>"
        if delta
        else ""
    )
    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; padding: 0.6rem 1rem;
                    background-color: rgba(127,127,127,0.08); border-radius: 6px;">
            <div style="font-size:0.85rem; color:gray;">{title}</div>
            <div style="font-size:1.7rem; font-weight:700;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_kpi_row(metrics_list: list) -> None:
    """Lays out a list of KPI cards in a single row of equal-width columns.

    Each item in metrics_list is a dict with keys: title, value, and optionally delta, color.
    """
    columns = st.columns(len(metrics_list))
    for column, metric in zip(columns, metrics_list):
        with column:
            display_kpi_card(
                metric.get("title", ""),
                metric.get("value", "-"),
                metric.get("delta"),
                metric.get("color", DEFAULT_COLOR),
            )
