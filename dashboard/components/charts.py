"""Reusable chart-building functions for the Streamlit dashboard (mostly Plotly figures)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

RISK_COLOR_MAP = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}


def churn_distribution_pie(df):
    """Pie chart of churn Yes/No. Expects a 'churn' column."""
    counts = df["churn"].value_counts().reset_index()
    counts.columns = ["churn", "count"]

    fig = px.pie(
        counts,
        names="churn",
        values="count",
        color="churn",
        color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
        title="Churn Distribution",
        hole=0.35,
    )
    return fig


def risk_segment_bar(df):
    """Bar chart of customer counts per risk segment. Expects a 'risk_segment' column."""
    counts = (
        df["risk_segment"]
        .value_counts()
        .reindex(["High", "Medium", "Low"])
        .dropna()
        .reset_index()
    )
    counts.columns = ["risk_segment", "count"]

    fig = px.bar(
        counts,
        x="risk_segment",
        y="count",
        color="risk_segment",
        color_discrete_map=RISK_COLOR_MAP,
        title="Risk Segment Distribution",
        text="count",
    )
    fig.update_layout(showlegend=False)
    return fig


def probability_histogram(df):
    """Histogram of churn probabilities. Expects a 'churn_probability' column."""
    fig = px.histogram(
        df,
        x="churn_probability",
        nbins=30,
        title="Churn Probability Distribution",
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(xaxis_title="Churn Probability", yaxis_title="Customer Count")
    return fig


def feature_importance_bar(df, top_n: int = 15):
    """Horizontal bar chart of the top-N most important features.

    Expects columns 'feature' and 'importance' (rf_feature_importance.csv format).
    """
    top = (
        df.sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )

    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances",
        color_discrete_sequence=["#1f77b4"],
    )
    return fig


def roc_curve_plot(fpr, tpr, auc: float, model_name: str = "Model"):
    """ROC curve line chart with a random-guess diagonal reference line."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{model_name} (AUC = {auc:.3f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def shap_summary_plot(shap_values, features):
    """SHAP beeswarm summary plot. Returns a matplotlib figure (SHAP has no native Plotly API) —
    display it in Streamlit with st.pyplot(fig)."""
    import shap

    fig = plt.figure()
    shap.summary_plot(shap_values, features, show=False)
    plt.tight_layout()
    return fig
