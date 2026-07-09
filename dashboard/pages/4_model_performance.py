"""Model Performance page — leaderboard, ROC curves, feature importance, SHAP, confusion matrix."""
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.components.charts import feature_importance_bar, shap_summary_plot
from dashboard.components.metrics import display_kpi_row, format_number
from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.utils.db import get_engine

st.set_page_config(page_title="Model Performance", page_icon="🔮", layout="wide")

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest_tuned", "xgboost_tuned", "lightgbm", "ensemble"]
CACHE_TTL_SECONDS = 300


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_model_performance"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_model_performance() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM vw_model_performance", engine)


def main() -> None:
    render_cache_controls()

    st.title("Model Performance")

    performance = load_model_performance()
    best_row = performance.loc[performance["auc_score"].idxmax()]

    display_kpi_row([
        {"title": "Best Model", "value": best_row["model_name"], "color": "#2ca02c"},
        {"title": "Best AUC Score", "value": f"{best_row['auc_score']:.4f}", "color": "#1f77b4"},
        {"title": "Best F1 Score", "value": f"{best_row['f1_score']:.4f}", "color": "#9467bd"},
        {"title": "Total Models Trained", "value": format_number(len(performance)), "color": "#ff7f0e"},
    ])

    st.divider()

    fig = px.bar(
        performance.sort_values("auc_score", ascending=True),
        x="auc_score", y="model_name", orientation="h",
        title="Model Comparison — AUC Score", color_discrete_sequence=["#1f77b4"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("ROC Curves — All Models")

    trainer = ModelTrainer()
    df = trainer.load_processed_data()
    trainer.split_data(df)

    roc_fig = go.Figure()
    for name in MODEL_NAMES:
        model = trainer.load_model(name)
        y_proba = model.predict_proba(trainer.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(trainer.y_test, y_proba)
        auc = roc_auc_score(trainer.y_test, y_proba)
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="gray")))
    roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance (Top 15)")
        importance_df = pd.read_csv("data/processed/rf_feature_importance.csv")
        st.plotly_chart(feature_importance_bar(importance_df), use_container_width=True)

    with col2:
        st.subheader(f"Confusion Matrix — {best_row['model_name']}")
        best_model = trainer.load_model(best_row["model_name"])
        evaluator = ModelEvaluator()
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        evaluator.plot_confusion_matrix(best_model, trainer.X_test, trainer.y_test, best_row["model_name"], ax=ax)
        st.pyplot(fig_cm)

    st.divider()
    st.subheader(f"SHAP Summary — {best_row['model_name']}")
    import shap
    X_test_float = trainer.X_test.astype(float)
    explainer = shap.TreeExplainer(best_model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_test_float)
    shap_fig = shap_summary_plot(shap_values, X_test_float)
    st.pyplot(shap_fig)


main()
