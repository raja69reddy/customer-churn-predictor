"""Model Performance page — leaderboard, ROC curves, feature importance, SHAP, confusion matrix.

The ROC/SHAP/confusion-matrix sections need live trained models plus sklearn/matplotlib/shap,
none of which are in the lightweight requirements_streamlit.txt used for cloud deploys — those
sections are skipped (with an explanatory message) whenever demo mode is active or those
packages simply aren't installed.
"""

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
from dashboard.components.charts import feature_importance_bar  # noqa: E402
from dashboard.components.metrics import display_kpi_row, format_number  # noqa: E402
from src.utils.db import get_engine  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from sklearn.metrics import roc_auc_score, roc_curve

    from dashboard.components.charts import shap_summary_plot
    from src.models.evaluate import ModelEvaluator
    from src.models.train import ModelTrainer

    LIVE_MODEL_LIBS_AVAILABLE = True
except ImportError:
    LIVE_MODEL_LIBS_AVAILABLE = False

st.set_page_config(page_title="Model Performance", page_icon="🔮", layout="wide")

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest_tuned",
    "xgboost_tuned",
    "lightgbm",
    "ensemble",
]
CACHE_TTL_SECONDS = 300


def render_cache_controls() -> None:
    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = datetime.now()

    st.sidebar.markdown("### Data")
    if st.sidebar.button("Clear Cache", key="clear_cache_model_performance"):
        st.cache_data.clear()
        st.session_state["last_updated"] = datetime.now()
        st.rerun()

    st.sidebar.caption(
        f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_model_performance() -> pd.DataFrame:
    if demo_mode.is_demo_mode():
        return demo_mode.get_demo_model_metrics()
    engine = get_engine()
    return pd.read_sql("SELECT * FROM vw_model_performance", engine)


def main() -> None:
    is_demo = demo_mode.is_demo_mode()
    if is_demo:
        st.warning(
            "🟡 **Demo Mode** — showing sample data (no live database connection)."
        )
    else:
        st.success("🟢 **Live Mode** — connected to the database.")

    render_cache_controls()

    st.title("Model Performance")

    try:
        with st.spinner("Loading model performance..."):
            performance = load_model_performance()
    except Exception:
        st.error(
            "⚠️ Could not connect to the database. Please check your connection and try again."
        )
        return

    if performance.empty:
        st.warning("No data available — no models have been registered yet.")
        return

    best_row = performance.loc[performance["auc_score"].idxmax()]

    display_kpi_row(
        [
            {
                "title": "Best Model",
                "value": best_row["model_name"],
                "color": "#2ca02c",
            },
            {
                "title": "Best AUC Score",
                "value": f"{best_row['auc_score']:.4f}",
                "color": "#1f77b4",
            },
            {
                "title": "Best F1 Score",
                "value": f"{best_row['f1_score']:.4f}",
                "color": "#9467bd",
            },
            {
                "title": "Total Models Trained",
                "value": format_number(len(performance)),
                "color": "#ff7f0e",
            },
        ]
    )

    st.divider()

    fig = px.bar(
        performance.sort_values("auc_score", ascending=True),
        x="auc_score",
        y="model_name",
        orientation="h",
        title="Model Comparison — AUC Score",
        color_discrete_sequence=["#1f77b4"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("ROC Curves — All Models")

    if is_demo or not LIVE_MODEL_LIBS_AVAILABLE:
        st.info(
            "ℹ️ ROC curves, live feature importance, confusion matrix, and SHAP analysis "
            "require trained model files and ML libraries (scikit-learn, matplotlib, shap) "
            "that aren't available in demo mode / the lightweight cloud deployment. "
            "Run this dashboard locally with the full `requirements.txt` and a populated "
            "database to see these sections."
        )
        return

    try:
        with st.spinner("Loading model performance..."):
            trainer = ModelTrainer()
            df = trainer.load_processed_data()
            trainer.split_data(df)

            roc_fig = go.Figure()
            for name in MODEL_NAMES:
                model = trainer.load_model(name)
                y_proba = model.predict_proba(trainer.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(trainer.y_test, y_proba)
                auc = roc_auc_score(trainer.y_test, y_proba)
                roc_fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"
                    )
                )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Guess",
                    line=dict(dash="dash", color="gray"),
                )
            )
            roc_fig.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
            )
        st.plotly_chart(roc_fig, use_container_width=True)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Importance (Top 15)")
            importance_df = pd.read_csv("data/processed/rf_feature_importance.csv")
            st.plotly_chart(
                feature_importance_bar(importance_df), use_container_width=True
            )

        with col2:
            st.subheader(f"Confusion Matrix — {best_row['model_name']}")
            best_model = trainer.load_model(best_row["model_name"])
            evaluator = ModelEvaluator()
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            evaluator.plot_confusion_matrix(
                best_model,
                trainer.X_test,
                trainer.y_test,
                best_row["model_name"],
                ax=ax,
            )
            st.pyplot(fig_cm)

        st.divider()
        st.subheader(f"SHAP Summary — {best_row['model_name']}")
        with st.spinner("Loading model performance..."):
            import shap

            X_test_float = trainer.X_test.astype(float)
            explainer = shap.TreeExplainer(
                best_model, feature_perturbation="tree_path_dependent"
            )
            shap_values = explainer(X_test_float)
            shap_fig = shap_summary_plot(shap_values, X_test_float)
        st.pyplot(shap_fig)
    except FileNotFoundError as e:
        st.warning(
            f"No data available — a required model or feature-importance file is missing: {e}"
        )
    except Exception as e:
        st.error(f"⚠️ Could not load model performance details: {e}")


main()
