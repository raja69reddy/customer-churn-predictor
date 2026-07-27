"""Customer segmentation — value, risk, and tenure segments, plus a Value x Risk 2D matrix for
prioritizing retention spend (the standard "who do we save first" framework: high value + high
risk customers are the most urgent, low value + low risk customers need no action).

Follows the Day 24 "optional df param, pure pandas/SQL" pattern (like revenue_analysis.py) so
every function is demo-mode safe when a pre-loaded df (e.g. demo_mode.get_demo_predictions())
is passed in — no heavy ML dependencies.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.db import get_engine  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

log = setup_logging("customer_segmentation")

OUTPUT_DIR = "data/processed"
CLV_MONTH_CAP = (
    60  # matches revenue_analysis.py's CLV formula, kept consistent project-wide
)
TENURE_BINS = [0, 12, 24, 48, float("inf")]
TENURE_LABELS = [
    "New Customer",
    "Growing Customer",
    "Established Customer",
    "Loyal Customer",
]
VALUE_ORDER = ["Low", "Medium", "High"]
RISK_ORDER = ["Low", "Medium", "High"]


def _load_customers() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(
        "SELECT cp.customer_id, cp.churn_probability, cp.risk_segment, rc.tenure, "
        "rc.monthly_charges, rc.contract "
        "FROM churn_predictions cp JOIN raw_customers rc ON rc.customer_id = cp.customer_id",
        engine,
    )


def segment_by_value(df: pd.DataFrame = None) -> pd.DataFrame:
    """Adds a 'value_segment' column (High/Medium/Low) based on monthly_charges terciles —
    today's revenue size, independent of predicted future churn.

    Also computes 'clv' (monthly_charges * min(60, 1/churn_probability), same formula as
    src/analysis/revenue_analysis.py) for reporting, but does NOT segment on it: CLV already
    bakes churn_probability into itself, so using it as the "value" axis of a Value x Risk
    matrix would make the two axes non-independent — every High risk customer would
    automatically fall into "Low value" (1/churn_probability shrinks as risk rises), collapsing
    the matrix onto one row regardless of how much they actually pay. Confirmed this
    empirically: a first pass using CLV put 100% of Medium/High risk customers in the Low value
    row. monthly_charges avoids that confound and produces a real 2D spread."""
    if df is None:
        df = _load_customers()

    df = df.copy()
    expected_months = (1.0 / df["churn_probability"].replace(0, pd.NA)).clip(
        upper=CLV_MONTH_CAP
    )
    df["clv"] = df["monthly_charges"] * expected_months.fillna(CLV_MONTH_CAP)
    df["value_segment"] = pd.qcut(df["monthly_charges"], q=3, labels=VALUE_ORDER)

    log.info("Segmented %d customers by value (monthly_charges terciles)", len(df))
    return df


def segment_by_risk(df: pd.DataFrame = None) -> pd.DataFrame:
    """Adds a 'risk_segment' column — reuses the model's existing High/Medium/Low
    risk_segment (src/models/predict.py) rather than re-deriving it."""
    if df is None:
        df = _load_customers()

    df = df.copy()
    if "risk_segment" not in df.columns:
        raise ValueError("df must already contain a 'risk_segment' column.")

    log.info("Segmented %d customers by risk (existing model risk_segment)", len(df))
    return df


def segment_by_tenure(df: pd.DataFrame = None) -> pd.DataFrame:
    """Adds a 'tenure_segment' column (New/Growing/Established/Loyal Customer), using the
    same bins as the model's engineered tenure_group feature
    (src/features/engineering.py::TENURE_BINS/TENURE_LABELS)."""
    if df is None:
        df = _load_customers()

    df = df.copy()
    df["tenure_segment"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    )

    log.info("Segmented %d customers by tenure", len(df))
    return df


def create_2d_segments(value: pd.Series, risk: pd.Series) -> pd.DataFrame:
    """Cross-tabulates two aligned segment Series (from segment_by_value/segment_by_risk)
    into a Value x Risk matrix of customer counts."""
    matrix = pd.crosstab(value, risk)
    matrix = matrix.reindex(index=VALUE_ORDER, columns=RISK_ORDER, fill_value=0)

    log.info("Built %dx%d value-risk segment matrix", *matrix.shape)
    return matrix


def plot_segment_matrix(
    matrix: pd.DataFrame = None, output_dir: str = OUTPUT_DIR
) -> str:
    """Saves the Value x Risk matrix as a heatmap PNG. Returns the saved path."""
    if matrix is None:
        df = segment_by_risk(segment_by_value())
        matrix = create_2d_segments(df["value_segment"], df["risk_segment"])

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values, cmap="Reds")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Risk Segment")
    ax.set_ylabel("Value Segment")
    ax.set_title("Customer Segmentation Matrix (Value x Risk)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, int(matrix.values[i, j]), ha="center", va="center", color="black"
            )

    fig.colorbar(im, ax=ax, label="Customer count")
    plt.tight_layout()

    path = os.path.join(output_dir, "segment_matrix.png")
    plt.savefig(path)
    plt.close(fig)

    log.info("Saved segment matrix heatmap to %s", path)
    return path


def save_segmentation_report(output_dir: str = OUTPUT_DIR) -> dict:
    """Runs the full segmentation (value, risk, tenure, 2D matrix) and saves a CSV of every
    customer's segments + a markdown summary + the heatmap PNG to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    df = _load_customers()
    df = segment_by_value(df)
    df = segment_by_risk(df)
    df = segment_by_tenure(df)

    matrix = create_2d_segments(df["value_segment"], df["risk_segment"])
    heatmap_path = plot_segment_matrix(matrix, output_dir)

    customers_csv = os.path.join(output_dir, "customer_segments.csv")
    df[
        [
            "customer_id",
            "clv",
            "value_segment",
            "risk_segment",
            "tenure_segment",
            "monthly_charges",
            "churn_probability",
        ]
    ].to_csv(customers_csv, index=False)

    matrix_csv = os.path.join(output_dir, "segment_matrix.csv")
    matrix.to_csv(matrix_csv)

    lines = [
        "# Customer Segmentation Report",
        "",
        "## Value x Risk Matrix (customer counts)",
        "",
    ]
    lines.append(matrix.to_string())
    lines.append("")
    lines.append("## Segment Sizes")
    for col, order in (("value_segment", VALUE_ORDER), ("risk_segment", RISK_ORDER)):
        lines.append(f"### {col}")
        counts = df[col].value_counts().reindex(order, fill_value=0)
        for segment, count in counts.items():
            lines.append(f"- {segment}: {count}")
        lines.append("")
    lines.append("### tenure_segment")
    for segment, count in (
        df["tenure_segment"].value_counts().reindex(TENURE_LABELS, fill_value=0).items()
    ):
        lines.append(f"- {segment}: {count}")

    report_path = os.path.join(output_dir, "segmentation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved segmentation report to %s", report_path)
    return {
        "customers_csv": customers_csv,
        "matrix_csv": matrix_csv,
        "report_path": report_path,
        "heatmap_path": heatmap_path,
    }


if __name__ == "__main__":
    df = _load_customers()
    df = segment_by_value(df)
    df = segment_by_risk(df)
    df = segment_by_tenure(df)

    print("=== Value Segment Sizes ===")
    print(df["value_segment"].value_counts().reindex(VALUE_ORDER, fill_value=0))

    print("\n=== Risk Segment Sizes ===")
    print(df["risk_segment"].value_counts().reindex(RISK_ORDER, fill_value=0))

    print("\n=== Tenure Segment Sizes ===")
    print(df["tenure_segment"].value_counts().reindex(TENURE_LABELS, fill_value=0))

    matrix = create_2d_segments(df["value_segment"], df["risk_segment"])
    print("\n=== Value x Risk Matrix ===")
    print(matrix)

    result = save_segmentation_report()
    print(f"\nSaved segmentation report to {result['report_path']}")
    print(f"Saved heatmap to {result['heatmap_path']}")
