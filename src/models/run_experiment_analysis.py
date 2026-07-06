"""Runs the full experiment analysis: best run, comparison table, AUC plot, CSV export."""
from src.models.experiment_analyzer import (
    export_run_summary,
    get_best_run,
    get_run_history,
    plot_metric_comparison,
)


def run() -> None:
    best = get_best_run()
    print(f"Best run: {best['tags.mlflow.runName']} (run_id={best['run_id']}, AUC={best['metrics.auc']:.4f})")

    history = get_run_history()
    print("\nRun comparison table:")
    print(
        history[["run_id", "tags.mlflow.runName", "metrics.accuracy", "metrics.auc", "metrics.f1"]]
        .rename(columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "run_name"))
        .to_string(index=False)
    )

    plot_path = plot_metric_comparison("auc")
    print(f"\nSaved AUC comparison plot to {plot_path}")

    csv_path = export_run_summary("data/processed/mlflow_summary.csv")
    print(f"Exported run summary to {csv_path}")

    top3 = history.head(3)
    print("\nTop 3 best performing runs:")
    for i, (_, row) in enumerate(top3.iterrows(), start=1):
        print(f"    {i}. {row['tags.mlflow.runName']} — AUC={row['metrics.auc']:.4f} (run_id={row['run_id']})")


if __name__ == "__main__":
    run()
