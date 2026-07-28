"""Alert/notification management — logs alerts for high-risk customers, model drift, pipeline
failures, and weekly digests, and persists them to the alerts table.

"Sends" here means logs (console + file via src.utils.logging_config, same pattern this
project has used since Day 22) and writes a row to the alerts table — there's no real
email/Slack/webhook integration. This is consistent with every other "send_*"/"digest" function
in this project (e.g. src/pipelines/weekly_pipeline.py::send_weekly_digest() saves a file
rather than actually emailing anyone).
"""

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("notifier")


def _build_alert(alert_type: str, severity: str, message: str) -> dict:
    return {"alert_type": alert_type, "severity": severity, "message": message}


def send_high_risk_alert(customer_list) -> dict:
    """Logs (and saves) an alert naming customers newly flagged High risk. customer_list can
    be a plain list of customer_ids or a DataFrame with a customer_id column (e.g. the output
    of src/pipelines/weekly_pipeline.py::identify_new_high_risk())."""
    if hasattr(customer_list, "columns") and "customer_id" in customer_list.columns:
        ids = list(customer_list["customer_id"])
    else:
        ids = list(customer_list)

    if not ids:
        message = "No new high-risk customers to alert on."
        severity = "info"
        log.info(message)
    else:
        preview = ", ".join(ids[:10])
        if len(ids) > 10:
            preview += f", and {len(ids) - 10} more"
        message = f"{len(ids)} customer(s) newly flagged High risk: {preview}"
        severity = "warning"
        log.warning(message)

    alert = _build_alert("high_risk", severity, message)
    save_notifications_to_db([alert])
    return alert


def send_model_drift_alert(psi_score: float, threshold: float = 0.2) -> dict:
    """Logs (and saves) a drift alert for a given PSI score against threshold (matches
    src/monitoring/model_monitor.py's DEFAULT_ALERT_THRESHOLD default of 0.2)."""
    is_significant = psi_score > threshold
    severity = "critical" if is_significant else "info"
    message = f"Model drift PSI={psi_score:.4f} (threshold={threshold}) — " + (
        "significant drift detected." if is_significant else "within threshold."
    )

    if is_significant:
        log.critical(message)
    else:
        log.info(message)

    alert = _build_alert("model_drift", severity, message)
    save_notifications_to_db([alert])
    return alert


def send_pipeline_failure_alert(error) -> dict:
    """Logs (and saves) a critical alert for a pipeline failure. `error` can be an exception
    or a plain string."""
    message = f"Pipeline failure: {error}"
    log.critical(message)

    alert = _build_alert("pipeline_failure", "critical", message)
    save_notifications_to_db([alert])
    return alert


def send_weekly_digest_alert(report: dict) -> dict:
    """Logs (and saves) a summary alert from a src.pipelines.weekly_pipeline.py-shaped weekly
    report dict (either run_weekly_summary()'s full result or its 'week_over_week' sub-dict).
    """
    wow = report.get("week_over_week", report)
    message = (
        f"Weekly digest: {wow.get('this_week_high_risk', 'n/a')} high-risk customers "
        f"(avg probability delta {wow.get('avg_probability_delta', 'n/a')})"
    )
    log.info(message)

    alert = _build_alert("weekly_digest", "info", message)
    save_notifications_to_db([alert])
    return alert


def save_notifications_to_db(alerts: list) -> int:
    """Inserts a list of alert dicts (each with alert_type, severity, message keys) into the
    alerts table. Returns the number of rows inserted."""
    if not alerts:
        return 0

    engine = get_engine()
    with engine.begin() as conn:
        for alert in alerts:
            conn.execute(
                text(
                    "INSERT INTO alerts (alert_type, severity, message) "
                    "VALUES (:alert_type, :severity, :message)"
                ),
                alert,
            )
    log.info("Saved %d alert(s) to the alerts table", len(alerts))
    return len(alerts)


def get_notification_history(limit: int = 20) -> pd.DataFrame:
    """Returns the most recent `limit` alerts, newest first."""
    engine = get_engine()
    return pd.read_sql(
        text("SELECT * FROM alerts ORDER BY created_at DESC LIMIT :limit"),
        engine,
        params={"limit": limit},
    )


if __name__ == "__main__":
    print("=== send_high_risk_alert (empty) ===")
    print(send_high_risk_alert([]))

    print("\n=== send_high_risk_alert (sample) ===")
    print(send_high_risk_alert(["CUST-00001", "CUST-00002", "CUST-00003"]))

    print("\n=== send_model_drift_alert (stable) ===")
    print(send_model_drift_alert(0.03))

    print("\n=== send_model_drift_alert (significant) ===")
    print(send_model_drift_alert(0.35))

    print("\n=== send_pipeline_failure_alert ===")
    print(send_pipeline_failure_alert("Database connection timed out"))

    print("\n=== send_weekly_digest_alert ===")
    print(
        send_weekly_digest_alert(
            {
                "week_over_week": {
                    "this_week_high_risk": 1341,
                    "avg_probability_delta": 0.0,
                }
            }
        )
    )

    print("\n=== get_notification_history ===")
    print(get_notification_history(limit=10).to_string(index=False))
