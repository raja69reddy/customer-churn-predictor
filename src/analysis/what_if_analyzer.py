"""What-if scenario analysis — re-scores a customer under a hypothetical change and reports
the resulting churn probability shift.

Non-obvious data note: the model's feature set (see src/models/predict.py::FEATURE_COLUMNS)
does NOT use raw_customers.monthly_charges directly — `charge_per_month` (the engineered
feature that actually feeds the model) is computed as `total_charges / tenure`
(src/features/engineering.py::create_charge_per_month). So what_if_reduce_charges() adjusts
total_charges (and monthly_charges, for a consistent customer profile), not monthly_charges
alone — changing monthly_charges by itself would have zero effect on the prediction.

Requires the full ML stack (same as churn_drivers.py) — not demo-mode safe.
"""

import pandas as pd

from src.models.predict import ChurnPredictor
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

log = setup_logging("what_if_analyzer")

CUSTOMER_INPUT_FIELDS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
]

_predictor = ChurnPredictor()


def _load_customer(customer_id: str) -> dict:
    engine = get_engine()
    df = pd.read_sql(
        "SELECT * FROM raw_customers WHERE customer_id = %(cid)s",
        engine,
        params={"cid": customer_id},
    )
    if df.empty:
        raise ValueError(f"Customer '{customer_id}' not found.")
    return {f: df.iloc[0][f] for f in CUSTOMER_INPUT_FIELDS}


def _score(customer_dict: dict) -> dict:
    # ChurnPredictor.predict_single() lazily calls load_best_model() itself if needed.
    return _predictor.predict_single(customer_dict)


def _scenario_result(
    customer_id: str, scenario_name: str, baseline: dict, modified_dict: dict
) -> dict:
    new_result = _score(modified_dict)
    delta = new_result["churn_probability"] - baseline["churn_probability"]
    return {
        "customer_id": customer_id,
        "scenario": scenario_name,
        "original_probability": round(baseline["churn_probability"], 4),
        "new_probability": round(new_result["churn_probability"], 4),
        "probability_change": round(delta, 4),
        "original_segment": baseline["risk_segment"],
        "new_segment": new_result["risk_segment"],
    }


def what_if_contract_change(customer_id: str, new_contract: str = "One year") -> dict:
    """What if this customer switched from their current contract to `new_contract`
    ('One year' or 'Two year')?"""
    customer = _load_customer(customer_id)
    baseline = _score(customer)

    modified = dict(customer)
    modified["contract"] = new_contract

    result = _scenario_result(
        customer_id, f"Switch to {new_contract} contract", baseline, modified
    )
    log.info(
        "what_if_contract_change(%s -> %s): %.4f -> %.4f",
        customer_id,
        new_contract,
        result["original_probability"],
        result["new_probability"],
    )
    return result


def what_if_add_security(customer_id: str) -> dict:
    """What if this customer added online security to their plan?"""
    customer = _load_customer(customer_id)
    baseline = _score(customer)

    if customer["internet_service"] == "No":
        log.warning(
            "%s has no internet service — online security isn't applicable, scenario is a no-op",
            customer_id,
        )

    modified = dict(customer)
    modified["online_security"] = "Yes"

    result = _scenario_result(customer_id, "Add online security", baseline, modified)
    log.info(
        "what_if_add_security(%s): %.4f -> %.4f",
        customer_id,
        result["original_probability"],
        result["new_probability"],
    )
    return result


def what_if_reduce_charges(customer_id: str, reduction_pct: float = 10.0) -> dict:
    """What if this customer's charges were reduced by reduction_pct%?

    Reduces both total_charges and monthly_charges by reduction_pct for a consistent
    profile — total_charges is what actually drives the model (via the charge_per_month
    engineered feature), see module docstring.
    """
    customer = _load_customer(customer_id)
    baseline = _score(customer)

    factor = 1 - (reduction_pct / 100.0)
    modified = dict(customer)
    modified["total_charges"] = customer["total_charges"] * factor
    modified["monthly_charges"] = customer["monthly_charges"] * factor

    result = _scenario_result(
        customer_id, f"Reduce charges by {reduction_pct:.0f}%", baseline, modified
    )
    log.info(
        "what_if_reduce_charges(%s, %.0f%%): %.4f -> %.4f",
        customer_id,
        reduction_pct,
        result["original_probability"],
        result["new_probability"],
    )
    return result


def compare_scenarios(customer_id: str, scenarios: list = None) -> pd.DataFrame:
    """Runs a set of what-if scenarios for a customer and returns a comparison table.

    scenarios defaults to the three built-in what-ifs (annual contract, add security,
    10% charge reduction). Pass a list of (label, callable) tuples to customize, e.g.
    [("Two year contract", lambda cid: what_if_contract_change(cid, "Two year"))].
    """
    if scenarios is None:
        scenarios = [
            (
                "Switch to One year contract",
                lambda cid: what_if_contract_change(cid, "One year"),
            ),
            ("Add online security", what_if_add_security),
            ("Reduce charges by 10%", lambda cid: what_if_reduce_charges(cid, 10.0)),
        ]

    rows = [fn(customer_id) for _, fn in scenarios]
    table = pd.DataFrame(rows)
    log.info("Compared %d scenarios for %s", len(table), customer_id)
    return table


if __name__ == "__main__":
    import sys

    customer_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not customer_id:
        print("Usage: python -m src.analysis.what_if_analyzer <customer_id>")
        raise SystemExit(1)

    print(f"What-if scenarios for {customer_id}:\n")
    print(compare_scenarios(customer_id).to_string(index=False))
