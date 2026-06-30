"""Generate 7,043 realistic Telco-style customer churn records."""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 7043
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "customers.csv")


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- demographics ---
    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # --- tenure & contract ---
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    # tenure distribution differs by contract type
    tenure = np.empty(n, dtype=int)
    for i, c in enumerate(contract):
        if c == "Month-to-month":
            tenure[i] = int(rng.integers(1, 25))
        elif c == "One year":
            tenure[i] = int(rng.integers(12, 48))
        else:
            tenure[i] = int(rng.integers(24, 73))
    tenure = np.clip(tenure, 1, 72)

    # --- services ---
    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    def internet_addon(has_internet, yes_prob=0.29):
        return np.where(
            has_internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], n, p=[yes_prob, 1 - yes_prob]),
        )

    online_security = internet_addon(internet_service, 0.29)
    online_backup = internet_addon(internet_service, 0.34)
    device_protection = internet_addon(internet_service, 0.34)
    tech_support = internet_addon(internet_service, 0.29)
    streaming_tv = internet_addon(internet_service, 0.38)
    streaming_movies = internet_addon(internet_service, 0.39)

    # --- billing ---
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # monthly charges depend on services
    base = np.where(internet_service == "No", 20.0, np.where(internet_service == "DSL", 45.0, 70.0))
    addon_count = (
        (online_security == "Yes").astype(float)
        + (online_backup == "Yes").astype(float)
        + (device_protection == "Yes").astype(float)
        + (tech_support == "Yes").astype(float)
        + (streaming_tv == "Yes").astype(float)
        + (streaming_movies == "Yes").astype(float)
        + (multiple_lines == "Yes").astype(float)
    )
    monthly_charges = np.clip(base + addon_count * 7.0 + rng.normal(0, 3, n), 18.0, 118.0).round(2)
    total_charges = np.clip(
        monthly_charges * tenure + rng.normal(0, tenure * 2, n), 18.0, None
    ).round(2)

    # --- churn (~26%) ---
    # higher churn probability for month-to-month, low tenure, fiber, electronic check
    churn_score = (
        (contract == "Month-to-month").astype(float) * 0.30
        + (tenure < 12).astype(float) * 0.20
        + (internet_service == "Fiber optic").astype(float) * 0.10
        + (payment_method == "Electronic check").astype(float) * 0.08
        + (online_security == "No").astype(float) * 0.05
        + (tech_support == "No").astype(float) * 0.05
        + rng.uniform(0, 0.22, n)
    )
    churn_prob = churn_score / churn_score.max()
    # scale to hit ~26% overall
    threshold = np.percentile(churn_prob, 74)
    churn = np.where(churn_prob >= threshold, "Yes", "No")

    customer_ids = [f"CUST-{i+1:05d}" for i in range(n)]

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "raw"), exist_ok=True)
    df = generate()
    out = os.path.normpath(OUTPUT)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows saved to data/raw/customers.csv")
