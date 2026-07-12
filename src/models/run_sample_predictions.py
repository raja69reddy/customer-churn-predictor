"""Runs ChurnPredictor against 5 representative sample customers."""

from src.models.predict import ChurnPredictor

SAMPLE_CUSTOMERS = {
    "New month-to-month, fiber, no add-ons (expected high risk)": {
        "tenure": 2,
        "monthly_charges": 90.0,
        "total_charges": 180.0,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "phone_service": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
    },
    "Long-tenure, two-year contract, autopay (expected low risk)": {
        "tenure": 65,
        "monthly_charges": 45.0,
        "total_charges": 2900.0,
        "contract": "Two year",
        "payment_method": "Bank transfer (automatic)",
        "phone_service": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
    },
    "Mid-tenure, one-year contract, mixed services (expected medium risk)": {
        "tenure": 20,
        "monthly_charges": 65.0,
        "total_charges": 1300.0,
        "contract": "One year",
        "payment_method": "Mailed check",
        "phone_service": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
    },
    "New customer, month-to-month, no internet (expected lower-than-typical-new risk)": {
        "tenure": 4,
        "monthly_charges": 22.0,
        "total_charges": 88.0,
        "contract": "Month-to-month",
        "payment_method": "Mailed check",
        "phone_service": "Yes",
        "internet_service": "No",
        "online_security": "No internet service",
        "online_backup": "No internet service",
        "device_protection": "No internet service",
        "tech_support": "No internet service",
        "streaming_tv": "No internet service",
        "streaming_movies": "No internet service",
    },
    "Established customer, month-to-month, high spend (expected medium-high risk)": {
        "tenure": 30,
        "monthly_charges": 100.0,
        "total_charges": 3000.0,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "phone_service": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
    },
}


def run() -> None:
    predictor = ChurnPredictor()
    predictor.load_best_model()
    print(f"Using best model: {predictor.model_name}\n")

    for label, customer in SAMPLE_CUSTOMERS.items():
        result = predictor.predict_single(customer)
        factors = predictor.explain_prediction(customer, top_n=3)

        print(f"=== {label} ===")
        print(f"    Churn probability: {result['churn_probability']:.4f}")
        print(f"    Risk segment: {result['risk_segment']}")
        print("    Top 3 factors:")
        for i, factor in enumerate(factors, start=1):
            print(
                f"        {i}. {factor['feature']} ({factor['direction']}, SHAP={factor['shap_value']:+.4f})"
            )
        print()


if __name__ == "__main__":
    run()
