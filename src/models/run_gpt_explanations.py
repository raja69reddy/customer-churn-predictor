"""Tests GPT-powered churn explanations against 3 high-risk customer profiles."""
from src.models.predict import ChurnPredictor

HIGH_RISK_CUSTOMERS = {
    "New fiber customer, no add-ons, electronic check": {
        "tenure": 2, "monthly_charges": 95.0, "total_charges": 190.0,
        "contract": "Month-to-month", "payment_method": "Electronic check",
        "phone_service": "Yes", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "Yes",
    },
    "New fiber customer, high spend, mailed check": {
        "tenure": 1, "monthly_charges": 105.0, "total_charges": 105.0,
        "contract": "Month-to-month", "payment_method": "Mailed check",
        "phone_service": "Yes", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "Yes",
        "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "Yes",
    },
    "New customer, DSL, no security or tech support": {
        "tenure": 4, "monthly_charges": 80.0, "total_charges": 320.0,
        "contract": "Month-to-month", "payment_method": "Electronic check",
        "phone_service": "Yes", "internet_service": "DSL",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    },
}


def run() -> None:
    predictor = ChurnPredictor()
    predictor.load_best_model()
    print(f"Using best model: {predictor.model_name}\n")

    for label, customer in HIGH_RISK_CUSTOMERS.items():
        result = predictor.predict_single(customer)
        explanation = predictor.explain_with_gpt(customer, result)
        formatted = predictor.format_explanation(explanation, result)

        print(f"=== {label} ===")
        print(f"Churn probability: {result['churn_probability']:.4f}")
        print(formatted)
        print()


if __name__ == "__main__":
    run()
