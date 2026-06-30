"""GPT-powered churn explanation generator."""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def explain_churn(customer: dict, shap_values: dict) -> str:
    """Return a plain-English explanation of why a customer is at risk of churning."""
    top_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    factors_text = "\n".join(f"- {k}: {v:+.3f}" for k, v in top_factors)

    prompt = (
        f"A telecom customer has a churn probability of {customer.get('churn_probability', 'N/A'):.1%}. "
        f"Their profile: tenure={customer.get('tenure')} months, "
        f"contract={customer.get('contract')}, "
        f"monthly_charges=${customer.get('monthly_charges', 0):.2f}, "
        f"internet_service={customer.get('internet_service')}.\n\n"
        f"Top SHAP factors driving this prediction:\n{factors_text}\n\n"
        "In 2–3 concise sentences, explain to a customer-success manager why this customer "
        "is at risk and what action they should take."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()
