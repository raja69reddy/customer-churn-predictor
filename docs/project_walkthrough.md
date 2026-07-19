# Project Walkthrough Script — Customer Churn Predictor

A ~5 minute spoken walkthrough for demos, interviews, or portfolio reviews. Read it
conversationally — it's written to be spoken, not read verbatim.

---

## Section 1 — Business Problem (30 seconds)

> "This is an end-to-end churn prediction system for a telecom company. The core problem is
> that retaining an existing customer is a lot cheaper than acquiring a new one, but by the
> time a customer actually cancels, it's too late to do anything about it. So the goal here
> is to flag customers who are *likely* to churn in the next billing cycle, while there's
> still time for a retention team to act — a discount, a call, a plan change — before they
> leave."

---

## Section 2 — Data and EDA (1 minute)

> "I'm using the Telco Customer Churn dataset — about 7,000 customers with fields like
> tenure, contract type, monthly charges, payment method, and which services they've signed
> up for, plus the ground-truth churn label. Roughly 26% of customers in the dataset
> churned, which is realistic for this industry.
>
> The EDA turned up a few clear patterns: month-to-month contract customers churn at almost
> 48%, versus close to zero for one- and two-year contracts. Customers with less than a year
> of tenure are by far the highest-risk group. And customers paying by electronic check
> churn more than any other payment method. Those three signals — contract type, tenure, and
> payment method — end up being some of the strongest predictors later on, which is a good
> sanity check that the model is learning something real, not noise."

---

## Section 3 — Feature Engineering (30 seconds)

> "From the raw fields I engineered a handful of derived features: a tenure-group bucket,
> a charge-per-month figure, a count of how many add-on services each customer has, and two
> composite risk scores — one for contract type, one for payment method — that encode the
> EDA findings numerically. I also used SMOTE to balance the training data, since a 26/74
> split is imbalanced enough that a naive model would just predict 'no churn' most of the
> time and still look accurate."

---

## Section 4 — Model Training and Results (1 minute)

> "I trained and compared six models: Logistic Regression, Decision Tree, Random Forest,
> XGBoost, LightGBM, and a soft-voting Ensemble of the three strongest. Everything is logged
> to MLflow — parameters, metrics, and the model artifacts themselves — so I can compare runs
> and roll back if a new model underperforms.
>
> After hyperparameter tuning, XGBoost came out on top with an AUC of 0.96 on the held-out
> test set, about 0.83 recall and 0.79 precision. That model gets promoted to 'Production' in
> the MLflow Model Registry, and that's the exact model both the API and the dashboard load
> at runtime — so there's one source of truth for what's actually live, not a model file
> floating around disconnected from what's deployed."

---

## Section 5 — Dashboard and API Demo (1 minute)

*(Live demo cue — switch to the running app here.)*

> "There are two ways to interact with this. First, a FastAPI REST API — you can POST a
> customer's details to `/predict` and get back a churn probability, a risk segment, the
> top SHAP factors driving that score, and recommended retention actions, all in one
> response. There's also a batch endpoint, and endpoints for looking up existing customers
> by ID.
>
> Second, a Streamlit dashboard with five pages: an overview with churn trends, a
> predictions page with a probability distribution and a live churn-probability gauge, a
> customer lookup page where you type in an ID and get the full explanation — including a
> GPT-generated plain-English summary of *why* that customer is at risk — a model
> performance page with an interactive model selector and SHAP analysis, and a retention
> targeting page with priority tiers and a campaign ROI estimator you can actually play with.
>
> One thing I'll point out: the dashboard also runs in a 'demo mode' with zero database
> setup — useful for a quick public demo without exposing a real database."

---

## Section 6 — Key Findings and Next Steps (1 minute)

> "The biggest takeaway from the SHAP analysis is that three features dominate the model's
> predictions: charge per month, being a brand-new customer, and the contract-risk score —
> which lines up exactly with what the EDA showed. So the actionable recommendation for the
> business is straightforward: prioritize retention outreach on new, high-paying,
> month-to-month customers, and incentivize them toward longer contracts.
>
> As for next steps — I'd want to add real-time streaming scoring with Kafka instead of
> batch runs, build an actual A/B test to measure whether the recommended retention actions
> move the needle, and set up an automated retraining pipeline with something like Airflow so
> the model doesn't go stale. It's all documented in the README's Future Work section. Happy
> to go deeper into any part of this — the code, the model choices, or the deployment setup."

---

**Total runtime:** ~5 minutes (30s + 60s + 30s + 60s + 60s + 60s)
