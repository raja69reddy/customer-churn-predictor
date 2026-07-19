# LinkedIn Post — Customer Churn Predictor

Ready-to-post announcement for the customer-churn-predictor project.

---

🚨 26% of telecom customers churn every year — and most companies only find out after they've already left.

**The problem:** Acquiring a new customer costs 5-25x more than retaining an existing one. Customer-success and marketing teams need to know *who* is about to churn and *why*, early enough to actually do something about it — not just a historical report of who already left.

**What I built:** An end-to-end machine learning system that predicts customer churn and tells you exactly why, in plain English.

- 🤖 Trained and compared **6 machine learning models** (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and a soft-voting Ensemble)
- 📊 Best model (tuned XGBoost) reached an **AUC of 0.96** on the held-out test set
- 🎯 Used **SHAP** to explain exactly which factors drive each prediction — not just a black-box score
- 🔮 Added **GPT-powered natural-language explanations** so a non-technical stakeholder can understand *why* a customer is flagged as high risk
- 📈 Tracked every experiment with **MLflow**, including full model registry and versioning
- 🚀 Shipped a **FastAPI** REST API with 17 endpoints for real-time and batch predictions
- 📱 Built an **interactive Streamlit dashboard** — churn overview, per-customer lookup, model performance, and a retention-targeting tool with a live campaign ROI estimator
- 🐳 **Dockerized** and deployment-ready, with a lightweight demo mode that runs entirely without a database

**Tech stack:** Python · PostgreSQL · scikit-learn · XGBoost · LightGBM · SHAP · MLflow · FastAPI · Streamlit · Docker · OpenAI GPT

The full source, architecture diagrams, and a project summary notebook are on GitHub:
🔗 https://github.com/raja69reddy/customer-churn-predictor

Would love feedback from anyone working on similar churn/retention problems, or anyone who wants to poke holes in the approach!

#DataScience #MachineLearning #Python #XGBoost #MLflow #DataAnalytics #Portfolio
