# Interview Prep — Customer Churn Predictor

Five likely interview questions about this project, answered in STAR format
(Situation, Task, Action, Result).

---

## Q1: Walk me through your churn prediction project.

**Situation:** Telecom companies lose roughly a quarter of their customer base to churn every
year, and acquiring a replacement customer costs far more than retaining an existing one.
Customer-success teams needed a way to identify at-risk customers *before* they cancel, with
enough explanation to act on it.

**Task:** Build an end-to-end system — from raw data to a usable product — that scores every
customer's churn risk, explains why, and surfaces that to both technical and non-technical
users.

**Action:** I built the full pipeline: ingestion of 7,043 customer records into PostgreSQL,
feature engineering (tenure buckets, contract/payment risk scores, services count), training
and comparing 6 models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM,
and a soft-voting Ensemble) with MLflow experiment tracking, SHAP for explainability, and a
GPT-powered natural-language layer on top of the SHAP output. I then shipped it two ways: a
FastAPI REST API for programmatic/batch scoring, and a Streamlit dashboard for interactive use
— both reading from the same MLflow-registered production model.

**Result:** The best model (tuned XGBoost) reached an AUC of 0.96 on the held-out test set.
The dashboard and API are containerized with Docker, and the dashboard also runs in a
dependency-light "demo mode" with no database required, so it can be deployed to Streamlit
Community Cloud from a fresh clone.

---

## Q2: Why did you choose XGBoost over other models?

**Situation:** I trained 4 baseline models first — Logistic Regression, Decision Tree, Random
Forest, and XGBoost — to establish a fair comparison before investing time in tuning.

**Task:** Pick the model to tune further and ultimately put into production, balancing
predictive performance against interpretability and inference cost.

**Action:** XGBoost had the best baseline AUC (0.9604) essentially tied with Logistic
Regression (0.9603), but XGBoost responded much better to hyperparameter tuning via
GridSearchCV — its tuned AUC improved to 0.9624 vs. Random Forest's tuned AUC of only 0.9598.
I also trained LightGBM and a soft-voting Ensemble of the three strongest models as
alternatives; the Ensemble (0.9611) didn't beat tuned XGBoost alone, which is a realistic
outcome for soft voting when one member is already dominant — not a bug, just how the data
behaved. XGBoost is also tree-based, so it works directly with SHAP's `TreeExplainer`, which
mattered a lot since explainability was a core product requirement, not an afterthought.

**Result:** Tuned XGBoost became the registered Production model in MLflow. It's what both the
API and dashboard load at runtime, and it's what the SHAP explanations and GPT summaries are
generated from.

---

## Q3: How did you handle class imbalance?

**Situation:** The dataset has a 26%/74% churn/no-churn split — not extreme, but enough that a
naive model could get decent accuracy by mostly predicting "no churn" and still miss most of
the customers who actually matter.

**Task:** Make sure the model actually learns to distinguish churners, and make sure the
downstream risk segmentation reflects real business risk tiers rather than a single blind
0.5 cutoff.

**Action:** I applied SMOTE (Synthetic Minority Oversampling) during training to balance the
26/74 split to roughly 50/50 in the training data, so the model saw enough churn examples to
learn meaningful patterns. On the scoring side, instead of a single 0.5 probability threshold,
I used two thresholds to create three risk segments — High (≥0.7), Medium (≥0.4), Low (<0.4)
— which better matches how a retention team actually wants to prioritize outreach (they don't
treat "51% risk" and "99% risk" the same way).

**Result:** The tuned XGBoost model reached 0.83 recall and 0.79 precision on the held-out test
set, and
the 3-tier segmentation is what actually drives the retention-targeting dashboard page —
customers get bucketed into priority tiers by risk segment *and* revenue value, not just a
raw probability number.

---

## Q4: How would you deploy this in production?

**Situation:** The project already has a Dockerized architecture and a working CI/CD-adjacent
setup, but going from "runs on my machine" to "production" means thinking about environments
where the full ML stack isn't available.

**Task:** Support both a full-featured internal deployment (with database access and the
complete ML toolchain) and a lightweight public-facing demo, without maintaining two separate
codebases.

**Action:** The `docker-compose.yml` defines three services — Postgres, FastAPI, and Streamlit
— so a full deployment is a single `docker-compose up --build`. For the API, `api/predictor.py`
loads whichever model is currently staged `Production` in the MLflow Model Registry, so
promoting a new model doesn't require redeploying the API. For the dashboard, I added a
`demo_mode` module: every page tries a real database connection first, and transparently falls
back to static, precomputed demo CSVs if it can't connect — with heavy ML imports (xgboost,
shap, matplotlib) made lazy/conditional so the whole app can run on a minimal
`requirements_streamlit.txt` with no compiled ML dependencies at all. That's what makes a
lightweight Streamlit Community Cloud deployment possible from the same repo.

**Result:** The same codebase supports a full local/internal deployment via Docker Compose and
a public read-only demo via Streamlit Cloud, with a GitHub Actions workflow running the
DB/model-independent tests on every push to `main`.

---

## Q5: What would you improve if you had more time?

**Situation:** Working solo against a fixed day-by-day plan meant some tradeoffs got made
deliberately and documented rather than fixed on the spot.

**Task:** Be honest about the real, known gaps rather than claiming the project is flawless.

**Action / things I'd prioritize:**
- **Fix `model_registry` accumulating duplicate rows** across retraining runs instead of
  upserting — I worked around it in the dashboard (dedupe by best AUC per model name) but the
  underlying table design should be fixed so a stale/worse model can't silently become
  "active" again after a rerun.
- **Expand CI coverage.** GitHub Actions currently only runs the tests that don't need a
  trained model or a populated database (a deliberate, documented scoping decision to keep CI
  green rather than fake success) — the next step is adding a seeding stage so more of the
  suite can run in a fresh checkout.
- **Add demo-mode support to the Retention Targeting dashboard page** — it's the one page that
  still reads a local, gitignored CSV with no cloud-safe fallback.
- **Automated retraining pipeline** (e.g., Airflow) so the model refreshes on a schedule
  instead of manual reruns, plus real-time streaming scoring via Kafka for production-scale
  traffic.
- **A/B testing framework** to actually measure whether the retention actions the model
  recommends improve real retention, not just churn-probability accuracy.

**Result:** These are captured in the README's Future Work section rather than left as
undocumented technical debt — the goal was to ship something real and be transparent about
what's next, not to over-engineer a solo side project.
