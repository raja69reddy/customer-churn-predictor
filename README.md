# Customer Churn Predictor

## Project Overview
An end-to-end machine learning system that predicts customer churn using the Telco Customer Churn dataset. The project covers the full ML lifecycle: data ingestion, feature engineering, model training, evaluation, serving via a REST API, and an interactive dashboard — with GPT-powered natural-language explanations for each at-risk customer.

## Business Problem
Retaining existing customers is significantly cheaper than acquiring new ones. This project helps customer-success and marketing teams identify which customers are likely to churn in the next billing cycle so they can take proactive action (discounts, outreach, plan changes) before the customer leaves.

## Tech Stack
| Layer | Tools |
|---|---|
| Data | PostgreSQL, SQLAlchemy, pandas |
| ML | scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn |
| Experiment Tracking | MLflow |
| API | FastAPI, uvicorn |
| Dashboard | Streamlit, Plotly |
| AI Explanations | OpenAI GPT |
| Containerisation | Docker, docker-compose |

## Project Structure
```
customer-churn-predictor/
├── data/              # raw and processed CSV data (git-ignored)
├── notebooks/         # EDA, feature analysis, model analysis
├── src/
│   ├── data/          # ingestion pipeline
│   ├── features/      # feature engineering
│   ├── models/        # train / evaluate / predict
│   └── utils/         # shared helpers (DB connection)
├── ai/                # GPT-powered churn explainer
├── dashboard/         # Streamlit app
├── api/               # FastAPI prediction service
├── sql/               # schema + views
├── tests/
├── models/            # saved .pkl model artefacts (git-ignored)
└── mlflow/            # MLflow tracking (git-ignored)
```

## Model Performance
_To be updated after training (Day 3)._

| Model | Accuracy | AUC | F1 |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| LightGBM | — | — | — |

## Setup Instructions
```bash
# 1. Clone the repo
git clone https://github.com/raja69reddy/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your PostgreSQL credentials and OpenAI key

# 5. Create the database and tables
psql -U postgres -c "CREATE DATABASE churn_db;"
psql -U postgres -d churn_db -f sql/schema.sql
```

## How to Run
```bash
# Run the data ingestion pipeline
python -m src.data.ingestion

# Train models
python -m src.models.train

# Start the FastAPI server
uvicorn api.main:app --reload

# Launch the Streamlit dashboard
streamlit run dashboard/app.py
```

## Dashboard
The Streamlit dashboard (dashboard/app.py) provides:
- Overall churn rate KPIs
- Risk segment breakdown
- Feature importance charts (SHAP)
- Per-customer churn probability and GPT explanation

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict churn for a single customer |
| POST | `/predict/batch` | Batch predictions |
| GET | `/model/info` | Active model metadata |

## 📋 Progress Log

✅ **Day 1 — Project Scaffold**
- Created complete folder structure
- Set up requirements.txt with all ML and AI packages
- Created .env.example with all required variables
- Wrote sql/schema.sql with 4 table definitions
- Created src/utils/db.py connection helper
- Initialized Git and pushed to GitHub

✅ **Day 2 — Telco Data Ingestion**
- Generated 7,043 synthetic customer records
- Realistic churn rate of 26% matching Telco dataset
- Created src/data/ingestion.py with full and incremental modes
- Added error handling and Python logging
- Created src/data/verify.py for data quality verification
- Loaded all 7,043 rows into raw_customers PostgreSQL table
- Created vw_churn_overview.sql and vw_high_risk.sql views
- All unit tests passing with pytest

✅ **Day 3 — Exploratory Data Analysis**
- Created notebooks/01_eda.ipynb with 7 analysis sections
- Analyzed churn distribution — 26% churn rate confirmed
- Analyzed all numeric features: tenure, monthly charges, total charges
- Analyzed all categorical features with churn breakdown
- Created correlation heatmap
- Key insights: month-to-month contracts 43% churn
- Created src/data/eda_summary.py automated summary script
- Updated vw_churn_overview.sql with tenure and charge tier breakdowns
- Created vw_churn_by_features.sql view
- All unit tests passing with pytest

✅ **Day 4 — Feature Engineering**
- Created src/features/engineering.py with FeatureEngineer class
- Engineered 3 new features: tenure_group, charge_per_month, services_count
- Added contract_risk_score and payment_risk_score features
- Converted churn Yes/No to binary label (0/1)
- Applied SMOTE oversampling — balanced 26% churn to 50/50
- Built sklearn preprocessing pipeline with imputer + scaler + encoder
- Saved preprocessor to models/preprocessor.pkl
- Created notebooks/02_feature_analysis.ipynb
- Updated vw_high_risk.sql with engineered features
- Created vw_feature_importance.sql view
- All unit tests passing with pytest

✅ **Day 5 — Baseline ML Models**
- Trained 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Created ModelTrainer class with save/load functionality
- Created ModelEvaluator with ROC curves and confusion matrices
- Generated model comparison table (accuracy, AUC, F1, precision, recall)
- Plotted ROC curves for all 4 models on single chart
- Plotted confusion matrices in 2x2 grid
- Logged all results to model_registry PostgreSQL table
- Best model identified and set as active
- Created vw_model_performance.sql view
- All unit tests passing with pytest

✅ **Day 6 — Model Evaluation & Hyperparameter Tuning**
- Added 5-fold stratified cross validation for all 4 models
- GridSearchCV tuning for Random Forest — best params found
- GridSearchCV tuning for XGBoost — best params found
- Compared tuned vs baseline models — AUC improved
- SHAP values analysis — top 15 churn drivers identified
- Precision-recall curve analysis
- Found optimal classification threshold
- Created notebooks/03_model_analysis.ipynb
- Built ChurnPredictor class with High/Medium/Low risk segments
- Tested ChurnPredictor with 5 sample customers
- All unit tests passing with pytest

✅ **Day 7 — Advanced ML + LightGBM + Ensemble**
- Trained LightGBM model — compared with XGBoost
- Built ensemble model using VotingClassifier with soft voting
- Generated final model leaderboard with all 6 models
- Selected best model by AUC score and registered in DB
- Added GPT-powered churn explanation to ChurnPredictor
- Tested GPT explanations with 3 high risk customer profiles
- Created batch_scorer.py — scored all 7,043 customers
- Saved all predictions to churn_predictions table
- Created vw_churn_predictions.sql and vw_risk_segments.sql
- All unit tests passing with pytest

✅ **Day 8 — MLflow Experiment Tracking**
- Set up MLflow experiment tracking infrastructure
- Retrained all 6 models with full MLflow logging
- Logged params, metrics, and artifacts for each run
- Registered best model in MLflow Model Registry
- Created model versioning system — v1 vs v2 comparison
- Created src/models/experiment_analyzer.py
- Updated batch_scorer.py to load from MLflow registry
- Created mlflow/README.md documentation
- All unit tests passing with pytest

✅ **Day 9 — Batch Prediction Pipeline**
- Updated batch_scorer.py with --mode, --threshold, --output flags
- Batch scored all 7,043 customers successfully
- Created prediction_monitor.py for prediction quality checks
- Created incremental_scorer.py for efficient re-scoring
- Created prediction_analysis.sql with drift detection
- Created vw_prediction_summary.sql view
- Created retention_targets.py with 3 priority tiers:
  Tier 1 (high risk + high value), Tier 2, Tier 3
- Updated churn_predictions table with new columns
- All unit tests passing with pytest

✅ **Day 10 — Streamlit Dashboard Complete**
- Created dashboard/app.py with sidebar filters
- Created metrics.py and charts.py reusable components
- Created 5 dashboard pages:
  * Overview — churn rate, KPIs, distributions
  * Predictions — risk segments, probability histogram
  * Customer Lookup — individual prediction + AI explanation
  * Model Performance — ROC curves, SHAP, feature importance
  * Retention Targeting — priority tiers, recommended actions
- Added caching, loading spinners, error handling
- All 5 pages verified working on localhost:8501
- All unit tests passing with pytest

✅ **Day 11 — FastAPI Prediction API**
- Created api/main.py FastAPI app with CORS and middleware
- Created api/models.py with Pydantic input/output models
- Created api/predictor.py APIPredictor class
- Added 12 endpoints: predict, batch predict, customer lookup, analytics, risk summary, retention targets
- Added request validation and friendly error handling
- Created api/database.py for DB operations
- Tested all endpoints — all returning correct responses
- Created api/README.md with curl examples
- All unit tests passing with pytest

## Future Work
- Add real-time streaming predictions (Kafka / Redis)
- Scheduled retraining pipeline (Airflow / cron)
- Slack / email alerts for high-risk customers
- A/B testing framework for model versions
- React frontend to replace Streamlit for production use
