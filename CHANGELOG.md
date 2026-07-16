# Changelog

## Day 17 - Portfolio Polish
- Added tech stack badges to README
- Added project highlights section
- Added Future Work section
- Added MIT LICENSE file

## Day 16 - Dashboard Polish
- Polished all 5 dashboard pages
- Added trend charts and gauge charts
- Improved color coding and badges
- Added download buttons to all pages
- Added campaign ROI estimator

## Day 15 - Streamlit Cloud Deployment
- Created requirements_streamlit.txt
- Updated .streamlit/config.toml theme
- Created dashboard/demo_mode.py
- Updated app.py with demo mode fallback
- Generated demo data files for cloud

## Day 14 - Week 2 Review + README Polish
- Full pipeline end-to-end test successful
- All ingestion tests passing
- Added schema and data flow diagrams to README
- Added Key Business Insights section
- Added How to Use section
- Week 2 complete!

## Day 13 - Code Quality + Documentation
- Ran black formatter on entire codebase
- Fixed all flake8 warnings
- Added type hints to all key functions
- Updated README with model performance table
- Created CONTRIBUTING.md

## Day 12 - Docker + Deployment
- Updated Dockerfile for production
- Updated docker-compose.yml with 3 services
- Created .dockerignore file
- Added Streamlit deployment config
- Set up GitHub Actions CI pipeline

## Day 11 — FastAPI Prediction API
- Created api/main.py FastAPI app with CORS and middleware
- Created api/models.py with Pydantic input/output models
- Created api/predictor.py APIPredictor class
- Added 12 endpoints: predict, batch predict, customer lookup, analytics, risk summary, retention targets
- Added request validation and friendly error handling
- Created api/database.py for DB operations
- Tested all endpoints — all returning correct responses
- Created api/README.md with curl examples
- All unit tests passing with pytest

## Day 10 — Streamlit Dashboard
- Created dashboard/app.py main entry point
- Created metrics.py and charts.py components
- Created 5 dashboard pages:
  1_overview, 2_predictions, 3_customer_lookup,
  4_model_performance, 5_retention
- Added caching, spinners, error handling
- All pages verified working
- All tests passing

## Day 9 — Batch Prediction Pipeline
- Updated batch_scorer.py with production flags
- Created prediction_monitor.py for quality checks
- Created incremental_scorer.py for stale predictions
- Created prediction_analysis.sql queries
- Created vw_prediction_summary.sql view
- Created retention_targets.py with 3 priority tiers
- Updated churn_predictions table schema
- All tests passing

## Day 8 — MLflow Experiment Tracking
- Set up MLflow experiment tracking
- Logged all 6 model runs with params and metrics
- Registered best model in MLflow Model Registry
- Created model versioning system (v1 vs v2)
- Created experiment_analyzer.py
- Updated batch_scorer to use MLflow registry
- All tests passing

## Day 7 — Advanced ML + LightGBM + Ensemble
- Trained LightGBM model
- Built ensemble model with soft voting
- Generated final model leaderboard (6 models)
- Selected and registered best model
- Added GPT explanation to ChurnPredictor
- Created batch_scorer.py — scored all 7,043 customers
- Created vw_churn_predictions.sql and vw_risk_segments.sql
- All tests passing

## Day 6 — Model Evaluation & Hyperparameter Tuning
- Added 5-fold cross validation for all models
- GridSearchCV tuning for Random Forest and XGBoost
- SHAP values analysis — top 15 features identified
- Precision-recall curve analysis
- Found optimal classification threshold
- Created notebooks/03_model_analysis.ipynb
- Built ChurnPredictor class with risk segmentation
- All tests passing

## Day 5 — Baseline ML Models
- Trained 4 baseline models: LR, Decision Tree, Random Forest, XGBoost
- Created ModelTrainer and ModelEvaluator classes
- Generated model comparison table
- Plotted ROC curves and confusion matrices
- Logged all results to model_registry table
- Best model identified by AUC score
- All tests passing

## Day 4 — Feature Engineering
- Created FeatureEngineer class with full pipeline
- Engineered features: tenure_group, charge_per_month, services_count
- Added risk scores: contract_risk_score, payment_risk_score
- Added SMOTE oversampling for class imbalance
- Built sklearn preprocessing pipeline
- Saved preprocessor to models/preprocessor.pkl
- Created notebooks/02_feature_analysis.ipynb
- All tests passing

## Day 3 — Exploratory Data Analysis
- Created notebooks/01_eda.ipynb with 7 sections
- Analyzed churn distribution (26% churn rate)
- Analyzed numeric and categorical features
- Created correlation heatmap
- Key finding: month-to-month contracts have 43% churn
- Created src/data/eda_summary.py
- Updated SQL views with tenure and charge tiers
- All tests passing

## Day 2 — Telco Data Ingestion
- Generated 7,043 synthetic customer records
- 26% churn rate matching real Telco dataset distribution
- Built src/data/ingestion.py with full and incremental modes
- Created src/data/verify.py for data quality checks
- Created vw_churn_overview.sql and vw_high_risk.sql views
- All tests passing

## Day 1 — Project Scaffold
- Created complete folder structure
- Set up requirements.txt with all ML and AI packages
- Created .env.example with all required variables
- Wrote sql/schema.sql with 4 table definitions
- Created src/utils/db.py connection helper
- Initialized Git and pushed to GitHub
