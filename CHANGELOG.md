# Changelog

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
