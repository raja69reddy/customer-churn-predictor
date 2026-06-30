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

## Progress Log

### Day 1 — Project Scaffold (2026-06-28)
- Created complete folder structure
- Set up requirements.txt with all ML and AI packages
- Created .env.example with all required variables
- Wrote sql/schema.sql with 4 table definitions
- Created src/utils/db.py connection helper
- Initialized Git and pushed to GitHub

## Future Work
- Add real-time streaming predictions (Kafka / Redis)
- Scheduled retraining pipeline (Airflow / cron)
- Slack / email alerts for high-risk customers
- A/B testing framework for model versions
- React frontend to replace Streamlit for production use
