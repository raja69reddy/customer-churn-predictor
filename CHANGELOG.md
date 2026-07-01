# Changelog

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
