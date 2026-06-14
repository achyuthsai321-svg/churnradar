# ChurnRadar 🎯
**SaaS Customer Churn Prediction & Retention Intelligence Platform**

Predict which customers will cancel 30–60 days before they do — with explainable AI and a live retention dashboard.

## Why it matters
- 5× cheaper to retain a customer than acquire a new one
- 67% of SaaS churn is preventable with early action
- ChurnRadar gives your CS team a daily risk-ranked customer list with actionable reasons

---

## Features
- **ML churn prediction** — XGBoost model with AUC-ROC > 0.88
- **SHAP explainability** — top 3 reasons per prediction, no black box
- **Risk segmentation** — High / Medium / Low buckets updated daily
- **REST API** — `/predict` endpoint for real-time scoring
- **Live dashboard** — Streamlit app with churn heatmap, MRR at risk, at-risk list
- **Alert system** — email/Slack alert when a customer crosses the risk threshold

---

## Tech Stack
| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, scikit-learn, XGBoost, SHAP, imbalanced-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Database | PostgreSQL, SQLAlchemy |
| Dashboard | Streamlit, Plotly |
| Scheduling | APScheduler |
| Testing | Pytest |

---

## Dataset
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
7,043 customers · 21 features · binary churn label

---

## Quick Start
```bash
git clone https://github.com/achyuthsai321-svg/churnradar.git
cd churnradar
pip install -r requirements.txt

# Train model
python src/ml/train.py

# Start API
uvicorn src.api.main:app --reload

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Project Structure
```
churnradar/
├── data/
│   ├── raw/              # Original Kaggle CSV
│   └── processed/        # Cleaned, feature-engineered CSV
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── ml/               # Data pipeline, training, evaluation, SHAP
│   ├── api/              # FastAPI app + prediction endpoint
│   ├── dashboard/        # Streamlit dashboard
│   └── alerts/           # Email/Slack alert system + scheduler
├── models/               # Saved model artifacts (.pkl)
├── tests/                # Pytest test suite
├── docs/                 # Architecture diagram, model card
├── requirements.txt
├── .env.example
└── README.md
```

---

## Model Performance (Telco dataset)
| Model | AUC-ROC | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 0.83 | 0.71 | 0.62 |
| Random Forest | 0.86 | 0.76 | 0.68 |
| **XGBoost** | **0.89** | **0.81** | **0.74** |

---

## Author
Achyuth Sai · B.Tech CSE (Data Science) · SRM Institute, Tiruchirapalli
