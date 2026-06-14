# ChurnRadar 🎯
**SaaS Customer Churn Prediction & Retention Intelligence Platform**

> Predict which customers will cancel **30–60 days before they do** — with explainable AI, a live retention dashboard, and automated CS team alerts.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC--ROC%200.89-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 The Problem

SaaS companies lose customers **silently**. By the time a cancellation shows up in reports, it's too late to act. Recruiters spend weeks on unqualified leads while customer success teams have no early warning system.

> **ChurnRadar changes that.** It flags at-risk customers with a probability score and explains *why* — so your CS team can intervene before the customer leaves.

### Business Impact
| Metric | Value |
|---|---|
| Cost of one churned customer | $15,000+ avg |
| Cost to retain vs acquire | 5× cheaper |
| Churn that is preventable | 67% with early action |
| Recruiter time saved (per hire) | 23 hrs |

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **ML Churn Prediction** | XGBoost model trained on 7,000+ customers, AUC-ROC = 0.89 |
| 🔍 **SHAP Explainability** | Top 3 plain-English reasons per prediction — no black box |
| 🚦 **Risk Segmentation** | Auto-buckets into HIGH / MEDIUM / LOW, updated daily |
| ⚡ **REST API** | FastAPI `/predict` and `/predict/batch` endpoints with Swagger docs |
| 📊 **Live Dashboard** | Streamlit app with churn heatmap, risk pie chart, at-risk table |
| 🔔 **Smart Alerts** | Email + Slack notification fired when customer crosses risk threshold |
| 🕑 **Nightly Scheduler** | APScheduler scores all customers at 02:00 automatically |
| ✅ **Test Suite** | Pytest with coverage for API, schema validation, and ML logic |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      ChurnRadar                         │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  Kaggle  │───▶│Preprocess│───▶│  XGBoost Model   │  │
│  │  Dataset │    │+ SMOTE   │    │  + SHAP Explain  │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
│                                           │             │
│              ┌────────────────────────────▼──────────┐  │
│              │         FastAPI REST API               │  │
│              │  POST /predict  POST /predict/batch    │  │
│              │  GET /customers/at-risk  GET /summary  │  │
│              └──────┬──────────────────┬─────────────┘  │
│                     │                  │                 │
│            ┌────────▼──────┐  ┌────────▼──────────┐    │
│            │   Streamlit   │  │   Alert System    │    │
│            │   Dashboard   │  │  Email + Slack    │    │
│            │ KPIs, Charts  │  │  APScheduler      │    │
│            │ At-risk Table │  │  Nightly 02:00    │    │
│            └───────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Pipeline

### 1. Data — Telco Customer Churn (Kaggle)
- 7,043 customers · 21 raw features · binary churn label (Yes/No)
- Download: [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### 2. Feature Engineering
5 new features created on top of the 21 raw ones:

| Feature | Logic | Why it helps |
|---|---|---|
| `AvgMonthlyCharge` | TotalCharges / (tenure + 1) | Detects overcharged new users |
| `NewCustomer` | tenure ≤ 3 → 1 else 0 | First 3 months = highest churn risk |
| `HighValue` | MonthlyCharges ≥ 75th percentile | High-value customers churn differently |
| `ServiceCount` | Count of active add-on services | More services = more sticky |

### 3. Class Imbalance — SMOTE
Only 26% of customers churned → imbalanced dataset → applied **SMOTE** on training set to oversample minority class before fitting.

### 4. Model Comparison

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.83 | 0.71 | 0.62 | 0.66 |
| Random Forest | 0.86 | 0.76 | 0.68 | 0.72 |
| **XGBoost ✅** | **0.89** | **0.81** | **0.74** | **0.77** |

**XGBoost** selected as final model — best AUC-ROC and F1, handles missing values natively, fastest inference.

### 5. SHAP Explainability
Every prediction includes top-3 churn drivers in plain English:
```json
{
  "customer_id": "CUST-4892",
  "churn_probability": 0.82,
  "risk_level": "HIGH",
  "top_reasons": [
    { "label": "Contract type",         "direction": "increases churn risk" },
    { "label": "Months as customer",    "direction": "increases churn risk" },
    { "label": "Has tech support add-on","direction": "reduces churn risk"  }
  ],
  "recommendation": "Assign to CS team immediately. Offer loyalty discount or contract upgrade."
}
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL (optional — SQLite used by default)
- Git

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/achyuthsai321-svg/churnradar.git
cd churnradar

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment config
cp .env.example .env
# Edit .env with your SMTP and Slack credentials

# 4. Download dataset from Kaggle → place at:
#    data/raw/telco_churn.csv

# 5. Train the model
python src/ml/train.py

# 6. Start the API
uvicorn src.api.main:app --reload

# 7. Launch the dashboard (new terminal)
streamlit run src/dashboard/app.py

# 8. (Optional) Start nightly scheduler
python src/alerts/scheduler.py
```

---

## 📡 API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### `GET /health`
```json
{ "status": "ok", "model_ready": true, "version": "1.0.0" }
```

### `POST /api/v1/predict`
Predict churn risk for a single customer.

**Request body:**
```json
{
  "customer_id": "CUST-001",
  "tenure": 2,
  "monthly_charges": 75.5,
  "total_charges": 150.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "tech_support": "No"
}
```

**Response:**
```json
{
  "customer_id": "CUST-001",
  "churn_probability": 0.82,
  "risk_level": "HIGH",
  "top_reasons": [...],
  "recommendation": "Assign to CS team immediately."
}
```

### `POST /api/v1/predict/batch`
Score multiple customers in one request. Returns full breakdown with MRR at risk.

### `GET /api/v1/customers/at-risk?risk_level=HIGH`
Returns paginated list of high-risk customers sorted by churn probability.

### `GET /api/v1/customers/summary`
Dashboard summary — total scored, risk breakdown, average churn probability.

---

## 📁 Project Structure

```
churnradar/
├── data/
│   ├── raw/                    # Original Kaggle CSV (not committed)
│   └── processed/              # Cleaned, feature-engineered CSV
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── ml/
│   │   ├── preprocess.py       # Cleaning, feature engineering, SMOTE
│   │   ├── train.py            # Model training + comparison
│   │   └── explain.py          # SHAP explainability
│   ├── api/
│   │   ├── main.py             # FastAPI entry point
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── database.py         # SQLAlchemy ORM
│   │   └── routes/
│   │       ├── predict.py      # /predict endpoints
│   │       ├── customers.py    # /customers endpoints
│   │       └── health.py       # /health endpoint
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard
│   └── alerts/
│       ├── notifier.py         # Email + Slack alerts
│       └── scheduler.py        # APScheduler nightly job
├── models/                     # Saved .pkl artifacts (not committed)
├── tests/
│   └── test_api.py             # Pytest test suite
├── docs/
│   └── GITHUB_SETUP.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🔔 Alert System Setup

To enable email + Slack alerts, fill in `.env`:

```env
# Email (Gmail App Password recommended)
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_16_char_app_password
ALERT_RECIPIENT=cs_team@company.com

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ

# Risk threshold (0.0–1.0)
CHURN_THRESHOLD=0.65
```

---

## 🏢 Who Would Use This?

| Industry | Use Case |
|---|---|
| **Fintech / BFSI** | Retain loan, insurance, and banking app users |
| **EdTech** | Flag students about to drop off a course subscription |
| **SaaS** | Reduce MRR churn in B2B/B2C subscription products |
| **Telecom** | Classic use case — Telco dataset proves direct applicability |

---

## 📄 License
MIT License — free to use, modify, and distribute.

---

## 👤 Author
**Achyuth Sai** · B.Tech CSE (Data Science) · SRM Institute of Science and Technology, Tiruchirapalli  
GitHub: [@achyuthsai321-svg](https://github.com/achyuthsai321-svg)
