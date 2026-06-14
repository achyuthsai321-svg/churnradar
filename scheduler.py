"""
src/alerts/scheduler.py
APScheduler job — runs nightly at 02:00, scores all customers,
saves to DB, fires alerts for high-risk accounts.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd
import joblib
import shap
from datetime import datetime
from src.alerts.notifier import fire_alert

MODEL_PATH    = "models/xgboost_churn.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"
DATA_PATH     = "data/processed/telco_clean.csv"

FEATURE_LABELS = {
    "tenure":           "Months as customer",
    "MonthlyCharges":   "Monthly charge amount",
    "Contract":         "Contract type",
    "ServiceCount":     "Number of active services",
    "NewCustomer":      "New customer (≤3 months)",
    "AvgMonthlyCharge": "Avg monthly charge vs tenure",
}


def _risk(prob: float) -> str:
    return "HIGH" if prob >= 0.65 else "MEDIUM" if prob >= 0.40 else "LOW"


def nightly_score():
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M}] Starting nightly churn scoring...")

    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    explainer = shap.TreeExplainer(model)

    df = pd.read_csv(DATA_PATH)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    X       = df[features]
    X_sc    = scaler.transform(X)
    probs   = model.predict_proba(X_sc)[:, 1]
    shap_v  = explainer.shap_values(X_sc)

    alerts_fired = 0

    for i, (prob, shap_row) in enumerate(zip(probs, shap_v)):
        customer_id = str(df.index[i])  # use row index as ID if no ID col
        risk        = _risk(prob)

        # Build top-3 reasons
        pairs   = sorted(zip(features, shap_row), key=lambda x: abs(x[1]), reverse=True)[:3]
        reasons = [
            {
                "feature":    f,
                "label":      FEATURE_LABELS.get(f, f),
                "shap_value": round(float(v), 4),
                "direction":  "increases churn risk" if v > 0 else "reduces churn risk",
            }
            for f, v in pairs
        ]

        # Fire alert if HIGH risk
        if risk == "HIGH":
            fire_alert(customer_id, prob, reasons)
            alerts_fired += 1

    print(f"[DONE] Scored {len(df)} customers. Alerts fired: {alerts_fired}")


def start():
    scheduler = BlockingScheduler()
    # Run every night at 02:00
    scheduler.add_job(
        nightly_score,
        trigger=CronTrigger(hour=2, minute=0),
        id="nightly_churn_score",
        name="Nightly Churn Scoring",
        replace_existing=True,
    )
    print("[Scheduler] ChurnRadar nightly scoring scheduled at 02:00 daily.")
    print("[Scheduler] Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("[Scheduler] Stopped.")


if __name__ == "__main__":
    # Run once immediately for testing, then schedule
    nightly_score()
    start()
