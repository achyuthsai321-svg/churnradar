"""
src/api/routes/predict.py
POST /api/v1/predict       — single customer churn prediction
POST /api/v1/predict/batch — batch prediction for multiple customers
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas import CustomerInput, PredictionResponse, BatchInput, BatchResponse, ChurnReason
import joblib
import pandas as pd
import numpy as np

router = APIRouter()

# Load model artifacts once at startup
try:
    MODEL   = joblib.load("models/xgboost_churn.pkl")
    SCALER  = joblib.load("models/scaler.pkl")
    FEATURES = joblib.load("models/feature_names.pkl")
except Exception as e:
    MODEL = SCALER = FEATURES = None
    print(f"[WARN] Model not loaded: {e}. Run train.py first.")


RECOMMENDATIONS = {
    "HIGH":   "Assign to CS team immediately. Offer loyalty discount or contract upgrade.",
    "MEDIUM": "Schedule a check-in call. Share feature highlights they haven't used yet.",
    "LOW":    "No immediate action needed. Monitor next 30 days.",
}


def _risk_level(prob: float) -> str:
    if prob >= 0.65:  return "HIGH"
    if prob >= 0.40:  return "MEDIUM"
    return "LOW"


def _prepare_features(customer: CustomerInput) -> pd.DataFrame:
    """Convert CustomerInput to a feature DataFrame matching training columns."""
    raw = {
        "tenure":           customer.tenure,
        "MonthlyCharges":   customer.monthly_charges,
        "TotalCharges":     customer.total_charges,
        "Contract":         customer.contract,
        "InternetService":  customer.internet_service,
        "OnlineSecurity":   customer.online_security,
        "TechSupport":      customer.tech_support,
        "PaymentMethod":    customer.payment_method,
        "PaperlessBilling": customer.paperless_billing,
        "PhoneService":     customer.phone_service,
        "MultipleLines":    customer.multiple_lines,
        "OnlineBackup":     customer.online_backup,
        "DeviceProtection": customer.device_protection,
        "StreamingTV":      customer.streaming_tv,
        "StreamingMovies":  customer.streaming_movies,
        "Dependents":       customer.dependents,
        "Partner":          customer.partner,
        "SeniorCitizen":    customer.senior_citizen,
        "gender":           customer.gender,
    }
    df = pd.DataFrame([raw])

    # Encode categoricals the same way as training
    from sklearn.preprocessing import LabelEncoder
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Engineer same features as preprocess.py
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighValue"]        = (df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(0.75)).astype(int)
    df["NewCustomer"]      = (df["tenure"] <= 3).astype(int)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["ServiceCount"] = df[service_cols].apply(
        lambda row: sum(v not in [0] for v in row), axis=1
    )

    # Align to training feature order
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURES]


@router.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    try:
        import shap
        df = _prepare_features(customer)
        X_scaled = SCALER.transform(df)

        prob       = float(MODEL.predict_proba(X_scaled)[0][1])
        risk       = _risk_level(prob)

        # SHAP reasons
        explainer  = shap.TreeExplainer(MODEL)
        shap_vals  = explainer.shap_values(X_scaled)[0]
        pairs      = sorted(zip(FEATURES, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        reasons    = [
            ChurnReason(feature=f, label=f.replace("_", " "), shap_value=round(float(v), 4),
                        direction="increases churn risk" if v > 0 else "reduces churn risk")
            for f, v in pairs
        ]

        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(prob, 4),
            risk_level=risk,
            top_reasons=reasons,
            recommendation=RECOMMENDATIONS[risk],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    predictions = [predict_single(c) for c in payload.customers]
    high   = sum(1 for p in predictions if p.risk_level == "HIGH")
    medium = sum(1 for p in predictions if p.risk_level == "MEDIUM")
    low    = sum(1 for p in predictions if p.risk_level == "LOW")

    return BatchResponse(
        total=len(predictions),
        high_risk=high,
        medium_risk=medium,
        low_risk=low,
        predictions=predictions,
    )
