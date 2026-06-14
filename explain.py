"""
src/ml/explain.py
SHAP-based explainability for ChurnRadar predictions.
Returns top-3 churn reasons per customer in plain English.
"""

import shap
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/xgboost_churn.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

# Human-readable feature labels shown in alerts and dashboard
FEATURE_LABELS = {
    "tenure":              "Months as customer",
    "MonthlyCharges":      "Monthly charge amount",
    "TotalCharges":        "Total spend to date",
    "Contract":            "Contract type",
    "InternetService":     "Internet service type",
    "OnlineSecurity":      "Has online security add-on",
    "TechSupport":         "Has tech support add-on",
    "ServiceCount":        "Number of active services",
    "NewCustomer":         "New customer (≤3 months)",
    "HighValue":           "High-value customer tier",
    "AvgMonthlyCharge":    "Avg monthly charge vs tenure",
    "PaymentMethod":       "Payment method",
    "PaperlessBilling":    "Paperless billing",
}


def load_explainer():
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return explainer


def get_shap_values(X: pd.DataFrame):
    """Compute SHAP values for a batch of customers."""
    explainer = load_explainer()
    shap_values = explainer.shap_values(X)
    return shap_values


def top_reasons(shap_row: np.ndarray, feature_names: list, n: int = 3) -> list[dict]:
    """
    Given one customer's SHAP values, return top-n churn drivers.
    Returns list of dicts: {feature, label, shap_value, direction}
    """
    pairs = list(zip(feature_names, shap_row))
    # Sort by absolute SHAP value descending
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:n]

    reasons = []
    for feat, val in pairs_sorted:
        reasons.append({
            "feature":   feat,
            "label":     FEATURE_LABELS.get(feat, feat),
            "shap_value": round(float(val), 4),
            "direction": "increases churn risk" if val > 0 else "reduces churn risk",
        })
    return reasons


def explain_customer(customer_df: pd.DataFrame) -> dict:
    """
    Full explanation for a single customer row.
    Returns churn probability + top 3 reasons.
    """
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    scaler = joblib.load("models/scaler.pkl")

    X_scaled = scaler.transform(customer_df[feature_names])
    prob = float(model.predict_proba(X_scaled)[0][1])

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled)[0]

    reasons = top_reasons(shap_vals, feature_names)

    return {
        "churn_probability": round(prob, 4),
        "risk_level": _risk_level(prob),
        "top_reasons": reasons,
    }


def _risk_level(prob: float) -> str:
    if prob >= 0.65:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


if __name__ == "__main__":
    # Quick sanity check
    import pandas as pd
    sample = pd.read_csv("data/processed/telco_clean.csv").drop(columns=["Churn"]).head(1)
    result = explain_customer(sample)
    print(result)
