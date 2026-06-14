"""
src/ml/preprocess.py
Loads raw Telco CSV, cleans, engineers features, handles class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_PATH = "data/processed/telco_clean.csv"


def load_raw() -> pd.DataFrame:
    """Load raw Kaggle Telco CSV."""
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, drop nulls, encode target."""
    # TotalCharges has blank strings — convert to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Binary target: Yes=1, No=0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Drop customer ID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new predictive features from existing columns."""

    # Charges per month relative to tenure
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

    # High-value flag: top 25% by monthly charges
    threshold = df["MonthlyCharges"].quantile(0.75)
    df["HighValue"] = (df["MonthlyCharges"] >= threshold).astype(int)

    # Short-tenure flag: customers in first 3 months churn most
    df["NewCustomer"] = (df["tenure"] <= 3).astype(int)

    # Service count — more services = more sticky
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["ServiceCount"] = df[service_cols].apply(
        lambda row: sum(v not in ["No", "No internet service", "No phone service"]
                        for v in row), axis=1
    )

    return df


def encode(df: pd.DataFrame):
    """Label-encode categorical columns. Return X, y, and encoder map."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    return X, y, encoders


def scale(X_train, X_test):
    """StandardScaler fit on train, applied to both splits."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")
    return X_train_sc, X_test_sc, scaler


def apply_smote(X_train, y_train):
    """Oversample minority class (churned) using SMOTE."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {sum(y_res==1)} churned, {sum(y_res==0)} retained")
    return X_res, y_res


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data → {PROCESSED_PATH}")
