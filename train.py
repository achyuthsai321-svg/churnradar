"""
src/ml/train.py
Trains Logistic Regression, Random Forest, and XGBoost.
Saves the best model. Prints comparison table.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
import joblib
import os

from preprocess import load_raw, clean, engineer_features, encode, scale, apply_smote

PROCESSED_PATH = "data/processed/telco_clean.csv"
MODEL_OUT = "models/xgboost_churn.pkl"


def load_processed():
    df = pd.read_csv(PROCESSED_PATH)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost":             XGBClassifier(
                                   n_estimators=300,
                                   max_depth=5,
                                   learning_rate=0.05,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   use_label_encoder=False,
                                   eval_metric="logloss",
                                   random_state=42
                               ),
    }


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"AUC-ROC : {auc:.4f}")
    print(f"Precision (churn): {report['1']['precision']:.4f}")
    print(f"Recall    (churn): {report['1']['recall']:.4f}")
    print(f"F1-score  (churn): {report['1']['f1-score']:.4f}")
    print(classification_report(y_test, y_pred))
    return auc


def train():
    # --- Build processed data if not exists ---
    if not os.path.exists(PROCESSED_PATH):
        print("Processed data not found. Running preprocessing...")
        from preprocess import load_raw, clean, engineer_features
        df = load_raw()
        df = clean(df)
        df = engineer_features(df)
        df.to_csv(PROCESSED_PATH, index=False)

    X, y = load_processed()
    X_train, X_test, y_train, y_test = split(X, y)

    # Scale features
    X_train_sc, X_test_sc, scaler = scale(X_train, X_test)

    # Handle class imbalance
    X_res, y_res = apply_smote(X_train_sc, y_train)

    models = build_models()
    best_auc = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_res, y_res)
        auc = evaluate(model, X_test_sc, y_test, name)
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    print(f"\n Best model: {best_name} (AUC={best_auc:.4f})")
    joblib.dump(best_model, MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")

    # Save feature names for SHAP
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    return best_model


if __name__ == "__main__":
    train()
