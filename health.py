"""
src/api/routes/health.py
GET /health — liveness check
"""
from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/health")
def health():
    model_ready = os.path.exists("models/xgboost_churn.pkl")
    return {
        "status":      "ok",
        "model_ready": model_ready,
        "version":     "1.0.0",
    }
