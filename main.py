"""
src/api/main.py
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import predict, customers, health

app = FastAPI(
    title="ChurnRadar API",
    description="SaaS Customer Churn Prediction & Retention Intelligence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(customers.router, prefix="/api/v1", tags=["Customers"])


@app.get("/")
def root():
    return {"message": "ChurnRadar API is running", "docs": "/docs"}
