"""
src/api/schemas.py
Pydantic models for request validation and response shaping.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class CustomerInput(BaseModel):
    """Single customer feature payload for /predict."""
    customer_id:        str
    tenure:             int   = Field(..., ge=0, description="Months with company")
    monthly_charges:    float = Field(..., ge=0)
    total_charges:      float = Field(..., ge=0)
    contract:           str   = Field(..., example="Month-to-month")
    internet_service:   str   = Field(..., example="Fiber optic")
    online_security:    str   = Field(..., example="No")
    tech_support:       str   = Field(..., example="No")
    payment_method:     str   = Field(..., example="Electronic check")
    paperless_billing:  str   = Field(..., example="Yes")
    phone_service:      str   = Field(..., example="Yes")
    multiple_lines:     str   = Field(..., example="No")
    online_backup:      str   = Field(..., example="No")
    device_protection:  str   = Field(..., example="No")
    streaming_tv:       str   = Field(..., example="No")
    streaming_movies:   str   = Field(..., example="No")
    dependents:         str   = Field(..., example="No")
    partner:            str   = Field(..., example="No")
    senior_citizen:     int   = Field(..., ge=0, le=1)
    gender:             str   = Field(..., example="Male")


class ChurnReason(BaseModel):
    feature:    str
    label:      str
    shap_value: float
    direction:  str


class PredictionResponse(BaseModel):
    customer_id:        str
    churn_probability:  float
    risk_level:         RiskLevel
    top_reasons:        List[ChurnReason]
    recommendation:     str


class BatchInput(BaseModel):
    customers: List[CustomerInput]


class BatchResponse(BaseModel):
    total:          int
    high_risk:      int
    medium_risk:    int
    low_risk:       int
    predictions:    List[PredictionResponse]
    mrr_at_risk:    Optional[float] = None
