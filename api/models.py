"""Pydantic request/response models for the churn prediction API."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CustomerInput(BaseModel):
    """Raw customer attributes needed to score churn risk."""

    customer_id: Optional[str] = Field(default=None, description="Known customer ID, if any.")
    gender: str = Field(..., examples=["Female"])
    senior_citizen: int = Field(..., ge=0, le=1, examples=[0])
    partner: str = Field(..., examples=["Yes"])
    dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=100, examples=[12])
    phone_service: str = Field(..., examples=["Yes"])
    multiple_lines: str = Field(..., examples=["No"])
    internet_service: str = Field(..., examples=["Fiber optic"])
    online_security: str = Field(..., examples=["No"])
    online_backup: str = Field(..., examples=["No"])
    device_protection: str = Field(..., examples=["No"])
    tech_support: str = Field(..., examples=["No"])
    streaming_tv: str = Field(..., examples=["Yes"])
    streaming_movies: str = Field(..., examples=["Yes"])
    contract: str = Field(..., examples=["Month-to-month"])
    paperless_billing: str = Field(..., examples=["Yes"])
    payment_method: str = Field(..., examples=["Electronic check"])
    monthly_charges: float = Field(..., gt=0, examples=[85.5])
    total_charges: float = Field(..., ge=0, examples=[1026.0])

    @model_validator(mode="after")
    def check_total_charges_at_least_monthly_charges(self) -> "CustomerInput":
        # A small tolerance (matches tests/test_eda.py) accounts for legitimate noise in
        # tenure=1 customers, where total_charges can dip a few dollars below monthly_charges.
        tolerance = 10.0
        if self.total_charges < self.monthly_charges - tolerance:
            raise ValueError(
                f"total_charges ({self.total_charges}) cannot be less than monthly_charges "
                f"({self.monthly_charges}) by more than ${tolerance}."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-DEMO-001",
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 85.5,
                "total_charges": 1026.0,
            }
        }
    }


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionOutput(BaseModel):
    """Result of scoring a single customer for churn risk."""

    customer_id: str
    churn_probability: float
    risk_segment: str
    top_risk_factors: list[RiskFactor]
    recommended_actions: list[str]
    model_version: str
    predicted_at: datetime


class BatchPredictionInput(BaseModel):
    customers: list[CustomerInput]


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
