"""Pydantic Request and Response Schemas for FastAPI Model Serving."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CustomerFeatures(BaseModel):
    """Input features for customer churn inference.

    Strictly enforces feature validation. Disallows extra/unexpected features,
    and explicitly forbids the target column 'Churn'.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 65,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 53.5,
                "TotalCharges": 3517.9,
            }
        },
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120, description="Months of customer tenure")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0, le=500.0, description="Current monthly charge amount in USD")
    TotalCharges: Optional[float] = Field(default=None, description="Total historical charges accrued in USD")

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def validate_total_charges(cls, v: Any) -> Optional[float]:
        """Safely coerces NaN or empty strings to None and enforces non-negativity."""
        if v is None:
            return None
        if isinstance(v, str):
            v_stripped = v.strip()
            if not v_stripped:
                return None
            try:
                v = float(v_stripped)
            except ValueError:
                raise ValueError("TotalCharges must be a valid numeric amount")
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("TotalCharges cannot be negative")
            return float(v)
        return v

    def to_inference_dict(self) -> Dict[str, Any]:
        """Dumps dictionary with exact keys required by preprocessing pipeline."""
        return self.model_dump()


class PredictionRequest(BaseModel):
    """Optional wrapper for single prediction request with customer metadata."""

    model_config = ConfigDict(extra="forbid")
    customer_id: Optional[str] = Field(default=None, description="Optional customer identifier for tracking")
    features: CustomerFeatures


class PredictionResponse(BaseModel):
    """Structured response returned by /predict endpoint."""

    request_id: str
    prediction: str = Field(description="'Churn' or 'No Churn'")
    churn_probability: float = Field(ge=0.0, le=1.0, description="Predicted probability of churn (class 1)")
    decision_threshold: float = Field(ge=0.0, le=1.0, description="Calibrated decision threshold applied")
    model_name: str
    model_version: str
    inference_latency_ms: float = Field(ge=0.0, description="Model execution latency in milliseconds")
    prediction_timestamp: str = Field(description="ISO 8601 UTC timestamp of prediction")


class ModelInfoResponse(BaseModel):
    """Metadata response returned by /model/info endpoint."""

    model_name: str
    resolved_version: str
    run_id: str
    alias: str
    load_timestamp: str
    decision_threshold: float
    status: str = "active"


class HealthResponse(BaseModel):
    """Response returned by /health endpoint."""

    status: str = "healthy"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReadyResponse(BaseModel):
    """Response returned by /ready endpoint."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_alias: Optional[str] = None
    ready_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReloadRequest(BaseModel):
    """Payload for POST /model/reload."""

    alias: Optional[str] = Field(default=None, description="Optional alias to reload; defaults to configured champion")


class ReloadResponse(BaseModel):
    """Response returned by POST /model/reload."""

    status: str = "reloaded"
    message: str
    previous_version: str
    new_version: str
    model_name: str
    reloaded_at: str


class GroundTruthFeedbackRequest(BaseModel):
    """Payload for POST /monitoring/feedback."""

    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(description="UUID of the inference request to match")
    actual_churn: Literal[0, 1] = Field(description="True ground-truth outcome: 0=No Churn, 1=Churn")
    source: str = Field(default="manual_feedback", description="Origin of ground-truth label")


class GroundTruthFeedbackResponse(BaseModel):
    """Response returned by POST /monitoring/feedback."""

    status: str = "success"
    request_id: str
    actual_churn: int
    message: str
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MonitoringRunResponse(BaseModel):
    """Response returned by POST /monitoring/run."""

    status: str
    report_id: str
    dataset_drift_detected: bool
    drift_share: float
    data_quality_status: str
    prediction_drift_detected: bool
    retraining_recommendation_status: str
    retraining_trigger_reason: Optional[str] = None
    report_html_path: str
    report_json_path: str
    executed_at: str
