"""Data Validation and Schema Enforcement Module using Pydantic."""

from typing import Dict, List, Literal, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.common.config import AppConfig, get_config
from src.common.exceptions import DataValidationError
from src.common.logger import get_logger

logger = get_logger(__name__)


# Schema contract for a single customer record
class CustomerChurnRecord(BaseModel):
    """Pydantic validation contract for an individual customer record."""

    customerID: Optional[str] = None
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
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
    MonthlyCharges: float = Field(ge=0.0, le=500.0)
    TotalCharges: Optional[float] = Field(default=None)
    Churn: Optional[Literal["Yes", "No"]] = None

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def validate_total_charges(cls, v: Any) -> Optional[float]:
        """Handles float NaN and enforces non-negative bounds."""
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("TotalCharges must be non-negative")
            return float(v)
        return v


class ValidationReport(BaseModel):
    """Detailed summary of the data validation pass."""

    is_valid: bool
    total_records: int
    clean_records: int
    missing_values_imputed: Dict[str, int]
    validation_errors: List[str]
    column_types: Dict[str, str]


def clean_and_coerce_dataframe(df: pd.DataFrame, config: AppConfig | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Performs defensive data cleaning, type coercion, and whitespace handling.

    In the Telco Churn dataset, 'TotalCharges' contains blank spaces ' ' for new
    customers with tenure=0. This function safely coerces them to NaN for subsequent
    imputation and casts numeric columns.

    Args:
        df: Raw input DataFrame.
        config: Application configuration.

    Returns:
        Tuple of (cleaned_df, missing_summary).
    """
    cfg = config or get_config()
    clean_df = df.copy()

    # Handle TotalCharges whitespace artifact
    if "TotalCharges" in clean_df.columns:
        if clean_df["TotalCharges"].dtype == object:
            # Replace whitespace or empty strings with NaN
            clean_df["TotalCharges"] = clean_df["TotalCharges"].astype(str).str.strip()
            clean_df["TotalCharges"] = clean_df["TotalCharges"].replace("", np.nan)
        clean_df["TotalCharges"] = pd.to_numeric(clean_df["TotalCharges"], errors="coerce")

    # Coerce numerical columns
    for num_col in cfg.features.numerical:
        if num_col in clean_df.columns:
            clean_df[num_col] = pd.to_numeric(clean_df[num_col], errors="coerce")

    # Coerce SeniorCitizen to integer
    if "SeniorCitizen" in clean_df.columns:
        clean_df["SeniorCitizen"] = pd.to_numeric(clean_df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Coerce categorical columns to string
    for cat_col in cfg.features.categorical:
        if cat_col in clean_df.columns and cat_col != "SeniorCitizen":
            clean_df[cat_col] = clean_df[cat_col].astype(str)

    missing_summary = {col: int(clean_df[col].isna().sum()) for col in clean_df.columns if clean_df[col].isna().sum() > 0}
    return clean_df, missing_summary


def validate_dataframe(df: pd.DataFrame, config: AppConfig | None = None) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validates raw DataFrame against the schema contract and business rules.

    Args:
        df: Raw input DataFrame.
        config: Application configuration object.

    Returns:
        Tuple of (validated_cleaned_df, validation_report).

    Raises:
        DataValidationError: If critical contract violations or missing required columns occur.
    """
    cfg = config or get_config()
    errors: List[str] = []

    # 1. Check required feature columns
    required_features = cfg.features.numerical + cfg.features.categorical
    missing_cols = [c for c in required_features if c not in df.columns]
    if missing_cols:
        msg = f"Schema violation: Missing required columns: {missing_cols}"
        logger.error(msg)
        raise DataValidationError(msg, details={"missing_columns": missing_cols})

    # 2. Clean and coerce types
    clean_df, missing_summary = clean_and_coerce_dataframe(df, cfg)

    # 3. Check target column if present
    target_col = cfg.data.target_column
    if target_col in clean_df.columns:
        valid_targets = {cfg.data.positive_target_value, cfg.data.negative_target_value}
        invalid_target_rows = ~clean_df[target_col].isin(valid_targets)
        if invalid_target_rows.sum() > 0:
            err = f"Target column '{target_col}' contains invalid values in {invalid_target_rows.sum()} rows."
            errors.append(err)

    # 4. Check negative values in numerical features
    for num_col in cfg.features.numerical:
        neg_count = (clean_df[num_col] < 0).sum()
        if neg_count > 0:
            err = f"Numerical column '{num_col}' contains {neg_count} negative values."
            errors.append(err)

    # 5. Spot-check sample records through Pydantic contract
    sample_records = clean_df.head(100).to_dict(orient="records")
    for idx, rec in enumerate(sample_records):
        try:
            CustomerChurnRecord(**rec)
        except ValidationError as e:
            errors.append(f"Row {idx} contract validation error: {e}")
            if len(errors) > 10:
                break

    is_valid = len(errors) == 0

    report = ValidationReport(
        is_valid=is_valid,
        total_records=len(df),
        clean_records=len(clean_df),
        missing_values_imputed=missing_summary,
        validation_errors=errors,
        column_types={col: str(clean_df[col].dtype) for col in clean_df.columns},
    )

    if not is_valid:
        logger.warning("Data validation completed with %d warnings/errors: %s", len(errors), errors[:3])
    else:
        logger.info(
            "Data validation passed successfully: %d records validated, %d columns verified.",
            len(clean_df),
            len(clean_df.columns),
        )

    return clean_df, report
