"""Data quality and integrity audit engine for model input batches."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.monitoring.monitoring_config import MonitoringConfig

logger = get_logger(__name__)

# Expected model features
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Permissible categorical values
VALID_CATEGORIES: Dict[str, Set[Any]] = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1, "0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

# Permissible numerical boundaries
NUMERICAL_BOUNDS: Dict[str, tuple[float, float]] = {
    "tenure": (0.0, 120.0),
    "MonthlyCharges": (0.0, 500.0),
    "TotalCharges": (0.0, 50000.0),
}


@dataclass
class DataQualityResult:
    """Structured data quality audit outcome."""

    status: str  # "PASSED", "WARNING", "FAILED"
    total_rows: int
    total_columns: int
    missing_values_count: int
    missing_values_share: float
    out_of_range_count: int
    unexpected_categories_count: int
    schema_mismatch: bool
    missing_columns: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    column_missing_counts: Dict[str, int] = field(default_factory=dict)
    invalid_category_details: Dict[str, List[Any]] = field(default_factory=dict)
    out_of_range_details: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values_count": self.missing_values_count,
            "missing_values_share": round(self.missing_values_share, 4),
            "out_of_range_count": self.out_of_range_count,
            "unexpected_categories_count": self.unexpected_categories_count,
            "schema_mismatch": self.schema_mismatch,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "column_missing_counts": self.column_missing_counts,
            "invalid_category_details": self.invalid_category_details,
            "out_of_range_details": self.out_of_range_details,
        }


class DataQualityMonitor:
    """Validates schemas, missing rates, out-of-range numericals, and category drift."""

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or MonitoringConfig()

    def evaluate(self, df: pd.DataFrame) -> DataQualityResult:
        """Runs comprehensive data quality checks on the provided observation DataFrame."""
        if df.empty:
            return DataQualityResult(
                status="FAILED",
                total_rows=0,
                total_columns=len(df.columns),
                missing_values_count=0,
                missing_values_share=0.0,
                out_of_range_count=0,
                unexpected_categories_count=0,
                schema_mismatch=True,
                missing_columns=ALL_FEATURES.copy(),
            )

        total_rows = len(df)
        total_columns = len(df.columns)

        # 1. Schema check
        existing_cols = set(df.columns)
        missing_cols = [c for c in ALL_FEATURES if c not in existing_cols]
        unexpected_cols = [c for c in df.columns if c not in set(ALL_FEATURES)]
        schema_mismatch = len(missing_cols) > 0

        # 2. Missing values check
        total_missing = int(df.isna().sum().sum())
        total_cells = total_rows * total_columns
        missing_share = float(total_missing / total_cells) if total_cells > 0 else 0.0

        col_missing = {}
        for col in df.columns:
            cnt = int(df[col].isna().sum())
            if cnt > 0:
                col_missing[col] = cnt

        # 3. Numerical range check
        out_of_range_count = 0
        out_of_range_details = {}
        for col, (low, high) in NUMERICAL_BOUNDS.items():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                violations = int(((series < low) | (series > high)).sum())
                if violations > 0:
                    out_of_range_count += violations
                    out_of_range_details[col] = violations

        # 4. Categorical validity check
        unexpected_cat_count = 0
        invalid_cat_details = {}
        for col, valid_set in VALID_CATEGORIES.items():
            if col in df.columns:
                series = df[col].dropna()
                invalid_mask = ~series.isin(valid_set)
                violations = int(invalid_mask.sum())
                if violations > 0:
                    unexpected_cat_count += violations
                    invalid_cat_details[col] = list(series[invalid_mask].unique()[:5])

        # 5. Determine overall quality status
        if schema_mismatch or (missing_share > self.config.data_quality_missing_threshold):
            status = "FAILED"
        elif out_of_range_count > 0 or unexpected_cat_count > 0 or total_missing > 0:
            status = "WARNING"
        else:
            status = "PASSED"

        result = DataQualityResult(
            status=status,
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values_count=total_missing,
            missing_values_share=missing_share,
            out_of_range_count=out_of_range_count,
            unexpected_categories_count=unexpected_cat_count,
            schema_mismatch=schema_mismatch,
            missing_columns=missing_cols,
            unexpected_columns=unexpected_cols,
            column_missing_counts=col_missing,
            invalid_category_details=invalid_cat_details,
            out_of_range_details=out_of_range_details,
        )

        logger.info(
            "Data quality evaluated: status=%s, missing_share=%.3f, out_of_range=%d, invalid_cats=%d",
            status,
            missing_share,
            out_of_range_count,
            unexpected_cat_count,
        )
        return result
