"""Data ingestion, validation, EDA, and preprocessing package."""

from src.data.loader import ingest_dataset, split_data, generate_synthetic_telco_dataset
from src.data.validator import validate_dataframe, CustomerChurnRecord, ValidationReport
from src.data.eda import run_eda
from src.data.preprocessor import build_preprocessor, build_full_pipeline, get_feature_names

__all__ = [
    "ingest_dataset",
    "split_data",
    "generate_synthetic_telco_dataset",
    "validate_dataframe",
    "CustomerChurnRecord",
    "ValidationReport",
    "run_eda",
    "build_preprocessor",
    "build_full_pipeline",
    "get_feature_names",
]
