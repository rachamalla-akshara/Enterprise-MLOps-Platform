"""Unit Tests for Data Validation, Schema Enforcement, and Defensive Cleaning."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.common.config import AppConfig
from src.common.exceptions import DataValidationError
from src.data.validator import (
    CustomerChurnRecord,
    clean_and_coerce_dataframe,
    validate_dataframe,
)


def test_customer_churn_record_valid_contract():
    """Tests Pydantic CustomerChurnRecord contract with valid payload."""
    valid_payload = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Churn": "No",
    }
    record = CustomerChurnRecord(**valid_payload)
    assert record.gender == "Female"
    assert record.tenure == 1
    assert record.MonthlyCharges == 29.85


def test_customer_churn_record_invalid_enum():
    """Tests that contract raises ValidationError when categorical enum is violated."""
    invalid_payload = {
        "gender": "UnknownGender",  # Invalid
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 500.0,
    }
    with pytest.raises(ValidationError):
        CustomerChurnRecord(**invalid_payload)


def test_clean_and_coerce_whitespace_total_charges(test_config: AppConfig):
    """Tests that blank space ' ' in TotalCharges is safely coerced to NaN."""
    df_raw = pd.DataFrame({
        "tenure": [0, 5],
        "MonthlyCharges": [20.0, 50.0],
        "TotalCharges": [" ", "250.0"],
        "SeniorCitizen": ["0", "1"],
        "gender": ["Male", "Female"],
    })
    clean_df, missing = clean_and_coerce_dataframe(df_raw, test_config)
    assert np.isnan(clean_df.loc[0, "TotalCharges"])
    assert clean_df.loc[1, "TotalCharges"] == 250.0
    assert "TotalCharges" in missing
    assert missing["TotalCharges"] == 1


def test_validate_dataframe_missing_required_column(test_config: AppConfig):
    """Tests that missing required columns triggers DataValidationError."""
    df_incomplete = pd.DataFrame({
        "tenure": [10, 20],
        "MonthlyCharges": [50.0, 60.0],
    })
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(df_incomplete, test_config)
    assert "Missing required columns" in str(exc_info.value)


def test_validate_dataframe_valid_data(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Tests validation pass on complete dataset."""
    df_res, report = validate_dataframe(clean_churn_df, test_config)
    assert report.is_valid is True
    assert len(df_res) == len(clean_churn_df)
    assert report.total_records == len(clean_churn_df)


def test_strict_provenance_validation_real_dataset(test_config: AppConfig):
    """Tests that real dataset passes strict canonical provenance verification."""
    from pathlib import Path
    from src.data.loader import validate_dataset_provenance, CANONICAL_IBM_TELCO_SHA256
    from src.common.config import find_project_root

    root = find_project_root()
    real_path = root / test_config.paths.raw_data

    if real_path.is_file():
        df, prov = validate_dataset_provenance(real_path, test_config)
        assert prov.is_real_data is True
        assert prov.row_count == 7043
        assert prov.column_count == 21
        assert prov.sha256 == CANONICAL_IBM_TELCO_SHA256
        assert prov.provenance_verified is True


def test_strict_provenance_missing_file_raises_error(test_config: AppConfig):
    """Tests that a missing real dataset raises DataIngestionError rather than silently falling back."""
    from pathlib import Path
    from src.data.loader import validate_dataset_provenance
    from src.common.exceptions import DataIngestionError

    fake_path = Path("non_existent_raw_path.csv")
    with pytest.raises(DataIngestionError) as exc_info:
        validate_dataset_provenance(fake_path, test_config)
    assert "PROVENANCE REMEDIATION" in str(exc_info.value)
