"""Pytest Fixtures for Enterprise MLOps Platform Unit & Integration Tests."""

import numpy as np
import pandas as pd
import pytest

from src.common.config import AppConfig, get_config
from src.data.loader import generate_synthetic_telco_dataset
from src.data.validator import clean_and_coerce_dataframe, validate_dataframe


@pytest.fixture(scope="session")
def test_config() -> AppConfig:
    """Provides application configuration instance."""
    return get_config()


@pytest.fixture(scope="session")
def raw_churn_df() -> pd.DataFrame:
    """Generates a small deterministic raw dataset for testing."""
    return generate_synthetic_telco_dataset(n_samples=200, random_seed=42)


@pytest.fixture(scope="session")
def clean_churn_df(raw_churn_df: pd.DataFrame, test_config: AppConfig) -> pd.DataFrame:
    """Provides validated and cleaned DataFrame."""
    clean_df, _ = validate_dataframe(raw_churn_df, test_config)
    return clean_df
