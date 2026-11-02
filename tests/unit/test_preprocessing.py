"""Unit Tests for Feature Preprocessor and Pipeline Encapsulation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.common.config import AppConfig
from src.data.preprocessor import (
    build_full_pipeline,
    build_preprocessor,
    get_feature_names,
)
from src.models.train import prepare_features_and_target


def test_build_preprocessor_transformation_shape(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Tests that ColumnTransformer transforms data into expected dense 2D array."""
    X, _ = prepare_features_and_target(clean_churn_df, test_config)
    preprocessor = build_preprocessor(test_config)
    X_trans = preprocessor.fit_transform(X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(X)
    assert X_trans.shape[1] > len(test_config.features.numerical)  # Categoricals expanded by OneHot


def test_handling_unseen_categorical_categories(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Verifies that unknown categorical levels during inference do not crash the pipeline."""
    X, _ = prepare_features_and_target(clean_churn_df, test_config)
    preprocessor = build_preprocessor(test_config)
    preprocessor.fit(X)

    # Create unseen category values
    X_unseen = X.iloc[:5].copy()
    X_unseen["InternetService"] = "Satellite_5G_Unseen"
    X_unseen["PaymentMethod"] = "Crypto_Token_Unseen"

    # Must transform gracefully without KeyError or ValueError
    X_trans = preprocessor.transform(X_unseen)
    assert X_trans.shape[0] == 5
    assert not np.isnan(X_trans).any()


def test_get_feature_names_matches_transformed_width(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Verifies that extracted human-readable feature names match transformed array dimensions."""
    X, _ = prepare_features_and_target(clean_churn_df, test_config)
    preprocessor = build_preprocessor(test_config)
    X_trans = preprocessor.fit_transform(X)

    names = get_feature_names(preprocessor)
    assert len(names) == X_trans.shape[1]
    assert all(isinstance(n, str) for n in names)


def test_full_pipeline_encapsulation_predict_proba(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Tests that unified Pipeline handles raw input and returns calibrated probabilities."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)
    full_pipe = build_full_pipeline(LogisticRegression(max_iter=500), test_config)

    full_pipe.fit(X, y)
    probs = full_pipe.predict_proba(X)

    assert probs.shape == (len(X), 2)
    # Each row sum must equal 1.0
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)
    # Probabilities in range [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
