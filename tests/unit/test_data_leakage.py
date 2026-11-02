"""Data Leakage and Isolation Verification Test Suite."""

import copy
import numpy as np
import pandas as pd
import pytest

from src.common.config import AppConfig
from src.data.loader import split_data
from src.data.preprocessor import build_preprocessor
from src.models.train import prepare_features_and_target


def test_stratified_split_independence(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Tests that train, validation, and test partitions are completely disjoint."""
    train_df, val_df, test_df = split_data(clean_churn_df, test_config)

    train_ids = set(train_df["customerID"])
    val_ids = set(val_df["customerID"])
    test_ids = set(test_df["customerID"])

    # Assert mutual exclusivity (no overlap between sets)
    assert len(train_ids.intersection(val_ids)) == 0, "Data Leakage: Train and Val share records!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data Leakage: Train and Test share records!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data Leakage: Val and Test share records!"

    # Assert total partition completeness
    assert len(train_df) + len(val_df) + len(test_df) == len(clean_churn_df)


def test_zero_leakage_scaler_fit_isolation(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Verifies that preprocessor statistics are derived strictly from X_train.

    Injecting extreme outliers or distribution shifts into X_test must NOT
    alter the fitted parameters (mean, scale, medians) of the preprocessor.
    """
    train_df, val_df, test_df = split_data(clean_churn_df, test_config)
    X_train, _ = prepare_features_and_target(train_df, test_config)
    X_test, _ = prepare_features_and_target(test_df, test_config)

    preprocessor = build_preprocessor(test_config)
    preprocessor.fit(X_train)

    # Extract fitted scaler statistics on X_train
    num_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    train_means_baseline = copy.deepcopy(num_scaler.mean_)
    train_scales_baseline = copy.deepcopy(num_scaler.scale_)

    # Inject extreme 100x outlier values into X_test
    X_test_perturbed = X_test.copy()
    X_test_perturbed["MonthlyCharges"] = X_test_perturbed["MonthlyCharges"] * 1000.0
    X_test_perturbed["tenure"] = X_test_perturbed["tenure"] + 5000

    # Transforming perturbed test set must not alter the already fitted scaler
    _ = preprocessor.transform(X_test_perturbed)

    np.testing.assert_allclose(
        num_scaler.mean_,
        train_means_baseline,
        err_msg="Data Leakage: Scaler mean changed after transforming perturbed test data!",
    )
    np.testing.assert_allclose(
        num_scaler.scale_,
        train_scales_baseline,
        err_msg="Data Leakage: Scaler variance changed after transforming perturbed test data!",
    )


def test_zero_leakage_imputer_isolation(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Verifies that imputer statistics reflect strictly X_train values."""
    train_df, _, _ = split_data(clean_churn_df, test_config)
    X_train, _ = prepare_features_and_target(train_df, test_config)

    preprocessor = build_preprocessor(test_config)
    preprocessor.fit(X_train)

    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    fitted_medians = num_imputer.statistics_

    expected_tenure_median = float(X_train["tenure"].median())
    expected_monthly_median = float(X_train["MonthlyCharges"].median())

    assert np.isclose(fitted_medians[0], expected_tenure_median, atol=1e-5), "Imputer statistics leaked or incorrect!"
    assert np.isclose(fitted_medians[1], expected_monthly_median, atol=1e-5), "Imputer statistics leaked or incorrect!"


def test_permutation_invariance_of_predictions(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Verifies that individual record predictions are independent of row ordering."""
    from src.data.preprocessor import build_full_pipeline
    from src.models.churn_models import get_logistic_regression

    train_df, _, test_df = split_data(clean_churn_df, test_config)
    X_train, y_train = prepare_features_and_target(train_df, test_config)
    X_test, _ = prepare_features_and_target(test_df, test_config)

    pipeline = build_full_pipeline(get_logistic_regression(test_config), test_config)
    pipeline.fit(X_train, y_train)

    # Standard predictions
    probs_orig = pipeline.predict_proba(X_test)[:, 1]

    # Shuffled order predictions
    shuffled_indices = np.random.RandomState(42).permutation(len(X_test))
    X_test_shuffled = X_test.iloc[shuffled_indices]
    probs_shuffled = pipeline.predict_proba(X_test_shuffled)[:, 1]

    # Unshuffle
    unshuffled_probs = np.zeros_like(probs_shuffled)
    unshuffled_probs[shuffled_indices] = probs_shuffled

    np.testing.assert_allclose(
        probs_orig,
        unshuffled_probs,
        atol=1e-6,
        err_msg="Predictions dependent on batch row order (Cross-row leakage violation)!",
    )
