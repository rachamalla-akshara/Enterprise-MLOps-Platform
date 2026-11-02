"""Unit Tests for Estimator Architectures, Determinism, and Evaluation Metrics."""

import numpy as np
import pandas as pd
import pytest

from src.common.config import AppConfig
from src.data.preprocessor import build_full_pipeline
from src.models.churn_models import (
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
)
from src.models.evaluate import (
    benchmark_inference_latency,
    evaluate_model,
    find_optimal_f1_threshold,
)
from src.models.train import prepare_features_and_target


def test_model_instantiations(test_config: AppConfig):
    """Verifies that all three core model types instantiate with proper hyperparameter types."""
    lr = get_logistic_regression(test_config)
    rf = get_random_forest(test_config)
    xgb = get_xgboost(test_config)

    assert lr.class_weight == "balanced"
    assert rf.random_state == test_config.project.random_seed
    assert xgb.random_state == test_config.project.random_seed
    assert xgb.scale_pos_weight > 1.0


def test_training_determinism(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Verifies that training with fixed seed produces identical predictions across runs."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)

    pipe1 = build_full_pipeline(get_random_forest(test_config), test_config)
    pipe1.fit(X, y)
    probs1 = pipe1.predict_proba(X)[:, 1]

    pipe2 = build_full_pipeline(get_random_forest(test_config), test_config)
    pipe2.fit(X, y)
    probs2 = pipe2.predict_proba(X)[:, 1]

    np.testing.assert_allclose(probs1, probs2, atol=1e-7, err_msg="Model training non-deterministic!")


def test_evaluate_model_metrics_structure(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Verifies that evaluate_model calculates all standard metrics, costs, and latencies."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)
    pipe = build_full_pipeline(get_logistic_regression(test_config), test_config)
    pipe.fit(X, y)

    metrics = evaluate_model(pipe, X, y, model_name="TestLR", config=test_config)

    required_keys = [
        "model_name",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "brier_score",
        "confusion_matrix",
        "business_cost",
        "latency_benchmark",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in evaluation metrics dictionary!"

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["confusion_matrix"]["tp"] >= 0
    assert metrics["business_cost"]["total_cost"] >= 0.0


def test_benchmark_inference_latency_positive(clean_churn_df: pd.DataFrame, test_config: AppConfig):
    """Tests that empirical latency benchmark produces non-zero positive millisecond measurements."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)
    pipe = build_full_pipeline(get_logistic_regression(test_config), test_config)
    pipe.fit(X, y)

    latency = benchmark_inference_latency(pipe, X.head(5), n_iterations=20)
    assert latency["mean_ms"] > 0.0
    assert latency["p50_ms"] > 0.0
    assert latency["p95_ms"] >= latency["p50_ms"]
    assert latency["p99_ms"] >= latency["p95_ms"]


def test_find_optimal_f1_threshold():
    """Tests optimal threshold locator with synthetic scores."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    best_thresh, best_f1 = find_optimal_f1_threshold(y_true, y_probs)
    assert 0.0 < best_thresh < 1.0
    assert best_f1 > 0.8
