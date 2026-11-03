"""Comprehensive unit test suite for statistical drift detection, data quality, and model monitoring."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_quality import ALL_FEATURES, DataQualityMonitor, DataQualityResult
from src.monitoring.drift_detector import DatasetDriftResult, DriftDetector, FeatureDriftResult
from src.monitoring.models import (
    DataQualityReportRecord,
    DriftReportRecord,
    FeatureObservationRecord,
    MonitoringBase,
    get_monitoring_engine,
)
from src.monitoring.monitoring_config import MonitoringConfig
from src.monitoring.performance_monitor import PerformanceMonitor, PerformanceResult
from src.monitoring.prediction_monitor import PredictionDistributionResult, PredictionMonitor
from src.monitoring.report_generator import ReportGenerator
from src.monitoring.retraining_recommendations import (
    RetrainingRecommendation,
    RetrainingRecommendationEngine,
)


@pytest.fixture
def monitoring_config():
    """Provides a deterministic test configuration."""
    return MonitoringConfig(
        min_sample_size=30,
        drift_share_threshold=0.30,
        significance_level=0.05,
        prediction_drift_threshold=0.10,
        data_quality_missing_threshold=0.05,
        cooldown_hours=24,
        reports_dir="reports/monitoring_tests",
    )


@pytest.fixture
def sample_feature_df():
    """Generates a realistic synthetic reference dataframe conforming to the Telco schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], size=n),
        "SeniorCitizen": np.random.choice([0, 1], size=n, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], size=n),
        "Dependents": np.random.choice(["Yes", "No"], size=n),
        "tenure": np.random.randint(1, 72, size=n),
        "PhoneService": np.random.choice(["Yes", "No"], size=n, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            size=n,
        ),
        "MonthlyCharges": np.random.uniform(18.0, 118.0, size=n),
        "TotalCharges": np.random.uniform(18.0, 8000.0, size=n),
    })


# ==============================================================================
# 1. Data Quality Monitoring Tests
# ==============================================================================

def test_data_quality_nominal(monitoring_config, sample_feature_df):
    """Verifies that clean conforming data passes quality evaluation with PASSED."""
    monitor = DataQualityMonitor(monitoring_config)
    res = monitor.evaluate(sample_feature_df)
    assert res.status == "PASSED"
    assert res.missing_values_count == 0
    assert res.missing_values_share == 0.0
    assert res.out_of_range_count == 0
    assert res.unexpected_categories_count == 0
    assert not res.schema_mismatch


def test_data_quality_missing_values_threshold(monitoring_config, sample_feature_df):
    """Verifies that high missing value rates trigger FAILED status."""
    monitor = DataQualityMonitor(monitoring_config)
    df_missing = sample_feature_df.copy()
    # Inject 10% missing values
    total_cells = df_missing.size
    for i in range(int(total_cells * 0.10)):
        col = df_missing.columns[i % len(df_missing.columns)]
        row = i // len(df_missing.columns)
        df_missing.loc[row, col] = np.nan

    res = monitor.evaluate(df_missing)
    assert res.status == "FAILED"
    assert res.missing_values_share > monitoring_config.data_quality_missing_threshold


def test_data_quality_out_of_range(monitoring_config, sample_feature_df):
    """Verifies that out-of-range numerical values trigger WARNING status."""
    monitor = DataQualityMonitor(monitoring_config)
    df_bad = sample_feature_df.copy()
    df_bad.loc[0, "tenure"] = -10
    df_bad.loc[1, "MonthlyCharges"] = 999.0

    res = monitor.evaluate(df_bad)
    assert res.status == "WARNING"
    assert res.out_of_range_count == 2
    assert "tenure" in res.out_of_range_details
    assert "MonthlyCharges" in res.out_of_range_details


def test_data_quality_unexpected_categories(monitoring_config, sample_feature_df):
    """Verifies that unapproved category values trigger WARNING status."""
    monitor = DataQualityMonitor(monitoring_config)
    df_bad = sample_feature_df.copy()
    df_bad.loc[0, "Contract"] = "Decade-to-decade"
    df_bad.loc[1, "InternetService"] = "QuantumBroadband"

    res = monitor.evaluate(df_bad)
    assert res.status == "WARNING"
    assert res.unexpected_categories_count == 2
    assert "Contract" in res.invalid_category_details


def test_data_quality_schema_mismatch(monitoring_config, sample_feature_df):
    """Verifies that missing required columns trigger schema_mismatch and FAILED status."""
    monitor = DataQualityMonitor(monitoring_config)
    df_bad = sample_feature_df.drop(columns=["tenure", "Contract"])

    res = monitor.evaluate(df_bad)
    assert res.status == "FAILED"
    assert res.schema_mismatch
    assert "tenure" in res.missing_columns
    assert "Contract" in res.missing_columns


def test_data_quality_empty_dataframe(monitoring_config):
    """Verifies that an empty dataframe returns FAILED status gracefully."""
    monitor = DataQualityMonitor(monitoring_config)
    res = monitor.evaluate(pd.DataFrame())
    assert res.status == "FAILED"
    assert res.schema_mismatch


# ==============================================================================
# 2. Feature & Dataset Drift Detection Tests
# ==============================================================================

def test_numerical_drift_detection(monitoring_config, sample_feature_df):
    """Verifies Kolmogorov-Smirnov test flags drift on shifted numerical distribution."""
    detector = DriftDetector(monitoring_config)
    ref_series = sample_feature_df["tenure"]
    # Shift current tenure substantially
    curr_series = sample_feature_df["tenure"] + 45.0

    res = detector.evaluate_numerical_feature(ref_series, curr_series, "tenure")
    assert res.drift_detected
    assert res.p_value < monitoring_config.significance_level
    assert res.metric_name == "ks"
    assert res.current_stats["wasserstein_distance"] > 10.0


def test_numerical_no_drift(monitoring_config, sample_feature_df):
    """Verifies that identical or in-distribution numerical series produce NO drift."""
    detector = DriftDetector(monitoring_config)
    series = sample_feature_df["MonthlyCharges"]
    res = detector.evaluate_numerical_feature(series, series.copy(), "MonthlyCharges")
    assert not res.drift_detected
    assert res.drift_score == 0.0
    assert res.p_value == 1.0


def test_categorical_drift_chisquare(monitoring_config, sample_feature_df):
    """Verifies Chi-square test flags drift on shifted categorical distribution."""
    detector = DriftDetector(monitoring_config)
    ref_series = pd.Series(["Month-to-month"] * 70 + ["Two year"] * 30)
    curr_series = pd.Series(["Month-to-month"] * 10 + ["Two year"] * 90)

    res = detector.evaluate_categorical_feature(ref_series, curr_series, "Contract")
    assert res.drift_detected
    assert res.metric_name in ["chisquare", "tvd"]


def test_categorical_drift_tvd_fallback(monitoring_config):
    """Verifies TVD fallback operates cleanly on small-sample category comparisons."""
    detector = DriftDetector(monitoring_config)
    ref_series = pd.Series(["DSL", "Fiber optic", "No"])
    curr_series = pd.Series(["Fiber optic", "Fiber optic", "Fiber optic"])

    res = detector.evaluate_categorical_feature(ref_series, curr_series, "InternetService")
    assert res.metric_name == "tvd"
    assert res.drift_score > 0.0


def test_dataset_drift_threshold_gating(monitoring_config, sample_feature_df):
    """Verifies that dataset drift is only flagged when drifted features share >= threshold."""
    detector = DriftDetector(monitoring_config)
    ref_df = sample_feature_df.copy()

    # 1. Zero drift
    res_zero = detector.evaluate_dataset_drift(ref_df, ref_df.copy())
    assert not res_zero.dataset_drift_detected
    assert res_zero.drift_share == 0.0

    # 2. Shift 8 out of 19 features (8/19 = 42.1% >= 30%)
    curr_drifted = ref_df.copy()
    curr_drifted["tenure"] = curr_drifted["tenure"] + 50
    curr_drifted["MonthlyCharges"] = curr_drifted["MonthlyCharges"] * 3.0
    curr_drifted["TotalCharges"] = curr_drifted["TotalCharges"] * 10.0
    curr_drifted["Contract"] = "Two year"
    curr_drifted["InternetService"] = "Fiber optic"
    curr_drifted["PaymentMethod"] = "Mailed check"
    curr_drifted["OnlineSecurity"] = "Yes"
    curr_drifted["TechSupport"] = "Yes"

    res_drift = detector.evaluate_dataset_drift(ref_df, curr_drifted)
    assert res_drift.dataset_drift_detected
    assert res_drift.drift_share >= monitoring_config.drift_share_threshold
    assert res_drift.drifted_features_count >= 6


def test_drift_detector_empty_and_insufficient_window(monitoring_config, sample_feature_df):
    """Verifies graceful handling of empty and below-minimum-sample windows."""
    detector = DriftDetector(monitoring_config)

    # Empty window
    res_empty = detector.evaluate_dataset_drift(sample_feature_df, pd.DataFrame())
    assert res_empty.status == "EMPTY_WINDOW"
    assert not res_empty.dataset_drift_detected

    # Insufficient observations (< min_sample_size=30)
    res_small = detector.evaluate_dataset_drift(sample_feature_df, sample_feature_df.iloc[:15])
    assert res_small.status == "INSUFFICIENT_DATA"
    assert not res_small.dataset_drift_detected
    assert len(res_small.warnings) > 0


# ==============================================================================
# 3. Prediction Output Drift Tests
# ==============================================================================

def test_prediction_drift_normal_and_shifted(monitoring_config):
    """Verifies prediction probability and class balance drift evaluation."""
    monitor = PredictionMonitor(monitoring_config)

    # In-distribution probabilities
    np.random.seed(42)
    probs_normal = np.random.beta(a=1.5, b=3.5, size=100)
    res_normal = monitor.evaluate_prediction_drift(probs_normal)
    assert res_normal.status == "EVALUATED"
    assert not res_normal.probability_drift_detected

    # Severely shifted probabilities (almost all high churn)
    probs_shifted = np.random.uniform(0.75, 0.95, size=100)
    res_shifted = monitor.evaluate_prediction_drift(probs_shifted)
    assert res_shifted.status == "EVALUATED"
    assert res_shifted.probability_drift_detected
    assert res_shifted.class_distribution_drift_detected
    assert res_shifted.ks_statistic > monitoring_config.prediction_drift_threshold


def test_prediction_drift_insufficient_samples(monitoring_config):
    """Verifies prediction monitor returns INSUFFICIENT_DATA when sample count < min_sample_size."""
    monitor = PredictionMonitor(monitoring_config)
    res = monitor.evaluate_prediction_drift(np.array([0.1, 0.2, 0.3]))
    assert res.status == "INSUFFICIENT_DATA"
    assert not res.probability_drift_detected


# ==============================================================================
# 4. Supervised Ground-Truth Performance Tests
# ==============================================================================

def test_performance_monitor_no_ground_truth(monitoring_config):
    """Verifies that when ground truth is absent, status is NO_LABELS and metrics are None."""
    monitor = PerformanceMonitor(monitoring_config)
    df = pd.DataFrame({"churn_probability": [0.1, 0.2, 0.8]})
    res = monitor.evaluate_performance(df)
    assert res.status == "NO_LABELS"
    assert res.roc_auc is None
    assert res.recall is None
    assert not res.performance_degraded


def test_performance_monitor_insufficient_labels(monitoring_config):
    """Verifies that sample size < min_sample_size yields INSUFFICIENT_LABELS."""
    monitor = PerformanceMonitor(monitoring_config)
    df = pd.DataFrame({
        "churn_probability": [0.1] * 10 + [0.9] * 10,
        "actual_churn": [0] * 10 + [1] * 10,
    })
    res = monitor.evaluate_performance(df)
    assert res.status == "INSUFFICIENT_LABELS"
    assert res.roc_auc is None


def test_performance_monitor_evaluated_metrics(monitoring_config):
    """Verifies calculation of supervised evaluation metrics when ground truth is sufficient."""
    monitor = PerformanceMonitor(monitoring_config)
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    y_prob = np.clip(y_true * 0.6 + np.random.uniform(0.1, 0.3, size=n), 0.0, 1.0)
    df = pd.DataFrame({"churn_probability": y_prob, "actual_churn": y_true})

    res = monitor.evaluate_performance(df)
    assert res.status == "EVALUATED"
    assert res.roc_auc is not None and res.roc_auc > 0.80
    assert res.pr_auc is not None and res.pr_auc > 0.60
    assert res.precision is not None
    assert res.recall is not None
    assert res.f1_score is not None
    assert res.brier_score is not None
    assert res.true_positives is not None
    assert not res.performance_degraded


def test_performance_monitor_degradation_flag(monitoring_config):
    """Verifies that low recall triggers performance_degraded=True with explanation."""
    monitor = PerformanceMonitor(monitoring_config)
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    # Severe degradation: probabilities close to 0 even for churners
    y_prob = np.random.uniform(0.01, 0.10, size=n)
    df = pd.DataFrame({"churn_probability": y_prob, "actual_churn": y_true})

    res = monitor.evaluate_performance(df)
    assert res.status == "EVALUATED"
    assert res.performance_degraded
    assert "Recall" in res.degradation_reason


# ==============================================================================
# 5. Retraining Recommendation Engine & Governance Tests
# ==============================================================================

def test_retraining_recommendation_nominal(monitoring_config):
    """Verifies that nominal conditions yield status=NO_ACTION."""
    engine = RetrainingRecommendationEngine(monitoring_config)
    drift = DatasetDriftResult(status="EVALUATED", reference_rows=100, current_rows=100, total_features=19, drifted_features_count=0, drift_share=0.0, drift_share_threshold=0.3, dataset_drift_detected=False)
    quality = DataQualityResult(status="PASSED", total_rows=100, total_columns=19, missing_values_count=0, missing_values_share=0.0, out_of_range_count=0, unexpected_categories_count=0, schema_mismatch=False)
    pred = PredictionDistributionResult(sample_count=100, mean_probability=0.28, std_probability=0.1, predicted_churn_rate=0.26, reference_churn_rate=0.265, ks_statistic=0.02, p_value=0.9, wasserstein_distance=0.01, probability_drift_detected=False, class_distribution_drift_detected=False)
    perf = PerformanceResult(status="NO_LABELS", sample_count=100, labeled_sample_count=0)

    rec = engine.evaluate_recommendation("churn-model", "4", drift, quality, pred, perf)
    assert rec.status == "NO_ACTION"
    assert rec.trigger_reason is None


def test_retraining_recommendation_drift_trigger(monitoring_config):
    """Verifies that sustained dataset drift triggers RECOMMENDED status."""
    engine = RetrainingRecommendationEngine(monitoring_config)
    drift = DatasetDriftResult(status="EVALUATED", reference_rows=100, current_rows=100, total_features=19, drifted_features_count=8, drift_share=0.42, drift_share_threshold=0.3, dataset_drift_detected=True)
    quality = DataQualityResult(status="PASSED", total_rows=100, total_columns=19, missing_values_count=0, missing_values_share=0.0, out_of_range_count=0, unexpected_categories_count=0, schema_mismatch=False)
    pred = PredictionDistributionResult(sample_count=100, mean_probability=0.28, std_probability=0.1, predicted_churn_rate=0.26, reference_churn_rate=0.265, ks_statistic=0.02, p_value=0.9, wasserstein_distance=0.01, probability_drift_detected=False, class_distribution_drift_detected=False)
    perf = PerformanceResult(status="NO_LABELS", sample_count=100, labeled_sample_count=0)

    rec = engine.evaluate_recommendation("churn-model", "4", drift, quality, pred, perf)
    assert rec.status == "RECOMMENDED"
    assert rec.trigger_reason == "FEATURE_DRIFT"
    assert rec.cooldown_until is not None
    assert any("Champion vs Challenger" in step for step in rec.remediation_steps)


def test_retraining_recommendation_cooldown_active(monitoring_config):
    """Verifies that recommendations within the cooldown period return COOLDOWN_ACTIVE."""
    engine = RetrainingRecommendationEngine(monitoring_config)
    drift = DatasetDriftResult(status="EVALUATED", reference_rows=100, current_rows=100, total_features=19, drifted_features_count=8, drift_share=0.42, drift_share_threshold=0.3, dataset_drift_detected=True)
    quality = DataQualityResult(status="PASSED", total_rows=100, total_columns=19, missing_values_count=0, missing_values_share=0.0, out_of_range_count=0, unexpected_categories_count=0, schema_mismatch=False)
    pred = PredictionDistributionResult(sample_count=100, mean_probability=0.28, std_probability=0.1, predicted_churn_rate=0.26, reference_churn_rate=0.265, ks_statistic=0.02, p_value=0.9, wasserstein_distance=0.01, probability_drift_detected=False, class_distribution_drift_detected=False)
    perf = PerformanceResult(status="NO_LABELS", sample_count=100, labeled_sample_count=0)

    now = datetime.now(timezone.utc)
    last_rec = now - timedelta(hours=2)  # 2 hours ago < 24 hours cooldown

    rec = engine.evaluate_recommendation(
        "churn-model", "4", drift, quality, pred, perf, last_recommendation_timestamp=last_rec, now=now
    )
    assert rec.status == "COOLDOWN_ACTIVE"
    assert rec.trigger_reason == "FEATURE_DRIFT"


def test_retraining_recommendation_no_automatic_promotion(monitoring_config):
    """CRITICAL: Verifies that retraining recommendation never alters or promotes champion model automatically."""
    engine = RetrainingRecommendationEngine(monitoring_config)
    drift = DatasetDriftResult(status="EVALUATED", reference_rows=100, current_rows=100, total_features=19, drifted_features_count=10, drift_share=0.52, drift_share_threshold=0.3, dataset_drift_detected=True)
    quality = DataQualityResult(status="PASSED", total_rows=100, total_columns=19, missing_values_count=0, missing_values_share=0.0, out_of_range_count=0, unexpected_categories_count=0, schema_mismatch=False)
    pred = PredictionDistributionResult(sample_count=100, mean_probability=0.28, std_probability=0.1, predicted_churn_rate=0.26, reference_churn_rate=0.265, ks_statistic=0.02, p_value=0.9, wasserstein_distance=0.01, probability_drift_detected=False, class_distribution_drift_detected=False)
    perf = PerformanceResult(status="NO_LABELS", sample_count=100, labeled_sample_count=0)

    rec = engine.evaluate_recommendation("churn-model", "4", drift, quality, pred, perf)
    assert rec.status == "RECOMMENDED"
    # Verify governance remediation guidance explicitly prohibits automatic promotion
    guidance = " ".join(rec.remediation_steps)
    assert "champion is NOT replaced automatically" in guidance
    assert "Model promotion requires formal verification" in guidance


# ==============================================================================
# 6. Privacy Safeguards & Database Persistence Tests
# ==============================================================================

def test_privacy_safeguards_feature_buffer(tmp_path):
    """Verifies that FeatureObservationRecord stores only de-identified features and auto-purges."""
    db_path = tmp_path / "privacy_test.db"
    db_uri = f"sqlite:///{db_path.as_posix()}"
    engine = get_monitoring_engine(db_uri)
    MonitoringBase.metadata.create_all(engine)

    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Verify no PII fields exist in table columns
    col_names = [c.name for c in FeatureObservationRecord.__table__.columns]
    assert "customerID" not in col_names
    assert "customer_name" not in col_names
    assert "email" not in col_names
    assert "phone_number" not in col_names
    assert "ssn" not in col_names

    # Insert an ephemeral observation
    now = datetime.now(timezone.utc)
    rec = FeatureObservationRecord(
        observation_id="anon-obs-001",
        timestamp=now,
        tenure=24.0,
        MonthlyCharges=65.5,
        TotalCharges=1572.0,
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="One year",
        PaperlessBilling="No",
        PaymentMethod="Mailed check",
    )
    session.add(rec)
    session.commit()

    retrieved = session.query(FeatureObservationRecord).first()
    assert retrieved.observation_id == "anon-obs-001"
    assert retrieved.tenure == 24.0

    session.close()
    engine.dispose()


def test_report_generator_json_and_html(tmp_path, monitoring_config, sample_feature_df):
    """Verifies that ReportGenerator produces valid JSON and interactive HTML files."""
    gen = ReportGenerator(monitoring_config)
    now = datetime.now(timezone.utc)
    rep_id = "test-report-1234"

    drift = DatasetDriftResult(status="EVALUATED", reference_rows=100, current_rows=100, total_features=19, drifted_features_count=2, drift_share=0.105, drift_share_threshold=0.3, dataset_drift_detected=False)
    quality = DataQualityResult(status="PASSED", total_rows=100, total_columns=19, missing_values_count=0, missing_values_share=0.0, out_of_range_count=0, unexpected_categories_count=0, schema_mismatch=False)
    pred = PredictionDistributionResult(sample_count=100, mean_probability=0.28, std_probability=0.1, predicted_churn_rate=0.26, reference_churn_rate=0.265, ks_statistic=0.02, p_value=0.9, wasserstein_distance=0.01, probability_drift_detected=False, class_distribution_drift_detected=False)
    perf = PerformanceResult(status="NO_LABELS", sample_count=100, labeled_sample_count=0)
    rec = RetrainingRecommendation(recommendation_id="rec-001", timestamp=now, model_name="churn-model", model_version="4", status="NO_ACTION")

    json_path = gen.generate_json_report(rep_id, now, "churn-model", "4", drift, quality, pred, perf, rec)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["report_id"] == rep_id
    assert data["data_quality"]["status"] == "PASSED"

    html_path = gen.generate_html_report(rep_id, now, "churn-model", "4", sample_feature_df, sample_feature_df, drift, quality, pred, perf, rec)
    assert html_path.exists()
    assert html_path.stat().st_size > 500
