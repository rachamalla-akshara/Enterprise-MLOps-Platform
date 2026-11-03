"""SQLAlchemy database models for statistical drift, data quality, and model monitoring."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

# Using explicit schema name for PostgreSQL production; SQLite maps via schema_translate_map
metadata = MetaData(schema="monitoring")
MonitoringBase = declarative_base(metadata=metadata)


class DriftReportRecord(MonitoringBase):
    """Aggregate summary record for a dataset and feature drift evaluation cycle."""

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), index=True, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    reference_dataset = Column(String(256), nullable=False)
    reference_rows = Column(Integer, nullable=False)
    current_rows = Column(Integer, nullable=False)
    total_features = Column(Integer, nullable=False)
    drifted_features = Column(Integer, nullable=False)
    drift_share = Column(Float, nullable=False)
    drift_share_threshold = Column(Float, nullable=False)
    dataset_drift_detected = Column(Boolean, nullable=False)
    prediction_drift_detected = Column(Boolean, nullable=False, default=False)
    report_html_path = Column(String(512), nullable=True)
    report_json_path = Column(String(512), nullable=True)
    config_version = Column(String(32), nullable=False, default="1.0.0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "reference_dataset": self.reference_dataset,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "drift_share": self.drift_share,
            "drift_share_threshold": self.drift_share_threshold,
            "dataset_drift_detected": self.dataset_drift_detected,
            "prediction_drift_detected": self.prediction_drift_detected,
            "report_html_path": self.report_html_path,
            "report_json_path": self.report_json_path,
            "config_version": self.config_version,
        }


class FeatureDriftResultRecord(MonitoringBase):
    """Statistical drift test results for an individual input feature."""

    __tablename__ = "feature_drift_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), index=True, nullable=False)
    feature_name = Column(String(64), index=True, nullable=False)
    feature_type = Column(String(32), nullable=False)  # "numerical" or "categorical"
    metric_name = Column(String(64), nullable=False)   # "ks", "wasserstein", "chisquare", etc.
    drift_score = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    reference_stats = Column(Text, nullable=True)       # JSON string of reference statistics
    current_stats = Column(Text, nullable=True)         # JSON string of current statistics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "report_id": self.report_id,
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "metric_name": self.metric_name,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "reference_stats": json.loads(self.reference_stats) if self.reference_stats else None,
            "current_stats": json.loads(self.current_stats) if self.current_stats else None,
        }


class DataQualityReportRecord(MonitoringBase):
    """Data quality and integrity audit report."""

    __tablename__ = "data_quality_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    total_columns = Column(Integer, nullable=False)
    total_rows = Column(Integer, nullable=False)
    missing_values_count = Column(Integer, nullable=False)
    missing_values_share = Column(Float, nullable=False)
    out_of_range_count = Column(Integer, nullable=False, default=0)
    unexpected_categories_count = Column(Integer, nullable=False, default=0)
    schema_mismatch = Column(Boolean, nullable=False, default=False)
    quality_status = Column(String(32), nullable=False, default="PASSED")  # "PASSED", "WARNING", "FAILED"
    details = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_columns": self.total_columns,
            "total_rows": self.total_rows,
            "missing_values_count": self.missing_values_count,
            "missing_values_share": self.missing_values_share,
            "out_of_range_count": self.out_of_range_count,
            "unexpected_categories_count": self.unexpected_categories_count,
            "schema_mismatch": self.schema_mismatch,
            "quality_status": self.quality_status,
            "details": json.loads(self.details) if self.details else None,
        }


class PredictionDistributionReportRecord(MonitoringBase):
    """Inference probability and prediction class output drift monitoring."""

    __tablename__ = "prediction_distribution_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    sample_count = Column(Integer, nullable=False)
    mean_probability = Column(Float, nullable=False)
    std_probability = Column(Float, nullable=False)
    predicted_churn_rate = Column(Float, nullable=False)
    reference_churn_rate = Column(Float, nullable=False)
    probability_drift_p_value = Column(Float, nullable=True)
    probability_drift_detected = Column(Boolean, nullable=False, default=False)
    class_distribution_drift_detected = Column(Boolean, nullable=False, default=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sample_count": self.sample_count,
            "mean_probability": self.mean_probability,
            "std_probability": self.std_probability,
            "predicted_churn_rate": self.predicted_churn_rate,
            "reference_churn_rate": self.reference_churn_rate,
            "probability_drift_p_value": self.probability_drift_p_value,
            "probability_drift_detected": self.probability_drift_detected,
            "class_distribution_drift_detected": self.class_distribution_drift_detected,
        }


class GroundTruthFeedbackRecord(MonitoringBase):
    """Delayed ground-truth churn outcomes matched securely by request_id."""

    __tablename__ = "ground_truth_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), index=True, unique=True, nullable=False)
    actual_churn = Column(Integer, nullable=False)  # 0: No Churn, 1: Churn
    feedback_timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    source = Column(String(64), nullable=False, default="manual_feedback")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "actual_churn": self.actual_churn,
            "feedback_timestamp": self.feedback_timestamp.isoformat() if self.feedback_timestamp else None,
            "source": self.source,
        }


class ModelPerformanceReportRecord(MonitoringBase):
    """Model evaluation metrics calculated only when verified ground-truth labels are available."""

    __tablename__ = "model_performance_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    sample_count = Column(Integer, nullable=False)
    labeled_sample_count = Column(Integer, nullable=False)
    roc_auc = Column(Float, nullable=True)
    pr_auc = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    true_positives = Column(Integer, nullable=True)
    false_positives = Column(Integer, nullable=True)
    true_negatives = Column(Integer, nullable=True)
    false_negatives = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False, default="EVALUATED")  # "EVALUATED", "INSUFFICIENT_LABELS", "NO_LABELS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "sample_count": self.sample_count,
            "labeled_sample_count": self.labeled_sample_count,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "brier_score": self.brier_score,
            "confusion_matrix": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "tn": self.true_negatives,
                "fn": self.false_negatives,
            } if self.true_positives is not None else None,
            "status": self.status,
        }


class RetrainingRecommendationRecord(MonitoringBase):
    """Audit record for retraining recommendations generated by the monitoring engine.

    IMPORTANT: A recommendation is an informational governance signal.
    It NEVER automatically promotes a model or replaces the champion without human/CI approval.
    """

    __tablename__ = "retraining_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(String(64), index=True, unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    trigger_reason = Column(String(64), nullable=False)  # "FEATURE_DRIFT", "PREDICTION_DRIFT", "PERFORMANCE_DROP", "DATA_QUALITY_FAIL"
    trigger_metrics = Column(Text, nullable=False)       # JSON string containing supporting metrics
    recommendation_status = Column(String(32), nullable=False)  # "RECOMMENDED", "COOLDOWN_ACTIVE", "INSUFFICIENT_EVIDENCE"
    cooldown_until = Column(DateTime(timezone=True), nullable=True)
    remediation_steps = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "recommendation_id": self.recommendation_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "trigger_reason": self.trigger_reason,
            "trigger_metrics": json.loads(self.trigger_metrics) if self.trigger_metrics else None,
            "recommendation_status": self.recommendation_status,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "remediation_steps": self.remediation_steps,
        }


class FeatureObservationRecord(MonitoringBase):
    """Bounded, ephemeral de-identified feature buffer for statistical distribution monitoring.

    PRIVACY GUARANTEES:
    - Strictly de-identified: Direct identifiers (customerID, names, phone numbers, emails, addresses)
      are NEVER stored.
    - Used exclusively to compute two-sample distribution comparisons (KS, Wasserstein, Chi-Square).
    - Subject to bounded FIFO limits (buffer_max_size=5000) and automated TTL purge.
    """

    __tablename__ = "feature_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    observation_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    tenure = Column(Float, nullable=False)
    MonthlyCharges = Column(Float, nullable=False)
    TotalCharges = Column(Float, nullable=False)
    gender = Column(String(16), nullable=False)
    SeniorCitizen = Column(Integer, nullable=False)
    Partner = Column(String(8), nullable=False)
    Dependents = Column(String(8), nullable=False)
    PhoneService = Column(String(8), nullable=False)
    MultipleLines = Column(String(32), nullable=False)
    InternetService = Column(String(32), nullable=False)
    OnlineSecurity = Column(String(32), nullable=False)
    OnlineBackup = Column(String(32), nullable=False)
    DeviceProtection = Column(String(32), nullable=False)
    TechSupport = Column(String(32), nullable=False)
    StreamingTV = Column(String(32), nullable=False)
    StreamingMovies = Column(String(32), nullable=False)
    Contract = Column(String(32), nullable=False)
    PaperlessBilling = Column(String(8), nullable=False)
    PaymentMethod = Column(String(64), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tenure": self.tenure,
            "MonthlyCharges": self.MonthlyCharges,
            "TotalCharges": self.TotalCharges,
            "gender": self.gender,
            "SeniorCitizen": self.SeniorCitizen,
            "Partner": self.Partner,
            "Dependents": self.Dependents,
            "PhoneService": self.PhoneService,
            "MultipleLines": self.MultipleLines,
            "InternetService": self.InternetService,
            "OnlineSecurity": self.OnlineSecurity,
            "OnlineBackup": self.OnlineBackup,
            "DeviceProtection": self.DeviceProtection,
            "TechSupport": self.TechSupport,
            "StreamingTV": self.StreamingTV,
            "StreamingMovies": self.StreamingMovies,
            "Contract": self.Contract,
            "PaperlessBilling": self.PaperlessBilling,
            "PaymentMethod": self.PaymentMethod,
        }


def get_monitoring_engine(db_uri: str):
    """Creates a SQLAlchemy engine configured for dialect-specific schema handling.

    When running on SQLite, uses schema_translate_map to strip the 'monitoring' schema prefix,
    allowing clean local unit tests while keeping PostgreSQL isolated under schema 'monitoring'.
    """
    is_sqlite = db_uri.startswith("sqlite")
    options = {}
    if is_sqlite:
        options["schema_translate_map"] = {"monitoring": None}

    engine = create_engine(
        db_uri,
        connect_args={"check_same_thread": False} if is_sqlite else {},
        pool_pre_ping=True,
    )
    if options:
        engine = engine.execution_options(**options)
    return engine
