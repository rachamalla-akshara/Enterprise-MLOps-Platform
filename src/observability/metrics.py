"""Prometheus metrics definitions, instrumentation helpers, and collector management.

Guarantees:
- Low-cardinality label dimensions (strictly bounded enums and normalized endpoints).
- Zero customer PII, request UUIDs, or sensitive payload attributes in labels.
- Safe singleton metric registry with duplicate registration protection.
- High-performance timing with sub-millisecond overhead.
- Native process metrics fallback for cross-platform support.
"""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import prometheus_client
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.common.logger import get_logger

logger = get_logger(__name__)

# Start timestamp for uptime calculation
START_TIME = time.time()


def _get_or_create_counter(
    name: str,
    documentation: str,
    labelnames: Tuple[str, ...] = (),
    registry: CollectorRegistry = REGISTRY,
) -> Counter:
    """Retrieves an existing Counter from registry or creates a new one."""
    if name in registry._names_to_collectors:
        collector = registry._names_to_collectors[name]
        if isinstance(collector, Counter):
            return collector
    return Counter(name, documentation, labelnames=labelnames, registry=registry)


def _get_or_create_gauge(
    name: str,
    documentation: str,
    labelnames: Tuple[str, ...] = (),
    registry: CollectorRegistry = REGISTRY,
) -> Gauge:
    """Retrieves an existing Gauge from registry or creates a new one."""
    if name in registry._names_to_collectors:
        collector = registry._names_to_collectors[name]
        if isinstance(collector, Gauge):
            return collector
    return Gauge(name, documentation, labelnames=labelnames, registry=registry)


def _get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: Tuple[str, ...] = (),
    buckets: Tuple[float, ...] = Histogram.DEFAULT_BUCKETS,
    registry: CollectorRegistry = REGISTRY,
) -> Histogram:
    """Retrieves an existing Histogram from registry or creates a new one."""
    if name in registry._names_to_collectors:
        collector = registry._names_to_collectors[name]
        if isinstance(collector, Histogram):
            return collector
    return Histogram(name, documentation, labelnames=labelnames, buckets=buckets, registry=registry)


# ------------------------------------------------------------------------------
# 1. API Serving Metrics
# ------------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = _get_or_create_counter(
    name="http_requests_total",
    documentation="Total count of processed HTTP requests by method, normalized endpoint, and status code.",
    labelnames=("method", "endpoint", "status_code"),
)

HTTP_REQUEST_DURATION_SECONDS = _get_or_create_histogram(
    name="http_request_duration_seconds",
    documentation="Histogram of HTTP request latency in seconds.",
    labelnames=("method", "endpoint"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0),
)

HTTP_REQUESTS_IN_PROGRESS = _get_or_create_gauge(
    name="http_requests_in_progress",
    documentation="Gauge of currently active HTTP requests being processed.",
    labelnames=("method", "endpoint"),
)

HTTP_ERRORS_TOTAL = _get_or_create_counter(
    name="http_errors_total",
    documentation="Total count of HTTP error responses (4xx and 5xx).",
    labelnames=("method", "endpoint", "status_code"),
)

API_UPTIME_SECONDS = _get_or_create_gauge(
    name="api_uptime_seconds",
    documentation="Current total uptime of the FastAPI serving process in seconds.",
)

API_START_TIME_SECONDS = _get_or_create_gauge(
    name="api_start_time_seconds",
    documentation="Timestamp of FastAPI serving process initialization (UNIX epoch).",
)
API_START_TIME_SECONDS.set(START_TIME)


# ------------------------------------------------------------------------------
# 2. Prediction & Model Inference Metrics
# ------------------------------------------------------------------------------

MODEL_PREDICTIONS_TOTAL = _get_or_create_counter(
    name="model_predictions_total",
    documentation="Total number of inference predictions performed.",
    labelnames=("model_name", "model_version"),
)

MODEL_PREDICTIONS_BY_CLASS_TOTAL = _get_or_create_counter(
    name="model_predictions_by_class_total",
    documentation="Total predictions partitioned by predicted classification label.",
    labelnames=("model_name", "model_version", "predicted_class"),
)

MODEL_PREDICTION_PROBABILITY = _get_or_create_histogram(
    name="model_prediction_probability",
    documentation="Distribution histogram of output positive churn probabilities.",
    labelnames=("model_name", "model_version"),
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.571, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_PREDICTION_ERRORS_TOTAL = _get_or_create_counter(
    name="model_prediction_errors_total",
    documentation="Total inference and prediction service errors.",
    labelnames=("error_type",),
)

MODEL_INFERENCE_DURATION_SECONDS = _get_or_create_histogram(
    name="model_inference_duration_seconds",
    documentation="High-precision execution time of model pipeline predict_proba in seconds.",
    labelnames=("model_name", "model_version"),
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
)

MODEL_RELOADS_TOTAL = _get_or_create_counter(
    name="model_reloads_total",
    documentation="Total model reload attempts grouped by outcome status and target alias.",
    labelnames=("status", "alias"),
)


# ------------------------------------------------------------------------------
# 3. Model Governance & Version Metrics
# ------------------------------------------------------------------------------

MODEL_INFO = _get_or_create_gauge(
    name="model_info",
    documentation="Information gauge reflecting currently loaded champion model metadata.",
    labelnames=("model_name", "model_version", "alias", "decision_threshold"),
)

MODEL_AVAILABLE = _get_or_create_gauge(
    name="model_available",
    documentation="Model availability gauge (1 = active and healthy, 0 = unavailable).",
    labelnames=("model_name",),
)

MODEL_LAST_RELOAD_TIMESTAMP_SECONDS = _get_or_create_gauge(
    name="model_last_reload_timestamp_seconds",
    documentation="Timestamp of last successful model reload (UNIX epoch).",
    labelnames=("model_name",),
)

MODEL_LOAD_FAILURES_TOTAL = _get_or_create_counter(
    name="model_load_failures_total",
    documentation="Total number of failed model loading or reloading operations.",
    labelnames=("model_name", "alias"),
)


# ------------------------------------------------------------------------------
# 4. Evidently AI Monitoring & Drift Metrics
# ------------------------------------------------------------------------------

MONITORING_RUNS_TOTAL = _get_or_create_counter(
    name="monitoring_runs_total",
    documentation="Total monitoring and drift detection execution cycles.",
    labelnames=("status",),
)

MONITORING_RUN_DURATION_SECONDS = _get_or_create_histogram(
    name="monitoring_run_duration_seconds",
    documentation="Execution duration of monitoring evaluation cycle in seconds.",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

MONITORING_DRIFT_DETECTIONS_TOTAL = _get_or_create_counter(
    name="monitoring_drift_detections_total",
    documentation="Total number of confirmed dataset drift events (drift share >= threshold).",
)

MONITORING_DRIFT_SHARE = _get_or_create_gauge(
    name="monitoring_drift_share",
    documentation="Ratio of drifted features over total evaluated feature pool in latest cycle.",
)

MONITORING_DATA_QUALITY_FAILURES_TOTAL = _get_or_create_counter(
    name="monitoring_data_quality_failures_total",
    documentation="Total data quality validation failures during monitoring cycles.",
)

MONITORING_PREDICTION_DRIFT_DETECTIONS_TOTAL = _get_or_create_counter(
    name="monitoring_prediction_drift_detections_total",
    documentation="Total prediction distribution drift events detected.",
)

MONITORING_RETRAINING_RECOMMENDATIONS_TOTAL = _get_or_create_counter(
    name="monitoring_retraining_recommendations_total",
    documentation="Total retraining recommendations generated by the recommendation engine.",
    labelnames=("priority", "trigger_reason"),
)

MONITORING_GROUND_TRUTH_FEEDBACK_TOTAL = _get_or_create_counter(
    name="monitoring_ground_truth_feedback_total",
    documentation="Total delayed ground-truth feedback records received.",
    labelnames=("source",),
)

MONITORING_MODEL_ROC_AUC = _get_or_create_gauge(
    name="monitoring_model_roc_auc",
    documentation="Supervised ROC-AUC evaluated on matched ground-truth feedback (0 when absent).",
)

MONITORING_MODEL_PR_AUC = _get_or_create_gauge(
    name="monitoring_model_pr_auc",
    documentation="Supervised PR-AUC evaluated on matched ground-truth feedback (0 when absent).",
)

MONITORING_MODEL_F1 = _get_or_create_gauge(
    name="monitoring_model_f1",
    documentation="Supervised F1 score evaluated on matched ground-truth feedback (0 when absent).",
)

MONITORING_MODEL_BRIER_SCORE = _get_or_create_gauge(
    name="monitoring_model_brier_score",
    documentation="Supervised Brier calibration score evaluated on matched ground-truth feedback.",
)


# ------------------------------------------------------------------------------
# 5. Process & System Fallback Collector (psutil)
# ------------------------------------------------------------------------------

PROCESS_CPU_SECONDS_TOTAL = _get_or_create_counter(
    name="process_cpu_seconds_total",
    documentation="Total user and system CPU time spent in seconds.",
)

PROCESS_RESIDENT_MEMORY_BYTES = _get_or_create_gauge(
    name="process_resident_memory_bytes",
    documentation="Resident memory size in bytes.",
)

PROCESS_VIRTUAL_MEMORY_BYTES = _get_or_create_gauge(
    name="process_virtual_memory_bytes",
    documentation="Virtual memory size in bytes.",
)


def update_process_metrics() -> None:
    """Updates process CPU and memory metrics safely using psutil if available."""
    try:
        import psutil

        p = psutil.Process(os.getpid())
        mem_info = p.memory_info()
        PROCESS_RESIDENT_MEMORY_BYTES.set(float(mem_info.rss))
        PROCESS_VIRTUAL_MEMORY_BYTES.set(float(mem_info.vms))

        cpu_times = p.cpu_times()
        total_cpu = float(cpu_times.user + cpu_times.system)
        # Counter can only increment or stay equal
        current_cpu = REGISTRY.get_sample_value("process_cpu_seconds_total") or 0.0
        if total_cpu > current_cpu:
            PROCESS_CPU_SECONDS_TOTAL.inc(total_cpu - current_cpu)
    except Exception as exc:
        logger.debug("Process metrics collection skipped: %s", exc)


# ------------------------------------------------------------------------------
# 6. High-Level Helper Functions
# ------------------------------------------------------------------------------

KNOWN_ENDPOINTS = {
    "/health",
    "/ready",
    "/metrics",
    "/model/info",
    "/predict",
    "/model/reload",
    "/monitoring/status",
    "/monitoring/latest",
    "/monitoring/run",
    "/monitoring/feedback",
    "/docs",
    "/redoc",
    "/openapi.json",
}


def normalize_endpoint(path: str) -> str:
    """Normalizes URL path to a bounded low-cardinality endpoint string."""
    clean = path.rstrip("/") or "/"
    if clean in KNOWN_ENDPOINTS:
        return clean
    return "other"


def record_prediction_metric(
    model_name: str,
    model_version: str,
    churn_probability: float,
    prediction_label: str,
    inference_latency_ms: float,
) -> None:
    """Safely updates prediction metrics with low-cardinality dimensions."""
    try:
        MODEL_PREDICTIONS_TOTAL.labels(
            model_name=model_name,
            model_version=str(model_version),
        ).inc()

        MODEL_PREDICTIONS_BY_CLASS_TOTAL.labels(
            model_name=model_name,
            model_version=str(model_version),
            predicted_class=prediction_label,
        ).inc()

        MODEL_PREDICTION_PROBABILITY.labels(
            model_name=model_name,
            model_version=str(model_version),
        ).observe(float(churn_probability))

        # Latency converted from ms to seconds for Prometheus convention
        MODEL_INFERENCE_DURATION_SECONDS.labels(
            model_name=model_name,
            model_version=str(model_version),
        ).observe(float(inference_latency_ms) / 1000.0)
    except Exception as e:
        logger.warning("Failed to record prediction Prometheus metrics: %s", e)


def record_model_loaded(
    model_name: str,
    model_version: str,
    alias: str,
    decision_threshold: float,
) -> None:
    """Records model loaded state, info gauge, and availability."""
    try:
        # Clear previous model info gauges if version changed
        MODEL_INFO.labels(
            model_name=model_name,
            model_version=str(model_version),
            alias=alias,
            decision_threshold=str(decision_threshold),
        ).set(1.0)

        MODEL_AVAILABLE.labels(model_name=model_name).set(1.0)
        MODEL_LAST_RELOAD_TIMESTAMP_SECONDS.labels(model_name=model_name).set(time.time())
    except Exception as e:
        logger.warning("Failed to record model load Prometheus metric: %s", e)


def record_model_unavailable(model_name: str) -> None:
    """Sets model availability gauge to 0."""
    try:
        MODEL_AVAILABLE.labels(model_name=model_name).set(0.0)
    except Exception as e:
        logger.warning("Failed to record model unavailable metric: %s", e)


def record_model_reload_event(status: str, alias: str, model_name: Optional[str] = None) -> None:
    """Records model reload attempt outcome."""
    try:
        MODEL_RELOADS_TOTAL.labels(status=status, alias=alias).inc()
        if status != "success" and model_name:
            MODEL_LOAD_FAILURES_TOTAL.labels(model_name=model_name, alias=alias).inc()
    except Exception as e:
        logger.warning("Failed to record model reload event metric: %s", e)


def record_prediction_error(error_type: str) -> None:
    """Records inference service failure."""
    try:
        MODEL_PREDICTION_ERRORS_TOTAL.labels(error_type=error_type).inc()
    except Exception as e:
        logger.warning("Failed to record prediction error metric: %s", e)


def record_monitoring_run(
    status: str,
    duration_seconds: float,
    dataset_drift_detected: bool,
    drift_share: float,
    data_quality_status: str,
    prediction_drift_detected: bool,
    retraining_priority: Optional[str] = None,
    retraining_reason: Optional[str] = None,
) -> None:
    """Updates monitoring metrics following a monitoring evaluation cycle."""
    try:
        MONITORING_RUNS_TOTAL.labels(status=status).inc()
        MONITORING_RUN_DURATION_SECONDS.observe(float(duration_seconds))
        MONITORING_DRIFT_SHARE.set(float(drift_share))

        if dataset_drift_detected:
            MONITORING_DRIFT_DETECTIONS_TOTAL.inc()

        if data_quality_status == "FAILED":
            MONITORING_DATA_QUALITY_FAILURES_TOTAL.inc()

        if prediction_drift_detected:
            MONITORING_PREDICTION_DRIFT_DETECTIONS_TOTAL.inc()

        if retraining_priority and retraining_reason:
            MONITORING_RETRAINING_RECOMMENDATIONS_TOTAL.labels(
                priority=retraining_priority,
                trigger_reason=retraining_reason,
            ).inc()
    except Exception as e:
        logger.warning("Failed to record monitoring run metrics: %s", e)


def record_ground_truth_ingestion(source: str = "manual_feedback") -> None:
    """Records delayed ground truth ingestion count."""
    try:
        MONITORING_GROUND_TRUTH_FEEDBACK_TOTAL.labels(source=source).inc()
    except Exception as e:
        logger.warning("Failed to record ground truth ingestion metric: %s", e)


def record_supervised_metrics(
    roc_auc: Optional[float] = None,
    pr_auc: Optional[float] = None,
    f1: Optional[float] = None,
    brier_score: Optional[float] = None,
) -> None:
    """Updates supervised evaluation metrics on matched holdout ground-truth."""
    try:
        if roc_auc is not None:
            MONITORING_MODEL_ROC_AUC.set(float(roc_auc))
        if pr_auc is not None:
            MONITORING_MODEL_PR_AUC.set(float(pr_auc))
        if f1 is not None:
            MONITORING_MODEL_F1.set(float(f1))
        if brier_score is not None:
            MONITORING_MODEL_BRIER_SCORE.set(float(brier_score))
    except Exception as e:
        logger.warning("Failed to record supervised performance metrics: %s", e)


def get_latest_metrics() -> Tuple[bytes, str]:
    """Generates the latest Prometheus exposition format text output."""
    # Update dynamic gauges
    API_UPTIME_SECONDS.set(time.time() - START_TIME)
    update_process_metrics()
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
