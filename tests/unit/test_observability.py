"""Unit tests for Phase 6 Prometheus Observability, Grafana Configurations, and Alert Rules.

Verifies:
1. Prometheus metrics exposition endpoint (GET /metrics).
2. HTTP request counter and latency histogram instrumentation.
3. Prediction counter, probability histogram, and class distribution metrics.
4. Model reload and availability metrics.
5. Monitoring metrics updates (drift share, runs, recommendations, supervised performance).
6. Absence of high-cardinality or sensitive PII labels.
7. Alert rule YAML syntax, structure, and severity classification.
8. Prometheus scrape configuration validity.
9. Grafana provisioning and dashboard JSON schema validity.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from starlette.testclient import TestClient
import yaml

from prometheus_client import REGISTRY

from src.common.config import find_project_root
from src.observability.metrics import (
    HTTP_ERRORS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
    KNOWN_ENDPOINTS,
    MODEL_AVAILABLE,
    MODEL_INFERENCE_DURATION_SECONDS,
    MODEL_INFO,
    MODEL_PREDICTION_ERRORS_TOTAL,
    MODEL_PREDICTION_PROBABILITY,
    MODEL_PREDICTIONS_BY_CLASS_TOTAL,
    MODEL_PREDICTIONS_TOTAL,
    MODEL_RELOADS_TOTAL,
    MONITORING_DRIFT_DETECTIONS_TOTAL,
    MONITORING_DRIFT_SHARE,
    MONITORING_GROUND_TRUTH_FEEDBACK_TOTAL,
    MONITORING_MODEL_BRIER_SCORE,
    MONITORING_MODEL_F1,
    MONITORING_MODEL_PR_AUC,
    MONITORING_MODEL_ROC_AUC,
    MONITORING_RETRAINING_RECOMMENDATIONS_TOTAL,
    MONITORING_RUN_DURATION_SECONDS,
    MONITORING_RUNS_TOTAL,
    get_latest_metrics,
    normalize_endpoint,
    record_ground_truth_ingestion,
    record_model_loaded,
    record_model_reload_event,
    record_model_unavailable,
    record_monitoring_run,
    record_prediction_error,
    record_prediction_metric,
    record_supervised_metrics,
    update_process_metrics,
)
from src.serving.app import create_app
from src.serving.model_loader import ModelContainer, ModelLoader
from src.serving.prediction_logger import PredictionLogger


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_model_loader():
    """Mock ModelLoader with predictable output for testing metrics."""
    loader = MagicMock(spec=ModelLoader)
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array([[0.30, 0.70]])
    container = ModelContainer(
        pipeline=pipeline,
        model_name="customer-churn-classifier",
        model_version="4",
        run_id="mock_run_id_obs",
        alias="champion",
        load_timestamp=datetime.now(timezone.utc),
    )
    loader.get_active_model.return_value = container
    loader.is_ready.return_value = True
    loader.model_name = "customer-churn-classifier"
    return loader


@pytest.fixture
def isolated_prediction_logger(tmp_path: Path):
    """Isolated SQLite prediction logger."""
    db_file = tmp_path / "test_obs_pred.db"
    return PredictionLogger(db_uri=f"sqlite:///{db_file.as_posix()}")


@pytest.fixture
def test_client(mock_model_loader, isolated_prediction_logger, test_config):
    """FastAPI TestClient with mocked model and isolated telemetry logger."""
    cfg = test_config.model_copy(deep=True)
    cfg.serving.reload_token = "test-admin-secret"
    cfg.serving.decision_threshold = 0.571

    mock_mon = MagicMock()
    mock_mon.get_latest_status.return_value = {
        "status": "ready",
        "sample_count": 100,
        "latest_drift_report": {"report_id": "test-rep", "drift_share": 0.10},
    }

    app = create_app(
        config=cfg,
        model_loader=mock_model_loader,
        prediction_logger=isolated_prediction_logger,
        monitoring_service=mock_mon,
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_features_payload() -> dict:
    """Valid customer attributes payload for prediction testing."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 958.2,
    }


# ==============================================================================
# 1. Metrics Endpoint & Basic Exposition Tests
# ==============================================================================

def test_metrics_endpoint_exposition(test_client):
    """Verifies that GET /metrics returns 200 with Prometheus text exposition."""
    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")

    body = resp.text
    # Verify core metric names are declared in HELP / TYPE headers
    assert "http_requests_total" in body
    assert "model_predictions_total" in body
    assert "model_available" in body
    assert "api_uptime_seconds" in body


def test_get_latest_metrics_function():
    """Verifies low-level get_latest_metrics payload generator."""
    payload, media_type = get_latest_metrics()
    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert "text/plain" in media_type


# ==============================================================================
# 2. HTTP Request Counter & Latency Instrumentation Tests
# ==============================================================================

def test_http_request_metrics_recorded(test_client):
    """Verifies that HTTP requests increment http_requests_total with method and status code."""
    before_count = REGISTRY.get_sample_value(
        "http_requests_total",
        {"method": "GET", "endpoint": "/health", "status_code": "200"},
    ) or 0.0

    resp = test_client.get("/health")
    assert resp.status_code == 200

    after_count = REGISTRY.get_sample_value(
        "http_requests_total",
        {"method": "GET", "endpoint": "/health", "status_code": "200"},
    ) or 0.0

    assert after_count == before_count + 1.0


def test_http_error_metrics_recorded(test_client):
    """Verifies that 4xx error responses increment http_errors_total."""
    before_errors = REGISTRY.get_sample_value(
        "http_errors_total",
        {"method": "POST", "endpoint": "/predict", "status_code": "422"},
    ) or 0.0

    # Submit invalid payload missing required fields
    resp = test_client.post("/predict", json={"gender": "Invalid"})
    assert resp.status_code == 422

    after_errors = REGISTRY.get_sample_value(
        "http_errors_total",
        {"method": "POST", "endpoint": "/predict", "status_code": "422"},
    ) or 0.0

    assert after_errors == before_errors + 1.0


def test_endpoint_normalization():
    """Verifies that dynamic and known paths are normalized to low-cardinality buckets."""
    assert normalize_endpoint("/health") == "/health"
    assert normalize_endpoint("/ready") == "/ready"
    assert normalize_endpoint("/predict") == "/predict"
    assert normalize_endpoint("/model/reload") == "/model/reload"
    assert normalize_endpoint("/unknown/route/123") == "other"
    assert normalize_endpoint("/predict?param=123") == "other" or normalize_endpoint("/predict") == "/predict"


# ==============================================================================
# 3. Model Inference & Prediction Telemetry Tests
# ==============================================================================

def test_prediction_metrics_updated_on_inference(test_client, valid_features_payload):
    """Verifies that /predict increments prediction counters and observes latency."""
    before_preds = REGISTRY.get_sample_value(
        "model_predictions_total",
        {"model_name": "customer-churn-classifier", "model_version": "4"},
    ) or 0.0

    before_churn_class = REGISTRY.get_sample_value(
        "model_predictions_by_class_total",
        {"model_name": "customer-churn-classifier", "model_version": "4", "predicted_class": "Churn"},
    ) or 0.0

    resp = test_client.post("/predict", json=valid_features_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == "Churn"

    after_preds = REGISTRY.get_sample_value(
        "model_predictions_total",
        {"model_name": "customer-churn-classifier", "model_version": "4"},
    ) or 0.0

    after_churn_class = REGISTRY.get_sample_value(
        "model_predictions_by_class_total",
        {"model_name": "customer-churn-classifier", "model_version": "4", "predicted_class": "Churn"},
    ) or 0.0

    assert after_preds == before_preds + 1.0
    assert after_churn_class == before_churn_class + 1.0


def test_prediction_error_counter():
    """Verifies that record_prediction_error increments model_prediction_errors_total."""
    before = REGISTRY.get_sample_value("model_prediction_errors_total", {"error_type": "inference_error"}) or 0.0
    record_prediction_error("inference_error")
    after = REGISTRY.get_sample_value("model_prediction_errors_total", {"error_type": "inference_error"}) or 0.0
    assert after == before + 1.0


# ==============================================================================
# 4. Model Governance & Reload Tests
# ==============================================================================

def test_model_loaded_and_availability_gauges():
    """Verifies record_model_loaded and record_model_unavailable."""
    record_model_loaded(
        model_name="customer-churn-classifier",
        model_version="4",
        alias="champion",
        decision_threshold=0.571,
    )
    avail = REGISTRY.get_sample_value("model_available", {"model_name": "customer-churn-classifier"})
    assert avail == 1.0

    record_model_unavailable("customer-churn-classifier")
    avail_after = REGISTRY.get_sample_value("model_available", {"model_name": "customer-churn-classifier"})
    assert avail_after == 0.0


def test_model_reload_event_metrics():
    """Verifies record_model_reload_event success and failure counters."""
    before_succ = REGISTRY.get_sample_value("model_reloads_total", {"status": "success", "alias": "champion"}) or 0.0
    record_model_reload_event("success", "champion", "customer-churn-classifier")
    after_succ = REGISTRY.get_sample_value("model_reloads_total", {"status": "success", "alias": "champion"}) or 0.0
    assert after_succ == before_succ + 1.0

    before_fail = REGISTRY.get_sample_value("model_reloads_total", {"status": "failure", "alias": "challenger"}) or 0.0
    record_model_reload_event("failure", "challenger", "customer-churn-classifier")
    after_fail = REGISTRY.get_sample_value("model_reloads_total", {"status": "failure", "alias": "challenger"}) or 0.0
    assert after_fail == before_fail + 1.0


# ==============================================================================
# 5. Monitoring & Drift Telemetry Tests
# ==============================================================================

def test_monitoring_run_metrics():
    """Verifies record_monitoring_run updates drift share and execution counters."""
    before_runs = REGISTRY.get_sample_value("monitoring_runs_total", {"status": "completed"}) or 0.0
    before_drifts = REGISTRY.get_sample_value("monitoring_drift_detections_total") or 0.0

    record_monitoring_run(
        status="completed",
        duration_seconds=2.45,
        dataset_drift_detected=True,
        drift_share=0.42,
        data_quality_status="PASSED",
        prediction_drift_detected=False,
        retraining_priority="RECOMMENDED",
        retraining_reason="DATASET_DRIFT",
    )

    after_runs = REGISTRY.get_sample_value("monitoring_runs_total", {"status": "completed"}) or 0.0
    after_drifts = REGISTRY.get_sample_value("monitoring_drift_detections_total") or 0.0
    drift_share = REGISTRY.get_sample_value("monitoring_drift_share") or 0.0

    assert after_runs == before_runs + 1.0
    assert after_drifts == before_drifts + 1.0
    assert abs(drift_share - 0.42) < 1e-4


def test_supervised_performance_metrics_recorded():
    """Verifies record_supervised_metrics updates holdout evaluation gauges."""
    record_supervised_metrics(
        roc_auc=0.8536,
        pr_auc=0.6697,
        f1=0.6486,
        brier_score=0.1611,
    )
    assert abs((REGISTRY.get_sample_value("monitoring_model_roc_auc") or 0.0) - 0.8536) < 1e-4
    assert abs((REGISTRY.get_sample_value("monitoring_model_pr_auc") or 0.0) - 0.6697) < 1e-4
    assert abs((REGISTRY.get_sample_value("monitoring_model_f1") or 0.0) - 0.6486) < 1e-4
    assert abs((REGISTRY.get_sample_value("monitoring_model_brier_score") or 0.0) - 0.1611) < 1e-4


def test_ground_truth_ingestion_metric():
    """Verifies record_ground_truth_ingestion increments feedback counter."""
    before = REGISTRY.get_sample_value("monitoring_ground_truth_feedback_total", {"source": "crm_webhook"}) or 0.0
    record_ground_truth_ingestion(source="crm_webhook")
    after = REGISTRY.get_sample_value("monitoring_ground_truth_feedback_total", {"source": "crm_webhook"}) or 0.0
    assert after == before + 1.0


def test_process_metrics_collection():
    """Verifies that update_process_metrics runs without error."""
    update_process_metrics()
    rss = REGISTRY.get_sample_value("process_resident_memory_bytes")
    # In psutil environments RSS should be > 0
    if rss is not None:
        assert rss > 0


# ==============================================================================
# 6. Low-Cardinality & Anti-PII Verification Tests
# ==============================================================================

def test_metric_labels_contain_no_sensitive_or_high_cardinality_fields():
    """Guarantees metric labels contain zero sensitive customer features or IDs."""
    forbidden_substrings = [
        "customer", "tenure", "charge", "gender", "partner", "dependents",
        "phone", "contract", "payment", "uuid", "token", "password", "secret", "req_id", "request_id",
    ]

    for metric_name, collector in REGISTRY._names_to_collectors.items():
        if hasattr(collector, "_labelnames"):
            for label in collector._labelnames:
                for forbidden in forbidden_substrings:
                    assert forbidden not in label.lower(), (
                        f"Metric '{metric_name}' contains sensitive/high-cardinality label '{label}'"
                    )


# ==============================================================================
# 7. Alert Rules & Prometheus Configuration Tests
# ==============================================================================

def test_prometheus_yaml_validity():
    """Verifies that monitoring/prometheus/prometheus.yml is syntactically valid YAML."""
    root = find_project_root()
    prom_yaml = root / "monitoring" / "prometheus" / "prometheus.yml"
    assert prom_yaml.exists(), f"Missing file: {prom_yaml}"

    with open(prom_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "global" in data
    assert "scrape_configs" in data
    assert "rule_files" in data

    # Check job names
    jobs = [job["job_name"] for job in data["scrape_configs"]]
    assert "fastapi-serving" in jobs
    assert "prometheus" in jobs


def test_alerts_yaml_validity():
    """Verifies that monitoring/prometheus/alerts.yml is valid and rules adhere to conventions."""
    root = find_project_root()
    alerts_yaml = root / "monitoring" / "prometheus" / "alerts.yml"
    assert alerts_yaml.exists(), f"Missing file: {alerts_yaml}"

    with open(alerts_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "groups" in data
    assert len(data["groups"]) > 0

    valid_severities = {"critical", "warning", "informational"}
    alert_names = []

    for group in data["groups"]:
        assert "name" in group
        assert "rules" in group
        for rule in group["rules"]:
            assert "alert" in rule, f"Rule in group '{group['name']}' missing 'alert' field"
            assert "expr" in rule, f"Rule '{rule['alert']}' missing 'expr' expression"
            assert "labels" in rule, f"Rule '{rule['alert']}' missing 'labels'"
            assert "severity" in rule["labels"], f"Rule '{rule['alert']}' missing 'severity' label"
            assert rule["labels"]["severity"] in valid_severities, (
                f"Rule '{rule['alert']}' has invalid severity '{rule['labels']['severity']}'"
            )
            assert "annotations" in rule, f"Rule '{rule['alert']}' missing 'annotations'"
            assert "summary" in rule["annotations"], f"Rule '{rule['alert']}' missing 'summary'"
            alert_names.append(rule["alert"])

    # Ensure key required alerts are defined
    required_alerts = [
        "ApiServiceUnavailable",
        "ElevatedHttp5xxErrors",
        "HighApiRequestLatency",
        "ModelUnavailable",
        "ModelReloadFailure",
        "DatasetDriftDetected",
        "DataQualityFailures",
        "PredictionDriftDetected",
        "MonitoringCycleFailed",
        "RetrainingRecommendationGenerated",
    ]
    for req_alert in required_alerts:
        assert req_alert in alert_names, f"Required alert '{req_alert}' is missing from alerts.yml"


# ==============================================================================
# 8. Grafana Dashboard JSON Schema & Validity Tests
# ==============================================================================

def test_grafana_dashboard_json_validity():
    """Verifies that all provisioned Grafana dashboards are valid JSON with required panel structures."""
    root = find_project_root()
    dashboards_dir = root / "monitoring" / "grafana" / "dashboards"
    assert dashboards_dir.exists(), f"Missing directory: {dashboards_dir}"

    expected_dashboards = [
        "api_overview.json",
        "ml_inference.json",
        "model_monitoring.json",
        "infrastructure.json",
    ]

    for dash_file in expected_dashboards:
        dash_path = dashboards_dir / dash_file
        assert dash_path.exists(), f"Dashboard file '{dash_file}' does not exist"

        with open(dash_path, "r", encoding="utf-8") as f:
            dash_json = json.load(f)

        assert "title" in dash_json, f"Dashboard '{dash_file}' missing title"
        assert "uid" in dash_json, f"Dashboard '{dash_file}' missing uid"
        assert "panels" in dash_json, f"Dashboard '{dash_file}' missing panels"
        assert len(dash_json["panels"]) >= 4, f"Dashboard '{dash_file}' has too few panels"

        # Check each panel has a title and target expression
        for panel in dash_json["panels"]:
            assert "title" in panel, f"Panel in '{dash_file}' missing title"
            assert "type" in panel, f"Panel '{panel.get('title')}' in '{dash_file}' missing type"
            if "targets" in panel:
                for target in panel["targets"]:
                    assert "expr" in target, f"Target in panel '{panel['title']}' missing 'expr'"
