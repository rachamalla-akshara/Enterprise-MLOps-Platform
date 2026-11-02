"""Unit Tests for FastAPI Model Serving Engine and Inference Components."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.common.config import AppConfig, get_config
from src.serving.app import create_app
from src.serving.exceptions import ModelNotLoadedError, ModelReloadError
from src.serving.model_loader import ModelContainer, ModelLoader
from src.serving.prediction_logger import PredictionLogger
from src.serving.prediction_service import PredictionService
from src.serving.schemas import CustomerFeatures, PredictionResponse


@pytest.fixture
def sample_valid_features() -> dict:
    """Returns a valid customer features dictionary without Churn."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
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
        "TotalCharges": 1912.4,
    }


@pytest.fixture
def mock_pipeline():
    """Creates a mock Scikit-learn Pipeline with predictable predict_proba."""
    pipe = MagicMock()
    # Default returns 70% probability for class 1 (Churn)
    pipe.predict_proba.return_value = np.array([[0.30, 0.70]])
    return pipe


@pytest.fixture
def mock_model_loader(mock_pipeline):
    """Creates a mock ModelLoader with a loaded ModelContainer."""
    loader = MagicMock(spec=ModelLoader)
    container = ModelContainer(
        pipeline=mock_pipeline,
        model_name="customer-churn-classifier",
        model_version="4",
        run_id="mock_run_id_12345",
        alias="champion",
        load_timestamp=datetime.now(timezone.utc),
    )
    loader.get_active_model.return_value = container
    loader.is_ready.return_value = True
    return loader


@pytest.fixture
def test_prediction_logger(tmp_path: Path):
    """Creates an isolated PredictionLogger with temporary SQLite DB."""
    temp_db = tmp_path / "test_predictions.db"
    return PredictionLogger(db_uri=f"sqlite:///{temp_db.as_posix()}")


@pytest.fixture
def test_app_client(mock_model_loader, test_prediction_logger, test_config):
    """Creates a TestClient backed by mocked model and isolated logger."""
    cfg = test_config.model_copy(deep=True)
    cfg.serving.reload_token = "test-secret-token"
    cfg.serving.decision_threshold = 0.571

    app = create_app(
        config=cfg,
        model_loader=mock_model_loader,
        prediction_logger=test_prediction_logger,
    )
    with TestClient(app) as client:
        yield client


# ==============================================================================
# 1. Health and Readiness Tests
# ==============================================================================
def test_health_endpoint(test_app_client):
    """Verifies that GET /health returns 200 with healthy status."""
    resp = test_app_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "X-Request-ID" in resp.headers


def test_readiness_endpoint_when_ready(test_app_client):
    """Verifies that GET /ready returns 200 when model is loaded."""
    resp = test_app_client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_version"] == "4"
    assert data["model_alias"] == "champion"


def test_readiness_endpoint_when_unready(mock_model_loader, test_prediction_logger, test_config):
    """Verifies that GET /ready returns 503 when model is not ready."""
    mock_model_loader.is_ready.return_value = False
    mock_model_loader.get_active_model.side_effect = ModelNotLoadedError("No model active")

    app = create_app(config=test_config, model_loader=mock_model_loader, prediction_logger=test_prediction_logger)
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False


def test_model_info_endpoint(test_app_client):
    """Verifies that GET /model/info returns model metadata."""
    resp = test_app_client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "customer-churn-classifier"
    assert data["resolved_version"] == "4"
    assert data["run_id"] == "mock_run_id_12345"
    assert data["alias"] == "champion"
    assert data["decision_threshold"] == 0.571
    assert data["status"] == "active"


# ==============================================================================
# 2. Input Validation Tests
# ==============================================================================
def test_predict_missing_feature(test_app_client, sample_valid_features):
    """Verifies that missing a required feature returns 422 Unprocessable Entity."""
    payload = sample_valid_features.copy()
    del payload["Contract"]  # Missing required field

    resp = test_app_client.post("/predict", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "InputValidationError"
    assert any("Contract" in str(err) for err in data["details"])


def test_predict_invalid_enum_value(test_app_client, sample_valid_features):
    """Verifies that invalid categorical values return 422."""
    payload = sample_valid_features.copy()
    payload["InternetService"] = "Satellite"  # Invalid enum value

    resp = test_app_client.post("/predict", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert any("InternetService" in str(err) for err in data["details"])


def test_predict_unexpected_extra_feature_forbidden(test_app_client, sample_valid_features):
    """Verifies that extra/unknown features are forbidden and return 422."""
    payload = sample_valid_features.copy()
    payload["unexpected_extra_field"] = "malicious_or_unknown_input"

    resp = test_app_client.post("/predict", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "extra_forbidden" in str(data["details"])


def test_predict_forbidden_churn_target_column(test_app_client, sample_valid_features):
    """Verifies that passing the target column 'Churn' is rejected with 422."""
    payload = sample_valid_features.copy()
    payload["Churn"] = "Yes"

    resp = test_app_client.post("/predict", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "extra_forbidden" in str(data["details"])


def test_predict_numeric_boundary_validation(test_app_client, sample_valid_features):
    """Verifies that tenure or charge out-of-bound inputs are rejected with 422."""
    payload = sample_valid_features.copy()
    payload["tenure"] = -5  # Negative tenure invalid

    resp = test_app_client.post("/predict", json=payload)
    assert resp.status_code == 422


# ==============================================================================
# 3. Prediction Logic & Threshold Application Tests
# ==============================================================================
def test_predict_churn_class_above_threshold(test_app_client, sample_valid_features, mock_pipeline):
    """Verifies that probability (0.70) >= threshold (0.571) yields 'Churn'."""
    mock_pipeline.predict_proba.return_value = np.array([[0.30, 0.70]])

    resp = test_app_client.post("/predict", json=sample_valid_features)
    assert resp.status_code == 200
    data = resp.json()

    assert data["prediction"] == "Churn"
    assert data["churn_probability"] == 0.70
    assert data["decision_threshold"] == 0.571
    assert data["model_name"] == "customer-churn-classifier"
    assert data["model_version"] == "4"
    assert "request_id" in data
    assert "inference_latency_ms" in data
    assert data["inference_latency_ms"] >= 0.0
    assert "prediction_timestamp" in data


def test_predict_no_churn_class_below_threshold(test_app_client, sample_valid_features, mock_pipeline):
    """Verifies that probability (0.45) < threshold (0.571) yields 'No Churn'."""
    mock_pipeline.predict_proba.return_value = np.array([[0.55, 0.45]])

    resp = test_app_client.post("/predict", json=sample_valid_features)
    assert resp.status_code == 200
    data = resp.json()

    assert data["prediction"] == "No Churn"
    assert data["churn_probability"] == 0.45
    assert data["decision_threshold"] == 0.571


def test_predict_wrapped_request_payload(test_app_client, sample_valid_features):
    """Verifies that wrapped PredictionRequest format is also supported."""
    wrapped_payload = {
        "customer_id": "CUST-9999",
        "features": sample_valid_features,
    }
    resp = test_app_client.post("/predict", json=wrapped_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in ["Churn", "No Churn"]


# ==============================================================================
# 4. Telemetry Logging Tests
# ==============================================================================
def test_prediction_telemetry_logging(test_app_client, sample_valid_features, test_prediction_logger):
    """Verifies that predictions are logged to the database without sensitive customer info."""
    resp = test_app_client.post("/predict", json=sample_valid_features)
    assert resp.status_code == 200
    req_id = resp.json()["request_id"]

    logs = test_prediction_logger.get_recent_logs(limit=10)
    assert len(logs) >= 1
    latest_log = logs[0]

    assert latest_log["request_id"] == req_id
    assert latest_log["model_name"] == "customer-churn-classifier"
    assert latest_log["model_version"] == "4"
    assert latest_log["prediction"] in ["Churn", "No Churn"]
    assert latest_log["request_status"] == "SUCCESS"
    assert latest_log["inference_latency_ms"] >= 0.0


def test_prediction_resilience_when_logging_fails(
    mock_model_loader,
    sample_valid_features,
    test_config,
    mock_pipeline,
):
    """Verifies that a database failure during logging does NOT break the HTTP response."""
    failing_logger = MagicMock(spec=PredictionLogger)
    failing_logger.log_prediction.side_effect = Exception("Disk I/O error or SQLite database locked")

    cfg = test_config.model_copy(deep=True)
    app = create_app(config=cfg, model_loader=mock_model_loader, prediction_logger=failing_logger)

    with TestClient(app) as client:
        resp = client.post("/predict", json=sample_valid_features)
        # Prediction response succeeds despite logger failure
        assert resp.status_code == 200
        assert resp.json()["prediction"] in ["Churn", "No Churn"]


# ==============================================================================
# 5. Protected Model Reload Tests
# ==============================================================================
def test_reload_unauthorized_without_token(test_app_client):
    """Verifies that POST /model/reload without token returns 403 Forbidden."""
    resp = test_app_client.post("/model/reload", json={})
    assert resp.status_code == 403


def test_reload_unauthorized_with_wrong_token(test_app_client):
    """Verifies that POST /model/reload with incorrect token returns 403 Forbidden."""
    resp = test_app_client.post(
        "/model/reload",
        headers={"X-Admin-API-Key": "wrong-secret-key"},
        json={},
    )
    assert resp.status_code == 403


def test_reload_authorized_success(test_app_client, mock_model_loader, mock_pipeline):
    """Verifies that POST /model/reload with valid token successfully swaps model."""
    old_c = mock_model_loader.get_active_model()
    new_c = ModelContainer(
        pipeline=mock_pipeline,
        model_name="customer-churn-classifier",
        model_version="5",
        run_id="new_run_id_67890",
        alias="champion",
        load_timestamp=datetime.now(timezone.utc),
    )
    mock_model_loader.reload.return_value = (old_c, new_c)

    resp = test_app_client.post(
        "/model/reload",
        headers={"X-Admin-API-Key": "test-secret-token"},
        json={"alias": "champion"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reloaded"
    assert data["previous_version"] == "4"
    assert data["new_version"] == "5"


def test_reload_failure_preserves_previous_model(test_app_client, mock_model_loader):
    """Verifies that if reloading fails, previous model is preserved and 400 is returned."""
    mock_model_loader.reload.side_effect = ModelReloadError(
        "Candidate model failed verification. Previous model version 4 retained.",
        details={"active_version": "4"},
    )

    resp = test_app_client.post(
        "/model/reload",
        headers={"X-Admin-API-Key": "test-secret-token"},
        json={"alias": "candidate_invalid"},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "ModelReloadError"
    assert "retained" in data["message"]
