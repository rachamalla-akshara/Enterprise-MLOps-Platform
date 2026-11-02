"""Integration Tests for FastAPI Serving Engine with Real MLflow Registry and Model Artifacts."""

from pathlib import Path
import time
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from src.common.config import AppConfig, find_project_root, get_config
from src.serving.app import create_app
from src.serving.model_loader import ModelLoader
from src.serving.prediction_logger import PredictionLogger


@pytest.fixture(scope="module")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="module")
def real_sample_record(project_root: Path) -> dict:
    """Loads first real customer record from data/samples/sample_churn_input.csv."""
    sample_file = project_root / "data" / "samples" / "sample_churn_input.csv"
    assert sample_file.exists(), f"Sample file not found at: {sample_file}"
    df = pd.read_csv(sample_file)
    first_row = df.iloc[0].to_dict()
    # Ensure NaN in numeric fields is None
    if "TotalCharges" in first_row and pd.isna(first_row["TotalCharges"]):
        first_row["TotalCharges"] = None
    return first_row


@pytest.fixture(scope="module")
def integration_app(tmp_path_factory):
    """Creates a full FastAPI app wired to the real MLflow SQLite registry and temporary prediction DB."""
    temp_dir = tmp_path_factory.mktemp("serving_integration")
    temp_db = temp_dir / "integration_predictions.db"

    cfg = get_config()
    test_cfg = cfg.model_copy(deep=True)
    test_cfg.serving.prediction_db_uri = f"sqlite:///{temp_db.as_posix()}"
    test_cfg.serving.reload_token = "integration-test-admin-token"

    loader = ModelLoader(config=test_cfg)
    logger_db = PredictionLogger(db_uri=f"sqlite:///{temp_db.as_posix()}", config=test_cfg)

    app = create_app(
        config=test_cfg,
        model_loader=loader,
        prediction_logger=logger_db,
    )
    return app, loader, logger_db, test_cfg


def test_real_model_loading_and_readiness(integration_app):
    """Verifies that ModelLoader resolves real @champion alias (Version 4, XGBoost_Tuned)."""
    app, loader, _, _ = integration_app

    with TestClient(app) as client:
        # Check /health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # Check /ready
        ready_resp = client.get("/ready")
        assert ready_resp.status_code == 200
        ready_data = ready_resp.json()
        assert ready_data["status"] == "ready"
        assert ready_data["model_loaded"] is True
        assert ready_data["model_name"] == "customer-churn-classifier"
        assert ready_data["model_version"] == "4"
        assert ready_data["model_alias"] == "champion"

        # Check /model/info
        info_resp = client.get("/model/info")
        assert info_resp.status_code == 200
        info_data = info_resp.json()
        assert info_data["resolved_version"] == "4"
        assert info_data["alias"] == "champion"
        assert info_data["decision_threshold"] == 0.571


def test_real_customer_prediction(integration_app, real_sample_record):
    """Verifies end-to-end prediction against genuine registered XGBoost champion."""
    app, _, logger_db, _ = integration_app

    with TestClient(app) as client:
        resp = client.post("/predict", json=real_sample_record)
        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        assert "request_id" in data
        assert data["model_name"] == "customer-churn-classifier"
        assert data["model_version"] == "4"
        assert data["decision_threshold"] == 0.571
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["prediction"] in ["Churn", "No Churn"]

        # Validate decision cutoff consistency
        expected_class = "Churn" if data["churn_probability"] >= 0.571 else "No Churn"
        assert data["prediction"] == expected_class
        assert data["inference_latency_ms"] > 0.0

        # Verify telemetry in SQLite
        logs = logger_db.get_recent_logs(limit=5)
        assert len(logs) >= 1
        matched_log = next((l for l in logs if l["request_id"] == data["request_id"]), None)
        assert matched_log is not None
        assert matched_log["prediction"] == data["prediction"]
        assert matched_log["model_version"] == "4"
        assert matched_log["request_status"] == "SUCCESS"


def test_real_model_reload_and_dynamic_alias_switching(integration_app):
    """Verifies that reloading alias switches between registered versions dynamically."""
    app, loader, _, test_cfg = integration_app

    # Determine actual version assigned to challenger in registry
    challenger_mv = loader.client.get_model_version_by_alias(loader.model_name, "challenger")
    expected_challenger_ver = str(challenger_mv.version)

    with TestClient(app) as client:
        # Reload to @challenger
        reload_resp = client.post(
            "/model/reload",
            headers={"X-Admin-API-Key": test_cfg.serving.reload_token},
            json={"alias": "challenger"},
        )
        assert reload_resp.status_code == 200
        reload_data = reload_resp.json()
        assert reload_data["status"] == "reloaded"
        assert reload_data["new_version"] == expected_challenger_ver

        # Verify /model/info now reflects challenger version
        info_resp = client.get("/model/info")
        assert info_resp.status_code == 200
        assert info_resp.json()["resolved_version"] == expected_challenger_ver
        assert info_resp.json()["alias"] == "challenger"

        # Reload back to @champion (Version 4, XGBoost_Tuned)
        restore_resp = client.post(
            "/model/reload",
            headers={"X-Admin-API-Key": test_cfg.serving.reload_token},
            json={"alias": "champion"},
        )
        assert restore_resp.status_code == 200
        assert restore_resp.json()["new_version"] == "4"

        # Confirm restored champion
        restored_info = client.get("/model/info").json()
        assert restored_info["resolved_version"] == "4"
        assert restored_info["alias"] == "champion"


def test_actual_inference_latency_benchmarking(integration_app, real_sample_record):
    """Benchmarks actual end-to-end inference latency over 30 requests."""
    app, _, _, _ = integration_app

    with TestClient(app) as client:
        # Warmup
        for _ in range(5):
            client.post("/predict", json=real_sample_record)

        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=real_sample_record)
            t1 = time.perf_counter()
            assert resp.status_code == 200
            latencies.append((t1 - t0) * 1000.0)

        mean_lat = float(np.mean(latencies))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))

        print(f"\n[BENCHMARK] Actual Serving API Latency: Mean={mean_lat:.2f} ms | p50={p50_lat:.2f} ms | p95={p95_lat:.2f} ms")
        assert p95_lat < 150.0, f"Serving latency exceeded acceptable ceiling: p95={p95_lat:.2f} ms"
