"""Integration tests for end-to-end monitoring service, database persistence, and FastAPI routes."""

from datetime import datetime, timezone
import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from src.common.config import get_config
from src.monitoring.models import (
    DataQualityReportRecord,
    DriftReportRecord,
    FeatureDriftResultRecord,
    FeatureObservationRecord,
    GroundTruthFeedbackRecord,
    ModelPerformanceReportRecord,
    MonitoringBase,
    RetrainingRecommendationRecord,
    get_monitoring_engine,
)
from src.monitoring.monitoring_config import MonitoringConfig
from src.monitoring.monitoring_service import MonitoringService
from src.serving.app import create_app


@pytest.fixture
def test_db_uri(tmp_path):
    """Provides an isolated SQLite database URI for integration tests."""
    db_file = tmp_path / "integration_monitoring.db"
    return f"sqlite:///{db_file.as_posix()}"


@pytest.fixture
def monitoring_service(test_db_uri, tmp_path):
    """Initializes MonitoringService against test database and test reports directory."""
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    cfg = MonitoringConfig(
        reference_data_path="data/processed/reference_dataset.parquet",
        monitoring_db_uri=test_db_uri,
        reports_dir=str(rep_dir),
        min_sample_size=30,
        drift_share_threshold=0.30,
        significance_level=0.05,
        prediction_drift_threshold=0.10,
        cooldown_hours=24,
        buffer_max_size=50,
        buffer_retention_hours=24,
    )
    svc = MonitoringService(db_uri=test_db_uri, config=cfg)
    yield svc
    # Dispose engine cleanly
    svc.engine.dispose()


def test_full_monitoring_cycle_with_real_reference(monitoring_service):
    """Verifies end-to-end monitoring cycle against authentic IBM Telco reference data."""
    ref_df = monitoring_service.load_reference_dataset()
    assert len(ref_df) == 4929
    assert len(ref_df.columns) == 19

    # Ingest 60 observations (in-distribution subset)
    current_batch = ref_df.iloc[:60].copy()
    added = monitoring_service.ingest_feature_observations(current_batch)
    assert added == 60

    # Run monitoring cycle
    summary = monitoring_service.run_monitoring_cycle()
    assert summary.report_id is not None
    assert summary.data_quality_status == "PASSED"
    assert not summary.dataset_drift_detected
    assert summary.drift_share < monitoring_service.config.drift_share_threshold

    # Verify report files were created
    html_p = Path(summary.report_html_path)
    json_p = Path(summary.report_json_path)
    assert html_p.exists()
    assert html_p.stat().st_size > 1000
    assert json_p.exists()
    assert json_p.stat().st_size > 1000

    # Verify database persistence across all tables
    session = monitoring_service.SessionLocal()
    try:
        d_rec = session.query(DriftReportRecord).filter_by(report_id=summary.report_id).first()
        assert d_rec is not None
        assert d_rec.total_features == 19
        assert d_rec.drift_share < monitoring_service.config.drift_share_threshold
        assert not d_rec.dataset_drift_detected

        f_recs = session.query(FeatureDriftResultRecord).filter_by(report_id=summary.report_id).all()
        assert len(f_recs) == 19

        q_rec = session.query(DataQualityReportRecord).filter_by(report_id=summary.report_id).first()
        assert q_rec is not None
        assert q_rec.quality_status == "PASSED"

        r_rec = session.query(RetrainingRecommendationRecord).first()
        assert r_rec is not None
        assert r_rec.recommendation_status in ["NO_ACTION", "RECOMMENDED", "COOLDOWN_ACTIVE"]
    finally:
        session.close()


def test_buffer_fifo_and_ttl_purging(monitoring_service):
    """Verifies that the de-identified feature observation buffer enforces FIFO capacity limits."""
    ref_df = monitoring_service.load_reference_dataset()

    # buffer_max_size is set to 50 in fixture. Ingest 80 observations.
    monitoring_service.ingest_feature_observations(ref_df.iloc[:80])

    session = monitoring_service.SessionLocal()
    try:
        count = session.query(FeatureObservationRecord).count()
        # FIFO pruning must keep total <= buffer_max_size (50)
        assert count <= 50
    finally:
        session.close()


def test_ground_truth_ingestion_and_evaluation(monitoring_service):
    """Verifies linking delayed ground truth to predictions and computing evaluation metrics."""
    # Ingest 50 ground-truth outcomes
    np.random.seed(42)
    n = 60
    req_ids = [f"req-uuid-{i:04d}" for i in range(n)]
    actuals = np.random.choice([0, 1], size=n, p=[0.7, 0.3])

    for rid, act in zip(req_ids, actuals):
        monitoring_service.ingest_ground_truth(request_id=rid, actual_churn=int(act))

    # Create matching predictions dataframe
    probs = np.clip(actuals * 0.6 + np.random.uniform(0.1, 0.3, size=n), 0.0, 1.0)
    matched_df = pd.DataFrame({
        "request_id": req_ids,
        "churn_probability": probs,
        "actual_churn": actuals,
    })

    # Supply valid feature observations alongside ground-truth feedback
    ref_df = monitoring_service.load_reference_dataset()
    features_batch = ref_df.iloc[:n].copy()

    summary = monitoring_service.run_monitoring_cycle(
        current_data=features_batch,
        ground_truth_df=matched_df,
    )
    assert summary.performance_status == "EVALUATED"
    assert not summary.performance_degraded

    session = monitoring_service.SessionLocal()
    try:
        perf_rec = session.query(ModelPerformanceReportRecord).filter_by(report_id=summary.report_id).first()
        assert perf_rec is not None
        assert perf_rec.status == "EVALUATED"
        assert perf_rec.roc_auc is not None and perf_rec.roc_auc > 0.80
        assert perf_rec.true_positives is not None
    finally:
        session.close()


def test_fastapi_monitoring_endpoints(test_db_uri, tmp_path):
    """Verifies that FastAPI serving engine exposes operational monitoring endpoints."""
    app_cfg = get_config()
    app_cfg.monitoring.monitoring_db_uri = test_db_uri
    app_cfg.monitoring.reports_dir = str(tmp_path / "fastapi_reports")

    mon_svc = MonitoringService(db_uri=test_db_uri, app_config=app_cfg)
    app = create_app(config=app_cfg, monitoring_service=mon_svc)
    client = TestClient(app)

    # 1. GET /monitoring/status before any run
    r_status = client.get("/monitoring/status")
    assert r_status.status_code == 200
    data = r_status.json()
    assert "active_model" in data
    assert data["buffered_observations_count"] == 0

    # 2. Ingest observations into buffer
    ref_df = mon_svc.load_reference_dataset().head(40)
    mon_svc.ingest_feature_observations(ref_df)

    # 3. POST /monitoring/feedback
    r_fb = client.post("/monitoring/feedback", json={
        "request_id": "fastapi-req-001",
        "actual_churn": 1,
        "source": "api_test",
    })
    assert r_fb.status_code == 200
    assert r_fb.json()["status"] == "success"

    # 4. POST /monitoring/run with Admin Token
    r_run = client.post("/monitoring/run", headers={"X-Admin-API-Key": app_cfg.serving.reload_token})
    assert r_run.status_code == 200
    run_data = r_run.json()
    assert run_data["status"] == "completed"
    assert "report_id" in run_data

    # 5. GET /monitoring/latest
    r_latest = client.get("/monitoring/latest")
    assert r_latest.status_code == 200
    latest_data = r_latest.json()
    assert latest_data["report_id"] == run_data["report_id"]

    mon_svc.engine.dispose()
