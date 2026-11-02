"""Unit and Integration Tests for MLflow Experiment Tracking, Gating, and Model Registry."""

from pathlib import Path
import tempfile
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.common.config import AppConfig, get_config
from src.data.preprocessor import build_full_pipeline
from src.models.mlflow_manager import GatingResult, MLflowManager, RollbackResult
from src.models.train import prepare_features_and_target


@pytest.fixture
def temp_mlflow_manager(tmp_path: Path, test_config: AppConfig) -> MLflowManager:
    """Creates an isolated MLflowManager using a temporary SQLite tracking database."""
    temp_db = tmp_path / "test_mlflow.db"
    temp_artifacts = tmp_path / "test_mlruns"

    cfg = test_config.model_copy(deep=True)
    cfg.mlflow.tracking_uri = f"sqlite:///{temp_db.as_posix()}"
    cfg.mlflow.artifact_location = str(temp_artifacts)
    cfg.mlflow.experiment_name = "test-churn-prediction"
    cfg.mlflow.registered_model_name = "test-churn-classifier"

    return MLflowManager(cfg)


def test_mlflow_experiment_initialization(temp_mlflow_manager: MLflowManager):
    """Verifies that MLflow experiment is initialized correctly."""
    client = temp_mlflow_manager.client
    exp = client.get_experiment_by_name(temp_mlflow_manager.experiment_name)
    assert exp is not None
    assert exp.name == "test-churn-prediction"
    assert exp.lifecycle_stage == "active"


def test_mlflow_log_model_run_and_registration(
    temp_mlflow_manager: MLflowManager,
    clean_churn_df: pd.DataFrame,
    test_config: AppConfig,
    tmp_path: Path,
):
    """Verifies logging of parameters, separate split metrics, artifacts, signatures, and registration."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)
    pipeline = build_full_pipeline(LogisticRegression(max_iter=200), test_config)
    pipeline.fit(X, y)

    # Create dummy artifact
    dummy_art = tmp_path / "dummy_roc.png"
    dummy_art.write_text("dummy plot content", encoding="utf-8")

    from src.data.loader import validate_dataset_provenance
    from src.common.config import find_project_root
    root = find_project_root()
    _, prov = validate_dataset_provenance(root / test_config.paths.raw_data, test_config)

    train_m = {"roc_auc": 0.85, "pr_auc": 0.65, "f1": 0.60, "brier_score": 0.16}
    val_m = {"roc_auc": 0.84, "pr_auc": 0.64, "f1": 0.59, "brier_score": 0.17, "optimal_threshold": 0.55}
    test_m = {
        "roc_auc": 0.851,
        "pr_auc": 0.662,
        "f1": 0.635,
        "precision": 0.55,
        "recall": 0.75,
        "brier_score": 0.162,
        "business_cost": {"total_cost": 41000.0},
        "latency_benchmark": {"p95_ms": 12.5, "mean_ms": 11.2},
    }

    run_id, version = temp_mlflow_manager.log_model_run(
        model_name="TestLR",
        pipeline=pipeline,
        hyperparameters={"C": 1.0, "max_iter": 200},
        train_metrics=train_m,
        val_metrics=val_m,
        test_metrics=test_m,
        X_sample=X,
        provenance=prov,
        artifacts_to_log=[dummy_art],
    )

    client = temp_mlflow_manager.client
    run = client.get_run(run_id)

    # Verify separate metrics logged
    assert run.data.metrics["train_roc_auc"] == 0.85
    assert run.data.metrics["val_roc_auc"] == 0.84
    assert run.data.metrics["test_roc_auc"] == 0.851
    assert run.data.metrics["test_latency_p95_ms"] == 12.5

    # Verify tags
    assert run.data.tags["model_name"] == "TestLR"
    assert "git_commit" in run.data.tags
    assert "dataset_sha256" in run.data.tags

    # Verify model registered with candidate alias
    cand = temp_mlflow_manager.get_model_version_by_alias("candidate")
    assert cand is not None
    assert str(cand.version) == version


def test_champion_challenger_gating_and_promotion(
    temp_mlflow_manager: MLflowManager,
    clean_churn_df: pd.DataFrame,
    test_config: AppConfig,
):
    """Verifies champion vs challenger evaluation gates, alias assignment, and demotion."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)
    pipeline = build_full_pipeline(LogisticRegression(max_iter=200), test_config)
    pipeline.fit(X, y)

    from src.data.loader import validate_dataset_provenance
    from src.common.config import find_project_root
    root = find_project_root()
    _, prov = validate_dataset_provenance(root / test_config.paths.raw_data, test_config)

    # 1. Register Initial Model (Candidate 1)
    metrics_v1 = {
        "roc_auc": 0.850,
        "pr_auc": 0.660,
        "f1": 0.620,
        "precision": 0.54,
        "recall": 0.74,
        "brier_score": 0.165,
        "business_cost": {"total_cost": 42000.0},
        "latency_benchmark": {"p95_ms": 14.0, "mean_ms": 12.0},
    }
    run_id_v1, ver_1 = temp_mlflow_manager.log_model_run(
        model_name="Model_v1",
        pipeline=pipeline,
        hyperparameters={"C": 1.0},
        train_metrics={"roc_auc": 0.85, "pr_auc": 0.66, "f1": 0.62, "brier_score": 0.16},
        val_metrics={"roc_auc": 0.85, "pr_auc": 0.66, "f1": 0.62, "brier_score": 0.16, "optimal_threshold": 0.5},
        test_metrics=metrics_v1,
        X_sample=X,
        provenance=prov,
    )

    # Gate check on initial candidate -> becomes initial champion
    gate_res_1 = temp_mlflow_manager.evaluate_candidate_for_promotion(metrics_v1, ver_1)
    assert gate_res_1.passed is True
    assert gate_res_1.is_initial_champion is True

    # Promote to champion
    temp_mlflow_manager.promote_to_champion(ver_1)
    champ = temp_mlflow_manager.get_model_version_by_alias("champion")
    assert str(champ.version) == str(ver_1)

    # 2. Register Candidate 2 that FAILS improvement gate (PR-AUC worse)
    metrics_v2_failing = {
        "roc_auc": 0.840,
        "pr_auc": 0.655,  # 0.655 < 0.660 + 0.005 -> Fails Gate 5
        "f1": 0.600,
        "precision": 0.52,
        "recall": 0.70,
        "brier_score": 0.170,
        "business_cost": {"total_cost": 45000.0},
        "latency_benchmark": {"p95_ms": 15.0, "mean_ms": 13.0},
    }
    _, ver_2 = temp_mlflow_manager.log_model_run(
        model_name="Model_v2_Fail",
        pipeline=pipeline,
        hyperparameters={"C": 0.5},
        train_metrics={"roc_auc": 0.84, "pr_auc": 0.65, "f1": 0.60, "brier_score": 0.17},
        val_metrics={"roc_auc": 0.84, "pr_auc": 0.65, "f1": 0.60, "brier_score": 0.17, "optimal_threshold": 0.5},
        test_metrics=metrics_v2_failing,
        X_sample=X,
        provenance=prov,
    )

    gate_res_2 = temp_mlflow_manager.evaluate_candidate_for_promotion(metrics_v2_failing, ver_2)
    assert gate_res_2.passed is False
    assert any("FAILED Gate 5" in r for r in gate_res_2.reasons)

    # 3. Register Candidate 3 that PASSES improvement gate (PR-AUC improves > 0.005)
    metrics_v3_passing = {
        "roc_auc": 0.860,
        "pr_auc": 0.675,  # 0.675 >= 0.660 + 0.005 -> Passes Gate 5!
        "f1": 0.650,
        "precision": 0.58,
        "recall": 0.78,
        "brier_score": 0.155,
        "business_cost": {"total_cost": 39000.0},
        "latency_benchmark": {"p95_ms": 13.0, "mean_ms": 11.5},
    }
    _, ver_3 = temp_mlflow_manager.log_model_run(
        model_name="Model_v3_Pass",
        pipeline=pipeline,
        hyperparameters={"C": 2.0},
        train_metrics={"roc_auc": 0.86, "pr_auc": 0.68, "f1": 0.66, "brier_score": 0.15},
        val_metrics={"roc_auc": 0.86, "pr_auc": 0.68, "f1": 0.65, "brier_score": 0.15, "optimal_threshold": 0.5},
        test_metrics=metrics_v3_passing,
        X_sample=X,
        provenance=prov,
    )

    gate_res_3 = temp_mlflow_manager.evaluate_candidate_for_promotion(metrics_v3_passing, ver_3)
    assert gate_res_3.passed is True

    # Promote v3 to champion
    temp_mlflow_manager.promote_to_champion(ver_3)

    # Verify v3 is now champion
    new_champ = temp_mlflow_manager.get_model_version_by_alias("champion")
    assert str(new_champ.version) == str(ver_3)

    # Verify v1 was demoted to archived
    archived = temp_mlflow_manager.get_model_version_by_alias("archived")
    assert archived is not None
    assert str(archived.version) == str(ver_1)


def test_registry_rollback_functionality(
    temp_mlflow_manager: MLflowManager,
    clean_churn_df: pd.DataFrame,
    test_config: AppConfig,
):
    """Verifies that rollback swaps the champion alias back to the previous version."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)
    pipeline = build_full_pipeline(LogisticRegression(max_iter=100), test_config)
    pipeline.fit(X, y)

    from src.data.loader import validate_dataset_provenance
    from src.common.config import find_project_root
    root = find_project_root()
    _, prov = validate_dataset_provenance(root / test_config.paths.raw_data, test_config)

    dummy_m = {
        "roc_auc": 0.85,
        "pr_auc": 0.66,
        "f1": 0.62,
        "precision": 0.55,
        "recall": 0.75,
        "brier_score": 0.16,
        "business_cost": {"total_cost": 40000.0},
        "latency_benchmark": {"p95_ms": 12.0, "mean_ms": 10.0},
    }

    # Register v1 and promote to champion
    _, v1 = temp_mlflow_manager.log_model_run(
        model_name="V1",
        pipeline=pipeline,
        hyperparameters={"C": 1.0},
        train_metrics={"roc_auc": 0.85, "pr_auc": 0.66, "f1": 0.62, "brier_score": 0.16},
        val_metrics={"roc_auc": 0.85, "pr_auc": 0.66, "f1": 0.62, "brier_score": 0.16, "optimal_threshold": 0.5},
        test_metrics=dummy_m,
        X_sample=X,
        provenance=prov,
    )
    temp_mlflow_manager.promote_to_champion(v1)

    # Register v2 and promote to champion
    _, v2 = temp_mlflow_manager.log_model_run(
        model_name="V2",
        pipeline=pipeline,
        hyperparameters={"C": 2.0},
        train_metrics={"roc_auc": 0.86, "pr_auc": 0.67, "f1": 0.63, "brier_score": 0.16},
        val_metrics={"roc_auc": 0.86, "pr_auc": 0.67, "f1": 0.63, "brier_score": 0.16, "optimal_threshold": 0.5},
        test_metrics=dummy_m,
        X_sample=X,
        provenance=prov,
    )
    temp_mlflow_manager.promote_to_champion(v2)

    # Current champion should be v2
    assert str(temp_mlflow_manager.get_model_version_by_alias("champion").version) == str(v2)

    # Execute rollback
    rollback_res = temp_mlflow_manager.rollback_champion()
    assert rollback_res.success is True
    assert str(rollback_res.previous_version) == str(v2)
    assert str(rollback_res.restored_version) == str(v1)

    # Verify that registry resolution now points to v1 as champion
    resolved_champ = temp_mlflow_manager.get_model_version_by_alias("champion")
    assert str(resolved_champ.version) == str(v1)


def test_dynamic_model_loading_and_prediction_consistency(
    temp_mlflow_manager: MLflowManager,
    clean_churn_df: pd.DataFrame,
    test_config: AppConfig,
):
    """Verifies loading model via alias URI produces identical predictions to memory pipeline."""
    X, y = prepare_features_and_target(clean_churn_df, test_config)
    orig_pipeline = build_full_pipeline(LogisticRegression(max_iter=150, random_state=42), test_config)
    orig_pipeline.fit(X, y)

    from src.data.loader import validate_dataset_provenance
    from src.common.config import find_project_root
    root = find_project_root()
    _, prov = validate_dataset_provenance(root / test_config.paths.raw_data, test_config)

    dummy_m = {
        "roc_auc": 0.85,
        "pr_auc": 0.66,
        "f1": 0.62,
        "precision": 0.55,
        "recall": 0.75,
        "brier_score": 0.16,
        "business_cost": {"total_cost": 40000.0},
        "latency_benchmark": {"p95_ms": 12.0, "mean_ms": 10.0},
    }

    _, ver = temp_mlflow_manager.log_model_run(
        model_name="ConsistentModel",
        pipeline=orig_pipeline,
        hyperparameters={"C": 1.0},
        train_metrics={"roc_auc": 0.85, "pr_auc": 0.66, "f1": 0.62, "brier_score": 0.16},
        val_metrics={"roc_auc": 0.85, "pr_auc": 0.66, "f1": 0.62, "brier_score": 0.16, "optimal_threshold": 0.5},
        test_metrics=dummy_m,
        X_sample=X,
        provenance=prov,
    )
    temp_mlflow_manager.promote_to_champion(ver)

    # Dynamically load model via alias
    loaded_pipe = temp_mlflow_manager.load_model_by_alias("champion")

    # Predict with both pipelines on test sample
    X_test_sample = X.head(10)
    orig_probs = orig_pipeline.predict_proba(X_test_sample)
    loaded_probs = loaded_pipe.predict_proba(X_test_sample)

    np.testing.assert_allclose(
        orig_probs,
        loaded_probs,
        atol=1e-5,
        err_msg="Predictions from loaded alias model differ from original memory pipeline!",
    )
