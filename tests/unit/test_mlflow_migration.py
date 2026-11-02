"""Unit and Integration Tests for MLflow Registry Migration Tool."""

import json
import os
from pathlib import Path
import shutil
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from scripts.migrate_mlflow_registry import MLflowMigrationManager
from src.common.config import get_config


@pytest.fixture(scope="module")
def project_config():
    return get_config()


def test_migration_manager_inspect_source(project_config):
    """Verifies that MLflowMigrationManager accurately inventories the source SQLite registry."""
    manager = MLflowMigrationManager(config=project_config)
    inventory = manager.inspect_source_registry()

    assert inventory["model_name"] == "customer-churn-classifier"
    assert inventory["total_runs"] >= 4
    assert len(inventory["versions"]) >= 4
    assert "champion" in inventory["aliases"]
    assert "challenger" in inventory["aliases"]
    assert str(inventory["aliases"]["champion"]) == "4"


def test_migration_manager_audit_only(project_config, tmp_path):
    """Verifies in-place audit mode without modifying source or requiring external target."""
    manager = MLflowMigrationManager(config=project_config)
    result = manager.execute_migration(audit_only=True)

    assert result["status"] == "AUDIT_ONLY_SOURCE_VERIFIED"
    assert result["parity_verified"] is True
    assert result["source_inventory"]["total_runs"] >= 4


def test_migration_manager_unreachable_target_handling(project_config):
    """Verifies that an offline target PostgreSQL backend is cleanly handled and audited."""
    # Point to an inaccessible port/host
    unreachable_pg_uri = "postgresql+psycopg2://invalid_user:invalid_pass@127.0.0.1:54329/nonexistent_db"
    manager = MLflowMigrationManager(
        target_tracking_uri=unreachable_pg_uri,
        config=project_config,
    )

    result = manager.execute_migration(audit_only=False)

    assert result["status"] == "PENDING_TARGET_BACKEND"
    assert result["parity_verified"] is False
    assert result["source_tracking_uri"] != result["target_tracking_uri"]
    assert "Target database (e.g. PostgreSQL in Docker container) is offline" in result["notes"]


def test_full_end_to_end_migration_with_parity(project_config, tmp_path):
    """Validates complete zero-loss migration between two distinct backends with prediction parity."""
    # Create an isolated temporary SQLite database and artifact folder as destination backend
    target_db_path = tmp_path / "target_mlflow.db"
    target_artifacts_dir = tmp_path / "target_artifacts"
    target_artifacts_dir.mkdir(parents=True, exist_ok=True)

    target_uri = f"sqlite:///{target_db_path.resolve().as_posix()}"

    manager = MLflowMigrationManager(
        target_tracking_uri=target_uri,
        config=project_config,
    )

    # Execute full migration
    result = manager.execute_migration(
        target_artifact_root=str(target_artifacts_dir.resolve()),
        audit_only=False,
    )

    assert result["status"] == "SUCCESS"
    assert result["parity_verified"] is True
    assert result["source_tracking_uri"] != result["target_tracking_uri"]

    # Verify destination contents directly via target client
    target_client = MlflowClient(tracking_uri=target_uri)
    target_exp = target_client.get_experiment_by_name("customer-churn-prediction")
    assert target_exp is not None

    target_runs = target_client.search_runs([target_exp.experiment_id])
    assert len(target_runs) == result["source_inventory"]["total_runs"]

    # Verify model versions and aliases on target
    target_model = target_client.get_registered_model("customer-churn-classifier")
    assert target_model is not None
    assert "champion" in target_model.aliases
    assert "challenger" in target_model.aliases

    # Verify loading and prediction from target
    mlflow.set_tracking_uri(target_uri)
    target_champion = mlflow.sklearn.load_model("models:/customer-churn-classifier@champion")
    assert target_champion is not None

    sample_df = pd.read_csv(Path(project_config.paths.sample_data)).head(3)
    preds = target_champion.predict_proba(sample_df)
    assert preds.shape == (3, 2)
    assert np.all((preds >= 0.0) & (preds <= 1.0))
