"""MLflow Experiment Tracking and Model Registry Lifecycle Manager."""

import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Suppress agent hint warnings
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from pydantic import BaseModel
import sklearn
import xgboost

from src.common.config import AppConfig, find_project_root, get_config
from src.common.exceptions import ModelArtifactError, ModelTrainingError
from src.common.logger import get_logger

logger = get_logger(__name__)


class GatingResult(BaseModel):
    """Result of champion-vs-challenger evaluation gates."""

    passed: bool
    is_initial_champion: bool
    reasons: List[str]
    candidate_metrics: Dict[str, float]
    champion_metrics: Optional[Dict[str, float]] = None
    improvements: Dict[str, float] = {}


class RollbackResult(BaseModel):
    """Result of model rollback operation in the registry."""

    success: bool
    previous_version: str
    restored_version: str
    restored_alias: str
    message: str


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, with defensive fallback."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "uncommitted_or_git_missing"


class MLflowManager:
    """Manages MLflow experiment tracking, model signatures, and registry aliases."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self.root = find_project_root()

        # Format SQLite tracking URI with absolute project root path
        tracking_uri = self.cfg.mlflow.tracking_uri
        if tracking_uri.startswith("sqlite:///"):
            db_rel_path = tracking_uri.replace("sqlite:///", "")
            db_abs_path = (self.root / db_rel_path).resolve()
            self.tracking_uri = f"sqlite:///{db_abs_path.as_posix()}"
        else:
            self.tracking_uri = tracking_uri

        # Artifact directory
        self.artifact_location = str((self.root / self.cfg.mlflow.artifact_location).resolve())

        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_name = self.cfg.mlflow.experiment_name
        self.model_name = self.cfg.mlflow.registered_model_name

        self._ensure_experiment_exists()

    def _ensure_experiment_exists(self) -> str:
        """Ensures the target MLflow experiment is initialized with artifact store."""
        exp = self.client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            artifact_dir = Path(self.artifact_location)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            exp_id = self.client.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location,
            )
            logger.info("Created new MLflow experiment '%s' (ID: %s)", self.experiment_name, exp_id)
        else:
            exp_id = exp.experiment_id
            logger.info("Found existing MLflow experiment '%s' (ID: %s)", self.experiment_name, exp_id)

        mlflow.set_experiment(self.experiment_name)
        return exp_id

    def log_model_run(
        self,
        model_name: str,
        pipeline: Any,
        hyperparameters: Dict[str, Any],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, Any],
        X_sample: pd.DataFrame,
        provenance: Any,
        artifacts_to_log: Optional[List[Path]] = None,
        run_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Logs a complete reproducible experiment run, parameters, metrics, artifacts, and registers model.

        Args:
            model_name: Identifier for model (e.g. 'XGBoost_Tuned').
            pipeline: Fitted Scikit-learn Pipeline encapsulating preprocessing and classifier.
            hyperparameters: Estimator hyperparameters.
            train_metrics: Metrics computed on training split.
            val_metrics: Metrics computed on validation split.
            test_metrics: Metrics computed on holdout test partition.
            X_sample: Sample feature DataFrame for input example & schema signature.
            provenance: Dataset provenance record.
            artifacts_to_log: List of image/report file paths to upload as run artifacts.
            run_name: Optional custom name for the MLflow run.

        Returns:
            Tuple of (run_id, registered_model_version).
        """
        active_run_name = run_name or f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=active_run_name) as run:
            run_id = run.info.run_id
            logger.info("Logging MLflow run '%s' (Run ID: %s)", active_run_name, run_id)

            # 1. Log System & Provenance Tags
            git_sha = get_git_commit_hash()
            mlflow.set_tags({
                "model_name": model_name,
                "project": self.cfg.project.name,
                "git_commit": git_sha,
                "python_version": sys.version.split()[0],
                "sklearn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
                "mlflow_version": mlflow.__version__,
                "dataset_source": provenance.source_type,
                "dataset_sha256": provenance.sha256,
                "dataset_is_real": str(provenance.is_real_data),
            })

            # 2. Log Preprocessing & Hyperparameters
            mlflow.log_params({
                "model_type": model_name,
                "random_seed": self.cfg.project.random_seed,
                "numeric_imputer": self.cfg.preprocessing.numerical_imputer_strategy,
                "categorical_imputer": self.cfg.preprocessing.categorical_imputer_strategy,
                "scaler": self.cfg.preprocessing.scaler,
                "one_hot_drop": self.cfg.preprocessing.one_hot_drop,
                **{f"param_{k}": str(v) for k, v in hyperparameters.items()},
            })

            # 3. Log Separate Partition Metrics
            # Training metrics
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", float(v))

            # Validation metrics
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v))

            # Test metrics
            mlflow.log_metrics({
                "test_roc_auc": float(test_metrics["roc_auc"]),
                "test_pr_auc": float(test_metrics["pr_auc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_brier_score": float(test_metrics["brier_score"]),
                "test_total_cost": float(test_metrics["business_cost"]["total_cost"]),
                "test_latency_p95_ms": float(test_metrics["latency_benchmark"]["p95_ms"]),
                "test_latency_mean_ms": float(test_metrics["latency_benchmark"]["mean_ms"]),
            })

            # 4. Log Diagnostic Artifacts
            if artifacts_to_log:
                for art_path in artifacts_to_log:
                    if art_path.is_file():
                        mlflow.log_artifact(str(art_path), artifact_path="diagnostics")

            # 5. Infer Signature & Input Example
            X_example = X_sample.head(2)
            y_pred_example = pipeline.predict_proba(X_example)
            signature = infer_signature(X_example, y_pred_example)

            # 6. Log Scikit-learn Pipeline & Register Model
            logger.info("Logging fitted pipeline model with MLflow signature...")
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                signature=signature,
                input_example=X_example,
                registered_model_name=self.model_name,
            )

            # Retrieve registered model version
            registered_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            latest_version = max(int(mv.version) for mv in registered_versions)
            version_str = str(latest_version)

            # Assign candidate alias by default
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias=self.cfg.mlflow.aliases.candidate,
                version=version_str,
            )

            # Tag version with run ID and training timestamp
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version_str,
                key="model_name",
                value=model_name,
            )
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version_str,
                key="created_at",
                value=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                "Model registered in MLflow: Name='%s', Version=%s, Alias='%s', Run ID='%s'",
                self.model_name,
                version_str,
                self.cfg.mlflow.aliases.candidate,
                run_id,
            )

            return run_id, version_str

    def get_model_version_by_alias(self, alias: str) -> Optional[Any]:
        """Resolves model version object for a specified alias."""
        try:
            return self.client.get_model_version_by_alias(self.model_name, alias)
        except Exception:
            return None

    def evaluate_candidate_for_promotion(
        self,
        candidate_metrics: Dict[str, Any],
        candidate_version: str,
    ) -> GatingResult:
        """Evaluates candidate against configured acceptance criteria and current champion.

        Args:
            candidate_metrics: Holdout test metrics of the candidate.
            candidate_version: Registered model version number.

        Returns:
            GatingResult: Pass/fail determination with detailed audit trail.
        """
        gate_cfg = self.cfg.mlflow.gating
        reasons: List[str] = []
        cand_flat = {
            "roc_auc": float(candidate_metrics["roc_auc"]),
            "pr_auc": float(candidate_metrics["pr_auc"]),
            "f1": float(candidate_metrics["f1"]),
            "latency_p95_ms": float(candidate_metrics["latency_benchmark"]["p95_ms"]),
            "total_cost": float(candidate_metrics["business_cost"]["total_cost"]),
        }

        # Gate 1: ROC-AUC threshold
        if cand_flat["roc_auc"] < gate_cfg.min_roc_auc:
            reasons.append(f"FAILED Gate 1: ROC-AUC {cand_flat['roc_auc']:.4f} < min target {gate_cfg.min_roc_auc}")
        else:
            reasons.append(f"PASSED Gate 1: ROC-AUC {cand_flat['roc_auc']:.4f} >= {gate_cfg.min_roc_auc}")

        # Gate 2: PR-AUC threshold
        if cand_flat["pr_auc"] < gate_cfg.min_pr_auc:
            reasons.append(f"FAILED Gate 2: PR-AUC {cand_flat['pr_auc']:.4f} < min target {gate_cfg.min_pr_auc}")
        else:
            reasons.append(f"PASSED Gate 2: PR-AUC {cand_flat['pr_auc']:.4f} >= {gate_cfg.min_pr_auc}")

        # Gate 3: F1 threshold
        if cand_flat["f1"] < gate_cfg.min_f1:
            reasons.append(f"FAILED Gate 3: F1-Score {cand_flat['f1']:.4f} < min target {gate_cfg.min_f1}")
        else:
            reasons.append(f"PASSED Gate 3: F1-Score {cand_flat['f1']:.4f} >= {gate_cfg.min_f1}")

        # Gate 4: Latency SLA threshold
        if cand_flat["latency_p95_ms"] > gate_cfg.max_latency_p95_ms:
            reasons.append(
                f"FAILED Gate 4: Latency p95 {cand_flat['latency_p95_ms']:.2f} ms > max {gate_cfg.max_latency_p95_ms} ms"
            )
        else:
            reasons.append(
                f"PASSED Gate 4: Latency p95 {cand_flat['latency_p95_ms']:.2f} ms <= {gate_cfg.max_latency_p95_ms} ms"
            )

        # Champion Comparison
        champ_version_obj = self.get_model_version_by_alias(self.cfg.mlflow.aliases.champion)
        improvements: Dict[str, float] = {}

        if champ_version_obj is None:
            # First model to pass gates becomes initial champion
            base_passed = all("FAILED" not in r for r in reasons)
            reasons.append("No active champion in registry. Candidate qualifies as initial Champion.")
            return GatingResult(
                passed=base_passed,
                is_initial_champion=True,
                reasons=reasons,
                candidate_metrics=cand_flat,
                champion_metrics=None,
                improvements={},
            )

        # If existing champion exists, fetch run metrics
        champ_run = self.client.get_run(champ_version_obj.run_id)
        champ_metrics = {
            "roc_auc": champ_run.data.metrics.get("test_roc_auc", 0.0),
            "pr_auc": champ_run.data.metrics.get("test_pr_auc", 0.0),
            "f1": champ_run.data.metrics.get("test_f1", 0.0),
            "latency_p95_ms": champ_run.data.metrics.get("test_latency_p95_ms", 999.0),
            "total_cost": champ_run.data.metrics.get("test_total_cost", 1e9),
        }

        delta_pr_auc = cand_flat["pr_auc"] - champ_metrics["pr_auc"]
        improvements["delta_pr_auc"] = delta_pr_auc
        improvements["delta_roc_auc"] = cand_flat["roc_auc"] - champ_metrics["roc_auc"]
        improvements["delta_cost"] = champ_metrics["total_cost"] - cand_flat["total_cost"]

        # Gate 5: Challenger must improve PR-AUC over Champion by at least min_improvement
        if delta_pr_auc < gate_cfg.min_improvement_pr_auc:
            reasons.append(
                f"FAILED Gate 5: Challenger PR-AUC ({cand_flat['pr_auc']:.4f}) does not improve upon "
                f"Champion ({champ_metrics['pr_auc']:.4f}) by at least {gate_cfg.min_improvement_pr_auc} "
                f"(Delta: {delta_pr_auc:+.4f})"
            )
        else:
            reasons.append(
                f"PASSED Gate 5: Challenger PR-AUC improves over Champion by {delta_pr_auc:+.4f} "
                f"(Target: >= +{gate_cfg.min_improvement_pr_auc})"
            )

        passed = all("FAILED" not in r for r in reasons)

        return GatingResult(
            passed=passed,
            is_initial_champion=False,
            reasons=reasons,
            candidate_metrics=cand_flat,
            champion_metrics=champ_metrics,
            improvements=improvements,
        )

    def promote_to_champion(self, version: str, notes: Optional[str] = None) -> bool:
        """Promotes a registered version to 'champion' alias and archives the previous champion."""
        old_champ = self.get_model_version_by_alias(self.cfg.mlflow.aliases.champion)
        version = str(version)

        if old_champ and str(old_champ.version) != version:
            # Move old champion to archived
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias=self.cfg.mlflow.aliases.archived,
                version=str(old_champ.version),
            )
            self.client.set_model_version_tag(
                name=self.model_name,
                version=str(old_champ.version),
                key="lifecycle_state",
                value="archived",
            )
            logger.info("Demoted previous Champion (v%s) to 'archived' alias.", old_champ.version)

        # Set champion alias
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=self.cfg.mlflow.aliases.champion,
            version=version,
        )
        self.client.set_model_version_tag(
            name=self.model_name,
            version=version,
            key="lifecycle_state",
            value="champion",
        )
        self.client.set_model_version_tag(
            name=self.model_name,
            version=version,
            key="promoted_at",
            value=datetime.now(timezone.utc).isoformat(),
        )
        if notes:
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="promotion_notes",
                value=notes,
            )

        logger.info("PROMOTED version %s to 'champion' alias in registry '%s'.", version, self.model_name)
        return True

    def promote_to_challenger(self, version: str) -> bool:
        """Sets 'challenger' alias on a registered version."""
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=self.cfg.mlflow.aliases.challenger,
            version=version,
        )
        logger.info("Assigned 'challenger' alias to version %s in registry '%s'.", version, self.model_name)
        return True

    def rollback_champion(self, target_version: Optional[str] = None) -> RollbackResult:
        """Rolls back the active champion to a target version or the latest archived version.

        Verifies that registry resolution actually points to the restored version.

        Args:
            target_version: Specific version string to restore. If None, restores the latest archived version.

        Returns:
            RollbackResult: Audit record of the rollback transaction.
        """
        current_champ = self.get_model_version_by_alias(self.cfg.mlflow.aliases.champion)
        if current_champ is None:
            raise ModelArtifactError("Cannot rollback: No active champion currently exists in registry.")

        curr_ver = str(current_champ.version)

        if target_version is None:
            # Find the latest archived version or highest version < current
            all_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            prior_versions = [v for v in all_versions if int(v.version) < int(curr_ver)]
            if not prior_versions:
                raise ModelArtifactError(f"Cannot rollback: No prior model versions exist before version {curr_ver}.")
            # Sort by version descending
            prior_versions.sort(key=lambda x: int(x.version), reverse=True)
            target_version = str(prior_versions[0].version)
        else:
            target_version = str(target_version)

        # Demote current champion
        self.client.set_model_version_tag(
            name=self.model_name,
            version=curr_ver,
            key="lifecycle_state",
            value="rolled_back",
        )
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=self.cfg.mlflow.aliases.archived,
            version=curr_ver,
        )

        # Restore target version to champion
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=self.cfg.mlflow.aliases.champion,
            version=target_version,
        )
        self.client.set_model_version_tag(
            name=self.model_name,
            version=target_version,
            key="lifecycle_state",
            value="champion",
        )
        self.client.set_model_version_tag(
            name=self.model_name,
            version=target_version,
            key="rolled_back_to_at",
            value=datetime.now(timezone.utc).isoformat(),
        )

        # Verify resolution
        resolved_champ = self.get_model_version_by_alias(self.cfg.mlflow.aliases.champion)
        if resolved_champ is None or str(resolved_champ.version) != str(target_version):
            raise ModelArtifactError(
                f"Rollback verification failed: Registry resolves to version {resolved_champ.version if resolved_champ else None}, expected {target_version}."
            )

        logger.info(
            "ROLLBACK SUCCESS: Champion rolled back from v%s to v%s in registry '%s'.",
            curr_ver,
            target_version,
            self.model_name,
        )

        return RollbackResult(
            success=True,
            previous_version=curr_ver,
            restored_version=target_version,
            restored_alias="champion",
            message=f"Successfully restored champion from version {curr_ver} to version {target_version}.",
        )

    def load_model_by_alias(self, alias: str) -> Any:
        """Loads a model directly using its alias URI: 'models:/{name}@{alias}'.

        Ensures dynamic alias resolution without requiring explicit version pinning.

        Args:
            alias: Model alias ('champion', 'challenger', 'candidate', 'archived').

        Returns:
            The loaded Scikit-learn Pipeline instance.
        """
        model_uri = f"models:/{self.model_name}@{alias}"
        logger.info("Loading model dynamically via alias URI: %s", model_uri)
        try:
            pipeline = mlflow.sklearn.load_model(model_uri)
            return pipeline
        except Exception as e:
            raise ModelArtifactError(f"Failed to load model by alias URI '{model_uri}': {e}") from e
