"""Thread-Safe Dynamic MLflow Model Loader with Atomic Reload."""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
from typing import Any, Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.common.config import AppConfig, find_project_root, get_config
from src.common.logger import get_logger
from src.serving.exceptions import ModelNotLoadedError, ModelReloadError

logger = get_logger(__name__)

# Suppress MLflow agent hint in serving logs
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


@dataclass(frozen=True)
class ModelContainer:
    """Immutable container encapsulating an active pipeline model and registry metadata."""

    pipeline: Any
    model_name: str
    model_version: str
    run_id: str
    alias: str
    load_timestamp: datetime


class ModelLoader:
    """Manages thread-safe dynamic MLflow model loading, caching, and atomic reloads.

    Guarantees:
    - Zero per-request disk/network reload overhead (cached in-memory).
    - Thread-safe atomic pointer swap prevents concurrent requests from seeing partially loaded models.
    - Failed reloads preserve the previous healthy active model.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self.root = find_project_root()
        self._lock = threading.RLock()
        self._active_container: Optional[ModelContainer] = None

        # Resolve tracking URI with absolute project root
        tracking_uri = self.cfg.mlflow.tracking_uri
        if tracking_uri.startswith("sqlite:///"):
            db_rel = tracking_uri.replace("sqlite:///", "")
            db_abs = (self.root / db_rel).resolve()
            self.tracking_uri = f"sqlite:///{db_abs.as_posix()}"
        else:
            self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.model_name = self.cfg.mlflow.registered_model_name
        self.default_alias = self.cfg.serving.model_alias

    def _create_warmup_sample(self) -> pd.DataFrame:
        """Creates a single-row synthetic payload matching the schema for pipeline verification."""
        data = {
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [50.0],
            "TotalCharges": [600.0],
        }
        return pd.DataFrame(data)

    def load_champion_model(self, alias: Optional[str] = None) -> ModelContainer:
        """Loads and activates the target model alias from the MLflow model registry.

        Args:
            alias: Registry alias to load. Defaults to configured champion.

        Returns:
            ModelContainer: Initialized and validated immutable container.
        """
        target_alias = alias or self.default_alias
        logger.info(
            "Resolving model for registry='%s', alias='%s' from tracking URI='%s'...",
            self.model_name,
            target_alias,
            self.tracking_uri,
        )

        try:
            mv = self.client.get_model_version_by_alias(self.model_name, target_alias)
        except Exception as e:
            raise ModelNotLoadedError(
                f"Failed to query MLflow registry for model '{self.model_name}' alias '{target_alias}': {e}",
                details={"tracking_uri": self.tracking_uri, "model_name": self.model_name, "alias": target_alias},
            ) from e

        if mv is None:
            raise ModelNotLoadedError(
                f"No model version found for alias '{target_alias}' in registry '{self.model_name}'.",
                details={"model_name": self.model_name, "alias": target_alias},
            )

        model_uri = f"models:/{self.model_name}@{target_alias}"
        logger.info("Loading model pipeline from URI: %s (resolved Version: %s, Run ID: %s)", model_uri, mv.version, mv.run_id)

        try:
            pipeline = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise ModelNotLoadedError(
                f"Failed to deserialize model pipeline from URI '{model_uri}': {e}",
                details={"model_uri": model_uri, "resolved_version": str(mv.version)},
            ) from e

        # Warmup / sanity verification
        try:
            warmup_df = self._create_warmup_sample()
            warmup_preds = pipeline.predict_proba(warmup_df)
            if warmup_preds is None or warmup_preds.shape != (1, 2):
                raise ValueError(f"Warmup prediction returned unexpected shape: {warmup_preds.shape if warmup_preds is not None else None}")
        except Exception as e:
            raise ModelNotLoadedError(
                f"Model pipeline verification failed during warmup inference: {e}",
                details={"model_uri": model_uri, "error": str(e)},
            ) from e

        container = ModelContainer(
            pipeline=pipeline,
            model_name=self.model_name,
            model_version=str(mv.version),
            run_id=str(mv.run_id),
            alias=target_alias,
            load_timestamp=datetime.now(timezone.utc),
        )

        with self._lock:
            self._active_container = container

        logger.info(
            "Successfully initialized active model: Name='%s', Version=%s, Run ID='%s', Alias='%s'",
            container.model_name,
            container.model_version,
            container.run_id,
            container.alias,
        )
        return container

    def reload(self, alias: Optional[str] = None) -> Tuple[ModelContainer, ModelContainer]:
        """Safely reloads target model from MLflow registry and atomically swaps active model.

        Ensures that if loading or verification fails, the existing active model remains untouched.

        Args:
            alias: Registry alias to load. Defaults to configured champion.

        Returns:
            Tuple of (previous_container, new_container).
        """
        target_alias = alias or self.default_alias

        with self._lock:
            prev_container = self._active_container

        logger.info(
            "Initiating model reload for alias '%s'. Current active version: %s",
            target_alias,
            prev_container.model_version if prev_container else "None",
        )

        # 1. Query registry for candidate
        try:
            mv = self.client.get_model_version_by_alias(self.model_name, target_alias)
        except Exception as e:
            raise ModelReloadError(
                f"Reload failed: Could not query MLflow registry for alias '{target_alias}': {e}. Previous model retained.",
                details={"active_version": prev_container.model_version if prev_container else None},
            ) from e

        if mv is None:
            raise ModelReloadError(
                f"Reload failed: No version found for alias '{target_alias}'. Previous model retained.",
                details={"active_version": prev_container.model_version if prev_container else None},
            )

        model_uri = f"models:/{self.model_name}@{target_alias}"

        # 2. Load candidate in isolated local variable
        try:
            candidate_pipeline = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise ModelReloadError(
                f"Reload failed: Could not load candidate pipeline from '{model_uri}': {e}. Previous model retained.",
                details={"active_version": prev_container.model_version if prev_container else None},
            ) from e

        # 3. Verify candidate with warmup inference
        try:
            warmup_df = self._create_warmup_sample()
            warmup_preds = candidate_pipeline.predict_proba(warmup_df)
            if warmup_preds is None or warmup_preds.shape != (1, 2):
                raise ValueError("Invalid prediction output structure.")
        except Exception as e:
            raise ModelReloadError(
                f"Reload failed: Candidate model failed warmup verification: {e}. Previous model retained.",
                details={"active_version": prev_container.model_version if prev_container else None},
            ) from e

        candidate_container = ModelContainer(
            pipeline=candidate_pipeline,
            model_name=self.model_name,
            model_version=str(mv.version),
            run_id=str(mv.run_id),
            alias=target_alias,
            load_timestamp=datetime.now(timezone.utc),
        )

        # 4. Atomic pointer swap under lock
        with self._lock:
            old_container = self._active_container
            self._active_container = candidate_container

        logger.info(
            "MODEL RELOAD SUCCESS: Swapped active model from v%s to v%s (Run ID: %s)",
            old_container.model_version if old_container else "None",
            candidate_container.model_version,
            candidate_container.run_id,
        )
        return old_container, candidate_container

    def get_active_model(self) -> ModelContainer:
        """Retrieves currently active model container.

        Raises:
            ModelNotLoadedError: If no model has been loaded yet.
        """
        with self._lock:
            container = self._active_container

        if container is None:
            raise ModelNotLoadedError("Model is not loaded. The serving engine is not ready.")
        return container

    def is_ready(self) -> bool:
        """Checks if a valid functional model container is loaded."""
        with self._lock:
            return self._active_container is not None

    def check_and_update_if_alias_changed(self) -> bool:
        """Checks if the registry alias currently points to a different model version.

        If a version change is detected, triggers an atomic reload.

        Returns:
            bool: True if an update was performed, False otherwise.
        """
        container = self._active_container
        if container is None:
            self.load_champion_model()
            return True

        try:
            mv = self.client.get_model_version_by_alias(self.model_name, container.alias)
            if mv and str(mv.version) != str(container.model_version):
                logger.info(
                    "Detected alias version change: active v%s != registry v%s. Triggering automatic reload.",
                    container.model_version,
                    mv.version,
                )
                self.reload(alias=container.alias)
                return True
        except Exception as e:
            logger.warning("Periodic alias check encountered non-fatal error: %s", e)

        return False
