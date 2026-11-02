"""Prediction Service orchestrating inference, latency measurement, and telemetry logging."""

from datetime import datetime, timezone
import time
from typing import Optional
import pandas as pd

from src.common.config import AppConfig, get_config
from src.common.logger import get_logger
from src.serving.exceptions import PredictionError
from src.serving.model_loader import ModelLoader
from src.serving.prediction_logger import PredictionLogger
from src.serving.schemas import CustomerFeatures, PredictionResponse

logger = get_logger(__name__)


class PredictionService:
    """Orchestrates model inference, probability thresholding, and audit logging."""

    def __init__(
        self,
        model_loader: ModelLoader,
        prediction_logger: PredictionLogger,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.loader = model_loader
        self.logger_db = prediction_logger
        self.cfg = config or get_config()
        self.threshold = float(self.cfg.serving.decision_threshold)

    def predict(
        self,
        features: CustomerFeatures,
        request_id: str,
    ) -> PredictionResponse:
        """Executes customer churn prediction against the active champion model.

        Args:
            features: Validated customer feature set.
            request_id: Unique request identifier.

        Returns:
            PredictionResponse: Complete response with probability, prediction, and latency.
        """
        container = self.loader.get_active_model()
        now_utc = datetime.now(timezone.utc)
        timestamp_str = now_utc.isoformat()

        # Convert validated features to single-row DataFrame
        feature_dict = features.to_inference_dict()
        input_df = pd.DataFrame([feature_dict])

        # High-precision inference timing
        t_start = time.perf_counter()
        try:
            probabilities = container.pipeline.predict_proba(input_df)
        except Exception as e:
            logger.error("Inference execution failed for request_id=%s: %s", request_id, e)
            # Log failure status to telemetry
            self.logger_db.log_prediction(
                request_id=request_id,
                model_name=container.model_name,
                model_version=container.model_version,
                prediction="UNKNOWN",
                churn_probability=0.0,
                decision_threshold=self.threshold,
                inference_latency_ms=0.0,
                request_status="INFERENCE_ERROR",
                timestamp=now_utc,
            )
            raise PredictionError(f"Model prediction failed: {e}", details={"request_id": request_id}) from e

        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000.0, 3)

        # Churn probability (class index 1: positive churn)
        churn_prob = float(probabilities[0, 1])
        prediction_label = "Churn" if churn_prob >= self.threshold else "No Churn"

        response = PredictionResponse(
            request_id=request_id,
            prediction=prediction_label,
            churn_probability=round(churn_prob, 4),
            decision_threshold=self.threshold,
            model_name=container.model_name,
            model_version=container.model_version,
            inference_latency_ms=latency_ms,
            prediction_timestamp=timestamp_str,
        )

        # Safely persist telemetry log without failing caller on logging errors
        try:
            self.logger_db.log_prediction(
                request_id=request_id,
                model_name=container.model_name,
                model_version=container.model_version,
                prediction=prediction_label,
                churn_probability=churn_prob,
                decision_threshold=self.threshold,
                inference_latency_ms=latency_ms,
                request_status="SUCCESS",
                timestamp=now_utc,
            )
        except Exception as e:
            logger.error("Failed to persist prediction telemetry for request_id=%s: %s", request_id, e)

        return response
