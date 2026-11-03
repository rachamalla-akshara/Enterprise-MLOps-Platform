"""Retraining Recommendation Engine with threshold gating and cooldown governance."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import uuid

from src.common.logger import get_logger
from src.monitoring.data_quality import DataQualityResult
from src.monitoring.drift_detector import DatasetDriftResult
from src.monitoring.monitoring_config import MonitoringConfig
from src.monitoring.performance_monitor import PerformanceResult
from src.monitoring.prediction_monitor import PredictionDistributionResult

logger = get_logger(__name__)


@dataclass
class RetrainingRecommendation:
    """Audit record for retraining decision guidance."""

    recommendation_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    status: str  # "RECOMMENDED", "COOLDOWN_ACTIVE", "NO_ACTION", "INSUFFICIENT_EVIDENCE"
    trigger_reason: Optional[str] = None  # "FEATURE_DRIFT", "PERFORMANCE_DROP", "PREDICTION_DRIFT", "DATA_QUALITY_FAIL"
    trigger_metrics: Dict[str, Any] = field(default_factory=dict)
    cooldown_until: Optional[datetime] = None
    remediation_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status,
            "trigger_reason": self.trigger_reason,
            "trigger_metrics": self.trigger_metrics,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "remediation_steps": self.remediation_steps,
        }


class RetrainingRecommendationEngine:
    """Evaluates multi-source monitoring signals to generate controlled retraining recommendations.

    GOVERNANCE CONSTRAINTS:
    - Never automatically replaces or promotes the champion model.
    - Preserves MLflow experiment tracking, validation holdout isolation, and champion/challenger gating.
    - Prevents alert spam through configurable cooldown enforcement.
    - Never claims accuracy degradation when ground-truth labels are absent.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or MonitoringConfig()

    def evaluate_recommendation(
        self,
        model_name: str,
        model_version: str,
        drift_result: DatasetDriftResult,
        quality_result: DataQualityResult,
        prediction_result: PredictionDistributionResult,
        performance_result: PerformanceResult,
        last_recommendation_timestamp: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> RetrainingRecommendation:
        """Determines if retraining is recommended based on data drift, quality, and performance."""
        current_time = now or datetime.now(timezone.utc)
        rec_id = str(uuid.uuid4())

        triggers = []
        metrics_summary: Dict[str, Any] = {}

        # 1. Evaluate True Model Performance Degradation (Highest Priority)
        if performance_result.status == "EVALUATED" and performance_result.performance_degraded:
            triggers.append("PERFORMANCE_DROP")
            metrics_summary["performance"] = {
                "recall": performance_result.recall,
                "pr_auc": performance_result.pr_auc,
                "f1_score": performance_result.f1_score,
                "reason": performance_result.degradation_reason,
            }

        # 2. Evaluate Feature & Dataset Drift
        if drift_result.dataset_drift_detected:
            triggers.append("FEATURE_DRIFT")
            metrics_summary["feature_drift"] = {
                "drift_share": drift_result.drift_share,
                "threshold": drift_result.drift_share_threshold,
                "drifted_features": drift_result.drifted_features_count,
                "total_features": drift_result.total_features,
            }

        # 3. Evaluate Prediction Output Drift
        if prediction_result.probability_drift_detected or prediction_result.class_distribution_drift_detected:
            triggers.append("PREDICTION_DRIFT")
            metrics_summary["prediction_drift"] = {
                "ks_statistic": prediction_result.ks_statistic,
                "p_value": prediction_result.p_value,
                "predicted_churn_rate": prediction_result.predicted_churn_rate,
                "reference_churn_rate": prediction_result.reference_churn_rate,
            }

        # 4. Evaluate Severe Data Quality Failures
        if quality_result.status == "FAILED":
            triggers.append("DATA_QUALITY_FAIL")
            metrics_summary["data_quality"] = {
                "missing_share": quality_result.missing_values_share,
                "schema_mismatch": quality_result.schema_mismatch,
                "missing_columns": quality_result.missing_columns,
            }

        # If no trigger criteria met
        if not triggers:
            return RetrainingRecommendation(
                recommendation_id=rec_id,
                timestamp=current_time,
                model_name=model_name,
                model_version=model_version,
                status="NO_ACTION",
                trigger_reason=None,
                trigger_metrics={},
                remediation_steps=["Continue routine inference serving. All monitoring parameters nominal."],
            )

        primary_trigger = triggers[0]

        # 5. Check Cooldown Enforcement
        cooldown_delta = timedelta(hours=self.config.cooldown_hours)
        if last_recommendation_timestamp is not None:
            time_since_last = current_time - last_recommendation_timestamp
            if time_since_last < cooldown_delta:
                cooldown_until = last_recommendation_timestamp + cooldown_delta
                logger.info(
                    "Retraining trigger (%s) detected, but cooldown is active until %s",
                    primary_trigger,
                    cooldown_until.isoformat(),
                )
                return RetrainingRecommendation(
                    recommendation_id=rec_id,
                    timestamp=current_time,
                    model_name=model_name,
                    model_version=model_version,
                    status="COOLDOWN_ACTIVE",
                    trigger_reason=primary_trigger,
                    trigger_metrics=metrics_summary,
                    cooldown_until=cooldown_until,
                    remediation_steps=[
                        f"Retraining alert for {primary_trigger} suppressed by {self.config.cooldown_hours}h cooldown window.",
                        "Inspect underlying data pipelines while previous recommendation is addressed.",
                    ],
                )

        # 6. Active Recommendation with Governance Remediation Steps
        cooldown_until = current_time + cooldown_delta
        remediation_steps = [
            f"1. Triggered by verified operational condition: {primary_trigger}.",
            "2. Ingest latest validated dataset and perform schema validation.",
            "3. Execute reproducible model training pipeline on untouched holdout split.",
            "4. Log run, artifacts, and evaluation metrics into MLflow tracking server.",
            "5. Execute Champion vs Challenger comparison against promotion gates.",
            "6. Model promotion requires formal verification; champion is NOT replaced automatically.",
        ]

        logger.warning(
            "Retraining RECOMMENDED for model %s (v%s). Reason: %s, Metrics: %s",
            model_name,
            model_version,
            primary_trigger,
            metrics_summary,
        )

        return RetrainingRecommendation(
            recommendation_id=rec_id,
            timestamp=current_time,
            model_name=model_name,
            model_version=model_version,
            status="RECOMMENDED",
            trigger_reason=primary_trigger,
            trigger_metrics=metrics_summary,
            cooldown_until=cooldown_until,
            remediation_steps=remediation_steps,
        )
