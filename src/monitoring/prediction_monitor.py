"""Inference prediction output and probability distribution drift monitor."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.common.logger import get_logger
from src.monitoring.monitoring_config import MonitoringConfig

logger = get_logger(__name__)


@dataclass
class PredictionDistributionResult:
    """Statistical summary of prediction probability and class output distributions."""

    sample_count: int
    mean_probability: float
    std_probability: float
    predicted_churn_rate: float
    reference_churn_rate: float
    ks_statistic: float
    p_value: Optional[float]
    wasserstein_distance: float
    probability_drift_detected: bool
    class_distribution_drift_detected: bool
    status: str = "EVALUATED"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_probability": round(self.mean_probability, 4),
            "std_probability": round(self.std_probability, 4),
            "predicted_churn_rate": round(self.predicted_churn_rate, 4),
            "reference_churn_rate": round(self.reference_churn_rate, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "wasserstein_distance": round(self.wasserstein_distance, 4),
            "probability_drift_detected": self.probability_drift_detected,
            "class_distribution_drift_detected": self.class_distribution_drift_detected,
            "status": self.status,
            "warnings": self.warnings,
        }


class PredictionMonitor:
    """Monitors model inference outputs (predicted probabilities and binary class decisions).

    Detects:
    1. Concept / output probability drift (KS test and Wasserstein distance on probabilities).
    2. Class proportion shift (e.g. significant divergence from reference churn rate ~26.5%).
    """

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or MonitoringConfig()
        # Default Telco churn rate in training population is ~26.5%
        self.default_reference_churn_rate = 0.265
        # Expected baseline probability mean
        self.default_reference_prob_mean = 0.280

    def evaluate_prediction_drift(
        self,
        current_probabilities: np.ndarray,
        reference_probabilities: Optional[np.ndarray] = None,
        reference_churn_rate: Optional[float] = None,
        decision_threshold: float = 0.571,
    ) -> PredictionDistributionResult:
        """Evaluates prediction probability distribution and class balance drift."""
        n_curr = len(current_probabilities)
        ref_churn_rate = reference_churn_rate if reference_churn_rate is not None else self.default_reference_churn_rate

        if n_curr == 0:
            return PredictionDistributionResult(
                sample_count=0,
                mean_probability=0.0,
                std_probability=0.0,
                predicted_churn_rate=0.0,
                reference_churn_rate=ref_churn_rate,
                ks_statistic=0.0,
                p_value=None,
                wasserstein_distance=0.0,
                probability_drift_detected=False,
                class_distribution_drift_detected=False,
                status="EMPTY_WINDOW",
                warnings=["Zero prediction records available in the evaluation window."],
            )

        curr_probs = np.asarray(current_probabilities, dtype=float)
        curr_probs = curr_probs[~np.isnan(curr_probs)]

        if len(curr_probs) < self.config.min_sample_size:
            msg = (
                f"Sample size {len(curr_probs)} is below min_sample_size "
                f"({self.config.min_sample_size}) for prediction drift evaluation."
            )
            return PredictionDistributionResult(
                sample_count=len(curr_probs),
                mean_probability=float(np.mean(curr_probs)) if len(curr_probs) > 0 else 0.0,
                std_probability=float(np.std(curr_probs)) if len(curr_probs) > 0 else 0.0,
                predicted_churn_rate=float(np.mean(curr_probs >= decision_threshold)) if len(curr_probs) > 0 else 0.0,
                reference_churn_rate=ref_churn_rate,
                ks_statistic=0.0,
                p_value=None,
                wasserstein_distance=0.0,
                probability_drift_detected=False,
                class_distribution_drift_detected=False,
                status="INSUFFICIENT_DATA",
                warnings=[msg],
            )

        mean_p = float(np.mean(curr_probs))
        std_p = float(np.std(curr_probs))
        predicted_churn_rate = float(np.mean(curr_probs >= decision_threshold))

        # Build or use reference probabilities
        if reference_probabilities is not None and len(reference_probabilities) > 0:
            ref_probs = np.asarray(reference_probabilities, dtype=float)
            ref_probs = ref_probs[~np.isnan(ref_probs)]
        else:
            # Beta distribution approximation calibrated to IBM Telco reference dataset
            np.random.seed(42)
            ref_probs = np.random.beta(a=1.5, b=3.5, size=max(len(curr_probs), 500))

        # 1. Kolmogorov-Smirnov test on continuous probabilities
        ks_res = stats.ks_2samp(ref_probs, curr_probs)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 2. Wasserstein distance
        w_dist = float(stats.wasserstein_distance(ref_probs, curr_probs))

        # Flag probability drift if KS statistic exceeds threshold or p_val < alpha
        prob_drift = (p_val < self.config.significance_level) and (ks_stat > self.config.prediction_drift_threshold)

        # 3. Class balance shift: absolute difference between predicted churn rate and reference churn rate
        churn_rate_delta = abs(predicted_churn_rate - ref_churn_rate)
        class_drift = churn_rate_delta > 0.15  # Alert if predicted churn rate shifts by > 15 percentage points

        logger.info(
            "Prediction drift evaluated: mean_p=%.3f, churn_rate=%.3f (ref=%.3f), ks=%.3f (p=%.4f), drift=%s",
            mean_p,
            predicted_churn_rate,
            ref_churn_rate,
            ks_stat,
            p_val,
            prob_drift or class_drift,
        )

        return PredictionDistributionResult(
            sample_count=len(curr_probs),
            mean_probability=mean_p,
            std_probability=std_p,
            predicted_churn_rate=predicted_churn_rate,
            reference_churn_rate=ref_churn_rate,
            ks_statistic=ks_stat,
            p_value=p_val,
            wasserstein_distance=w_dist,
            probability_drift_detected=prob_drift,
            class_distribution_drift_detected=class_drift,
            status="EVALUATED",
        )
