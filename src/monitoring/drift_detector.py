"""Statistical Feature and Dataset Drift Detection Engine using SciPy and Evidently AI."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.common.logger import get_logger
from src.monitoring.data_quality import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.monitoring.monitoring_config import MonitoringConfig

logger = get_logger(__name__)

# Try importing Evidently Report and presets safely
try:
    from evidently.legacy.report import Report as EvidentlyReport
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    try:
        from evidently import Report as EvidentlyReport
        from evidently.presets import DataDriftPreset, DataQualityPreset
        EVIDENTLY_AVAILABLE = True
    except ImportError:
        EVIDENTLY_AVAILABLE = False
        logger.warning("Evidently AI could not be imported; falling back to native SciPy statistical tests.")


@dataclass
class FeatureDriftResult:
    """Individual feature statistical drift test result."""

    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    metric_name: str   # "ks", "wasserstein", "chisquare", "tvd"
    drift_score: float
    p_value: Optional[float]
    threshold: float
    drift_detected: bool
    reference_stats: Dict[str, Any] = field(default_factory=dict)
    current_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "metric_name": self.metric_name,
            "drift_score": round(self.drift_score, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
        }


@dataclass
class DatasetDriftResult:
    """Aggregated dataset-level drift evaluation."""

    status: str  # "EVALUATED", "INSUFFICIENT_DATA", "EMPTY_WINDOW"
    reference_rows: int
    current_rows: int
    total_features: int
    drifted_features_count: int
    drift_share: float
    drift_share_threshold: float
    dataset_drift_detected: bool
    feature_results: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "total_features": self.total_features,
            "drifted_features_count": self.drifted_features_count,
            "drift_share": round(self.drift_share, 4),
            "drift_share_threshold": self.drift_share_threshold,
            "dataset_drift_detected": self.dataset_drift_detected,
            "feature_results": {k: v.to_dict() for k, v in self.feature_results.items()},
            "warnings": self.warnings,
        }


class DriftDetector:
    """Evaluates statistical feature and dataset drift between reference and current windows.

    Applies statistically appropriate tests:
    - Numerical: Two-sample Kolmogorov-Smirnov test and Wasserstein distance.
    - Categorical: Chi-square test (when expected cell count >= 5) or Total Variation Distance (TVD).
    - Multi-comparison protection: Configurable drift share threshold (default 30%) and significance levels.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or MonitoringConfig()

    def evaluate_numerical_feature(
        self,
        ref_series: pd.Series,
        curr_series: pd.Series,
        feature_name: str,
    ) -> FeatureDriftResult:
        """Computes Kolmogorov-Smirnov and Wasserstein distance on numerical features."""
        ref_clean = pd.to_numeric(ref_series, errors="coerce").dropna().values
        curr_clean = pd.to_numeric(curr_series, errors="coerce").dropna().values

        ref_mean = float(np.mean(ref_clean)) if len(ref_clean) > 0 else 0.0
        ref_std = float(np.std(ref_clean)) if len(ref_clean) > 0 else 0.0
        curr_mean = float(np.mean(curr_clean)) if len(curr_clean) > 0 else 0.0
        curr_std = float(np.std(curr_clean)) if len(curr_clean) > 0 else 0.0

        ref_stats = {"mean": round(ref_mean, 3), "std": round(ref_std, 3), "count": len(ref_clean)}
        curr_stats = {"mean": round(curr_mean, 3), "std": round(curr_std, 3), "count": len(curr_clean)}

        if len(ref_clean) < 5 or len(curr_clean) < 5:
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type="numerical",
                metric_name="ks",
                drift_score=0.0,
                p_value=1.0,
                threshold=self.config.significance_level,
                drift_detected=False,
                reference_stats=ref_stats,
                current_stats=curr_stats,
            )

        # 1. Kolmogorov-Smirnov Two-Sample Test
        ks_res = stats.ks_2samp(ref_clean, curr_clean)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 2. Wasserstein Distance
        w_dist = float(stats.wasserstein_distance(ref_clean, curr_clean))

        # Decision rule: p_value < alpha flags drift
        drift_detected = p_val < self.config.significance_level

        ref_stats["median"] = float(np.median(ref_clean))
        curr_stats["median"] = float(np.median(curr_clean))
        curr_stats["wasserstein_distance"] = round(w_dist, 4)

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            metric_name="ks",
            drift_score=ks_stat,
            p_value=p_val,
            threshold=self.config.significance_level,
            drift_detected=drift_detected,
            reference_stats=ref_stats,
            current_stats=curr_stats,
        )

    def evaluate_categorical_feature(
        self,
        ref_series: pd.Series,
        curr_series: pd.Series,
        feature_name: str,
    ) -> FeatureDriftResult:
        """Computes Chi-square goodness-of-fit or Total Variation Distance on categorical features."""
        ref_clean = ref_series.dropna().astype(str)
        curr_clean = curr_series.dropna().astype(str)

        all_cats = sorted(list(set(ref_clean.unique()).union(set(curr_clean.unique()))))

        ref_counts = ref_clean.value_counts()
        curr_counts = curr_clean.value_counts()

        ref_dist = {cat: int(ref_counts.get(cat, 0)) for cat in all_cats}
        curr_dist = {cat: int(curr_counts.get(cat, 0)) for cat in all_cats}

        n_ref = len(ref_clean)
        n_curr = len(curr_clean)

        ref_props = {cat: ref_dist[cat] / n_ref if n_ref > 0 else 0.0 for cat in all_cats}
        curr_props = {cat: curr_dist[cat] / n_curr if n_curr > 0 else 0.0 for cat in all_cats}

        # Calculate Total Variation Distance: 0.5 * sum(|p_i - q_i|)
        tvd = 0.5 * sum(abs(ref_props[cat] - curr_props[cat]) for cat in all_cats)

        ref_stats = {"proportions": {k: round(v, 4) for k, v in ref_props.items()}, "count": n_ref}
        curr_stats = {"proportions": {k: round(v, 4) for k, v in curr_props.items()}, "count": n_curr, "tvd": round(tvd, 4)}

        # Check chi-square applicability (expected count >= 5 for all bins)
        expected_counts = np.array([ref_props[cat] * n_curr for cat in all_cats])
        observed_counts = np.array([curr_dist[cat] for cat in all_cats])

        if np.all(expected_counts >= 5.0) and n_curr >= 20:
            # Expected frequencies must sum to sum of observed for standard chisquare test
            expected_counts = expected_counts * (np.sum(observed_counts) / np.sum(expected_counts))
            try:
                chi2_stat, p_val = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
                drift_detected = float(p_val) < self.config.significance_level
                return FeatureDriftResult(
                    feature_name=feature_name,
                    feature_type="categorical",
                    metric_name="chisquare",
                    drift_score=float(chi2_stat),
                    p_value=float(p_val),
                    threshold=self.config.significance_level,
                    drift_detected=drift_detected,
                    reference_stats=ref_stats,
                    current_stats=curr_stats,
                )
            except Exception as e:
                logger.warning("Chi-square calculation failed for %s (%s); falling back to TVD", feature_name, e)

        # Fallback to Total Variation Distance when small cells invalidate Chi-Square
        tvd_threshold = 0.15  # 15% TVD threshold
        drift_detected = tvd > tvd_threshold
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            metric_name="tvd",
            drift_score=float(tvd),
            p_value=None,
            threshold=tvd_threshold,
            drift_detected=drift_detected,
            reference_stats=ref_stats,
            current_stats=curr_stats,
        )

    def evaluate_dataset_drift(
        self,
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DatasetDriftResult:
        """Evaluates overall dataset and per-feature drift across the monitoring window."""
        target_features = features or [c for c in ALL_FEATURES if c in ref_df.columns]

        # 1. Edge case handling: Empty window or insufficient observations
        if curr_df.empty:
            return DatasetDriftResult(
                status="EMPTY_WINDOW",
                reference_rows=len(ref_df),
                current_rows=0,
                total_features=len(target_features),
                drifted_features_count=0,
                drift_share=0.0,
                drift_share_threshold=self.config.drift_share_threshold,
                dataset_drift_detected=False,
                warnings=["Current monitoring window contains 0 observations."],
            )

        if len(curr_df) < self.config.min_sample_size:
            msg = (
                f"Current monitoring window has {len(curr_df)} observations, "
                f"which is below min_sample_size ({self.config.min_sample_size})."
            )
            logger.warning(msg)
            return DatasetDriftResult(
                status="INSUFFICIENT_DATA",
                reference_rows=len(ref_df),
                current_rows=len(curr_df),
                total_features=len(target_features),
                drifted_features_count=0,
                drift_share=0.0,
                drift_share_threshold=self.config.drift_share_threshold,
                dataset_drift_detected=False,
                warnings=[msg],
            )

        feature_results: Dict[str, FeatureDriftResult] = {}
        drifted_count = 0
        evaluated_count = 0

        for feat in target_features:
            if feat not in ref_df.columns or feat not in curr_df.columns:
                continue

            evaluated_count += 1
            if feat in NUMERICAL_FEATURES:
                res = self.evaluate_numerical_feature(ref_df[feat], curr_df[feat], feat)
            else:
                res = self.evaluate_categorical_feature(ref_df[feat], curr_df[feat], feat)

            feature_results[feat] = res
            if res.drift_detected:
                drifted_count += 1

        drift_share = float(drifted_count / evaluated_count) if evaluated_count > 0 else 0.0
        dataset_drift = drift_share >= self.config.drift_share_threshold

        logger.info(
            "Dataset drift evaluated: %d/%d features drifted (drift_share=%.3f, threshold=%.3f, drift_detected=%s)",
            drifted_count,
            evaluated_count,
            drift_share,
            self.config.drift_share_threshold,
            dataset_drift,
        )

        return DatasetDriftResult(
            status="EVALUATED",
            reference_rows=len(ref_df),
            current_rows=len(curr_df),
            total_features=evaluated_count,
            drifted_features_count=drifted_count,
            drift_share=drift_share,
            drift_share_threshold=self.config.drift_share_threshold,
            dataset_drift_detected=dataset_drift,
            feature_results=feature_results,
        )
