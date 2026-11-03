"""Delayed ground-truth feedback matching and model evaluation monitor."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.logger import get_logger
from src.monitoring.monitoring_config import MonitoringConfig

logger = get_logger(__name__)


@dataclass
class PerformanceResult:
    """Model performance evaluation outcome using delayed ground truth."""

    status: str  # "EVALUATED", "INSUFFICIENT_LABELS", "NO_LABELS"
    sample_count: int
    labeled_sample_count: int
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    brier_score: Optional[float] = None
    true_positives: Optional[int] = None
    false_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_negatives: Optional[int] = None
    performance_degraded: bool = False
    degradation_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "labeled_sample_count": self.labeled_sample_count,
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "pr_auc": round(self.pr_auc, 4) if self.pr_auc is not None else None,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else None,
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "confusion_matrix": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "tn": self.true_negatives,
                "fn": self.false_negatives,
            } if self.true_positives is not None else None,
            "performance_degraded": self.performance_degraded,
            "degradation_reason": self.degradation_reason,
            "warnings": self.warnings,
        }


class PerformanceMonitor:
    """Evaluates true model degradation when delayed ground-truth feedback is available.

    CRITICAL PRINCIPLE:
    - Never fabricates ground truth or performance metrics.
    - If labels are absent, explicitly returns status="NO_LABELS" with None metrics.
    - Strictly distinguishes feature drift from true performance degradation.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or MonitoringConfig()

    def evaluate_performance(
        self,
        merged_df: pd.DataFrame,
        prob_col: str = "churn_probability",
        actual_col: str = "actual_churn",
        decision_threshold: float = 0.571,
    ) -> PerformanceResult:
        """Calculates supervised metrics from paired prediction telemetry and ground truth."""
        total_samples = len(merged_df)

        if total_samples == 0:
            return PerformanceResult(
                status="NO_LABELS",
                sample_count=0,
                labeled_sample_count=0,
                warnings=["Zero prediction records available to evaluate."],
            )

        # Check if actual label column exists and has non-null entries
        if actual_col not in merged_df.columns:
            return PerformanceResult(
                status="NO_LABELS",
                sample_count=total_samples,
                labeled_sample_count=0,
                warnings=["Ground truth feedback column is absent."],
            )

        valid_mask = merged_df[actual_col].notna() & merged_df[prob_col].notna()
        labeled_df = merged_df[valid_mask]
        n_labeled = len(labeled_df)

        if n_labeled == 0:
            return PerformanceResult(
                status="NO_LABELS",
                sample_count=total_samples,
                labeled_sample_count=0,
                warnings=["No matching ground truth labels exist in evaluation window."],
            )

        if n_labeled < self.config.min_sample_size:
            msg = (
                f"Labeled sample size ({n_labeled}) is below min_sample_size "
                f"({self.config.min_sample_size}). Performance metrics withheld to avoid small-sample distortion."
            )
            logger.warning(msg)
            return PerformanceResult(
                status="INSUFFICIENT_LABELS",
                sample_count=total_samples,
                labeled_sample_count=n_labeled,
                warnings=[msg],
            )

        y_true = labeled_df[actual_col].astype(int).values
        y_prob = labeled_df[prob_col].astype(float).values
        y_pred = (y_prob >= decision_threshold).astype(int)

        # Ensure both classes are present for ROC-AUC / PR-AUC
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            msg = f"Only one class present in ground truth ({unique_classes}). Cannot compute ROC-AUC or PR-AUC."
            return PerformanceResult(
                status="INSUFFICIENT_LABELS",
                sample_count=total_samples,
                labeled_sample_count=n_labeled,
                warnings=[msg],
            )

        # 1. Supervised Classification Metrics
        roc_auc = float(roc_auc_score(y_true, y_prob))

        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(rec_curve, prec_curve))

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_prob))

        # 2. Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # 3. Assess True Model Degradation against configurable baselines
        degraded = False
        reasons = []
        if recall < self.config.performance_degradation_recall_threshold:
            degraded = True
            reasons.append(
                f"Recall ({recall:.3f}) fell below target ({self.config.performance_degradation_recall_threshold:.3f})"
            )
        if pr_auc < self.config.performance_degradation_pr_auc_threshold:
            degraded = True
            reasons.append(
                f"PR-AUC ({pr_auc:.3f}) fell below target ({self.config.performance_degradation_pr_auc_threshold:.3f})"
            )

        degradation_reason = "; ".join(reasons) if degraded else None

        logger.info(
            "Model performance evaluated: ROC-AUC=%.3f, PR-AUC=%.3f, Recall=%.3f, F1=%.3f, degraded=%s",
            roc_auc,
            pr_auc,
            recall,
            f1,
            degraded,
        )

        return PerformanceResult(
            status="EVALUATED",
            sample_count=total_samples,
            labeled_sample_count=n_labeled,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            brier_score=brier,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            performance_degraded=degraded,
            degradation_reason=degradation_reason,
        )
