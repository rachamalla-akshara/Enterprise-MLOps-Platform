"""Model Evaluation, Metric Computation, Benchmarking, and Diagnostics Module."""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.common.config import AppConfig, get_config, find_project_root
from src.common.exceptions import EvaluationError
from src.common.logger import get_logger

logger = get_logger(__name__)


def benchmark_inference_latency(pipeline: Any, X_sample: pd.DataFrame, n_iterations: int = 100) -> Dict[str, float]:
    """Measures actual, empirical single-record inference latency.

    Avoids assuming SLAs; executes repetitive single-item queries to obtain real latency distribution.

    Args:
        pipeline: Fitted Scikit-learn/XGBoost inference pipeline.
        X_sample: DataFrame with sample records.
        n_iterations: Number of repetitive iterations.

    Returns:
        Dict: Empirical latency metrics in milliseconds (mean, p50, p95, p99).
    """
    if len(X_sample) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    single_row = X_sample.iloc[[0]]
    # Warm-up run
    for _ in range(5):
        _ = pipeline.predict_proba(single_row)

    durations = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = pipeline.predict_proba(single_row)
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)  # ms

    durations_arr = np.array(durations)
    return {
        "mean_ms": round(float(np.mean(durations_arr)), 3),
        "p50_ms": round(float(np.percentile(durations_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(durations_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(durations_arr, 99)), 3),
    }


def find_optimal_f1_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """Determines decision threshold maximizing F1 score.

    Args:
        y_true: Ground truth binary labels.
        y_probs: Predicted positive class probabilities.

    Returns:
        Tuple of (optimal_threshold, max_f1_score).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    best_f1 = float(f1_scores[best_idx])
    return best_thresh, best_f1


def evaluate_model(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str,
    config: AppConfig | None = None,
    optimal_threshold: float | None = None,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics, cost metrics, and latency benchmarks.

    Args:
        pipeline: Fitted inference pipeline.
        X_test: Test features DataFrame.
        y_test: Ground truth binary target array.
        model_name: Identifier for model logging.
        config: Application configuration.
        optimal_threshold: Optional calibrated threshold.

    Returns:
        Dict: Complete evaluation metrics dictionary.
    """
    cfg = config or get_config()
    cost_cfg = cfg.evaluation.business_cost

    try:
        y_probs = pipeline.predict_proba(X_test)[:, 1]
    except Exception as e:
        raise EvaluationError(f"Prediction failed for model '{model_name}': {e}") from e

    # Determine thresholds
    threshold = 0.5 if optimal_threshold is None else optimal_threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Standard metrics
    roc_auc = float(roc_auc_score(y_test, y_probs))
    pr_auc = float(average_precision_score(y_test, y_probs))
    brier = float(brier_score_loss(y_test, y_probs))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Business Cost Calculation
    # False Negative: Churned customer lost = Customer Lifetime Value (LTV)
    # False Positive: Unnecessary retention incentive offered = Intervention Cost
    total_cost = float((fn * cost_cfg.customer_lifetime_value) + (fp * cost_cfg.intervention_cost))

    # Actual latency benchmarking
    latency = benchmark_inference_latency(pipeline, X_test, n_iterations=100)

    opt_thresh, opt_f1 = find_optimal_f1_threshold(y_test, y_probs)

    metrics = {
        "model_name": model_name,
        "decision_threshold": round(threshold, 3),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "optimal_threshold": round(opt_thresh, 3),
        "optimal_f1": round(opt_f1, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "business_cost": {
            "customer_lifetime_value": cost_cfg.customer_lifetime_value,
            "intervention_cost": cost_cfg.intervention_cost,
            "total_cost": round(total_cost, 2),
            "avg_cost_per_customer": round(total_cost / len(y_test), 2),
        },
        "latency_benchmark": latency,
    }

    logger.info(
        "Evaluated [%s] -> ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f | Recall: %.4f | Cost: $%.2f | Latency p95: %.2f ms",
        model_name,
        roc_auc,
        pr_auc,
        f1,
        rec,
        total_cost,
        latency["p95_ms"],
    )

    return metrics


def plot_evaluation_curves(
    models_dict: Dict[str, Tuple[Any, Dict[str, Any]]],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    config: AppConfig | None = None,
) -> None:
    """Generates and saves diagnostic evaluation plots for all candidate models.

    Plots:
    - ROC Curves (multi-model comparison)
    - Precision-Recall Curves (multi-model comparison)
    - Confusion Matrix Heatmap (for best model)
    - Model Comparison Bar Chart
    """
    cfg = config or get_config()
    root = find_project_root()
    figures_dir = root / cfg.paths.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (pipe, metrics) in models_dict.items():
        y_probs = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_val = metrics["roc_auc"]
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curves Comparison (Holdout Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(figures_dir / "roc_curve_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    no_skill = float(np.sum(y_test) / len(y_test))
    ax.plot([0, 1], [no_skill, no_skill], linestyle="--", color="gray", label=f"No Skill ({no_skill:.2f})")
    for name, (pipe, metrics) in models_dict.items():
        y_probs = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = metrics["pr_auc"]
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(figures_dir / "pr_curve_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Model Comparison Bar Chart
    df_compare = []
    for name, (_, metrics) in models_dict.items():
        df_compare.append({
            "Model": name,
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "F1-Score": metrics["f1"],
            "Recall": metrics["recall"],
            "Brier Score": metrics["brier_score"],
        })
    comp_df = pd.DataFrame(df_compare)

    fig, ax = plt.subplots(figsize=(10, 5))
    comp_df.set_index("Model")[["ROC-AUC", "PR-AUC", "F1-Score", "Recall"]].plot(
        kind="bar",
        ax=ax,
        colormap="viridis",
        width=0.8,
    )
    ax.set_title("Model Performance Metrics Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim([0.0, 1.0])
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / "model_metrics_comparison.png", dpi=150)
    plt.close(fig)

    logger.info("Saved evaluation diagnostic figures to %s", figures_dir)


def plot_confusion_matrix(cm_dict: Dict[str, int], model_name: str, config: AppConfig | None = None) -> None:
    """Plots and exports confusion matrix heatmap for the selected champion model."""
    cfg = config or get_config()
    root = find_project_root()
    figures_dir = root / cfg.paths.figures_dir

    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
