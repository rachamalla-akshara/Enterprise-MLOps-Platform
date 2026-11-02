"""Model Explainability Module using SHAP (SHapley Additive exPlanations)."""

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.common.config import AppConfig, get_config, find_project_root
from src.common.logger import get_logger
from src.data.preprocessor import get_feature_names

logger = get_logger(__name__)


def generate_shap_explanations(
    pipeline: Any,
    X_test: pd.DataFrame,
    config: AppConfig | None = None,
    sample_size: int = 300,
) -> Dict[str, Any]:
    """Generates SHAP global and local feature importance explanations for tree-based models.

    Args:
        pipeline: Fitted pipeline with named steps 'preprocessor' and 'classifier'.
        X_test: Raw test features DataFrame.
        config: Application configuration.
        sample_size: Number of test samples to evaluate for SHAP values.

    Returns:
        Dict: Feature importance rankings and metadata.
    """
    cfg = config or get_config()
    root = find_project_root()
    figures_dir = root / cfg.paths.figures_dir
    reports_dir = root / cfg.paths.evaluations_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP explainability on %d test samples...", min(sample_size, len(X_test)))

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform test subset
    X_sub = X_test.head(sample_size)
    X_trans = preprocessor.transform(X_sub)
    feature_names = get_feature_names(preprocessor)

    # Ensure feature names match transformed matrix dimensions
    if len(feature_names) != X_trans.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    df_trans = pd.DataFrame(X_trans, columns=feature_names)

    shap_summary: Dict[str, Any] = {}

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_trans)

        # Handle binary classification array shapes
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_matrix = shap_values[1]  # positive class (Churn=1)
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        # Compute mean absolute SHAP values per feature
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False)

        top_features = feature_importance_df.head(15).to_dict(orient="records")
        for item in top_features:
            item["mean_abs_shap"] = round(float(item["mean_abs_shap"]), 5)

        shap_summary = {
            "explainer_type": "TreeExplainer",
            "evaluated_samples": len(X_sub),
            "top_features": top_features,
        }

        # 1. SHAP Beeswarm Summary Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_matrix, df_trans, max_display=12, show=False)
        plt.title("SHAP Beeswarm Summary Plot (Customer Churn Drivers)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(figures_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. SHAP Global Feature Importance Bar Plot
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(shap_matrix, df_trans, plot_type="bar", max_display=12, show=False)
        plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(figures_dir / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Export JSON summary
        out_json = reports_dir / "shap_importances.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(shap_summary, f, indent=2)

        logger.info("Saved SHAP plots and feature importance rankings to %s", reports_dir)

    except Exception as e:
        logger.warning("SHAP explanation encountered non-fatal exception: %s", e)
        shap_summary = {"error": str(e), "top_features": []}

    return shap_summary
