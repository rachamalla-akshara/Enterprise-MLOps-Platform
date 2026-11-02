"""Exploratory Data Analysis (EDA) Module for Customer Churn."""

import json
from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.config import AppConfig, get_config, find_project_root
from src.common.logger import get_logger

logger = get_logger(__name__)


def run_eda(df: pd.DataFrame, config: AppConfig | None = None) -> Dict[str, Any]:
    """Executes exploratory data analysis and exports visualizations and metrics summary.

    Args:
        df: Cleaned and validated dataset.
        config: Application configuration.

    Returns:
        Dict: Statistical summary of findings.
    """
    cfg = config or get_config()
    root = find_project_root()
    figures_dir = root / cfg.paths.figures_dir
    reports_dir = root / cfg.paths.reports_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_col = cfg.data.target_column
    has_target = target_col in df.columns

    logger.info("Executing Exploratory Data Analysis on %d records...", len(df))

    # 1. Target distribution
    target_summary = {}
    churn_rate = 0.0
    if has_target:
        counts = df[target_col].value_counts().to_dict()
        percentages = (df[target_col].value_counts(normalize=True) * 100).to_dict()
        churn_rate = float(percentages.get(cfg.data.positive_target_value, 0.0))
        target_summary = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): round(float(v), 2) for k, v in percentages.items()},
            "churn_rate_percent": round(churn_rate, 2),
        }

    # 2. Numerical features stats
    num_cols = [c for c in cfg.features.numerical if c in df.columns]
    num_stats = {}
    for col in num_cols:
        series = df[col].dropna()
        num_stats[col] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "missing": int(df[col].isna().sum()),
        }

    # 3. Categorical features breakdown
    cat_cols = [c for c in cfg.features.categorical if c in df.columns]
    churn_by_category = {}
    if has_target:
        binary_target = (df[target_col] == cfg.data.positive_target_value).astype(int)
        for cat in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
            if cat in df.columns:
                group_rates = df.groupby(cat, observed=True).apply(
                    lambda g: float((g[target_col] == cfg.data.positive_target_value).mean() * 100)
                ).to_dict()
                churn_by_category[cat] = {str(k): round(v, 2) for k, v in group_rates.items()}

    # 4. Generate Visualizations
    try:
        sns.set_theme(style="whitegrid")

        # Plot 1: Target distribution
        if has_target:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=target_col, data=df, palette="Blues_r", ax=ax)
            ax.set_title("Customer Churn Target Distribution", fontsize=12, fontweight="bold")
            ax.set_ylabel("Customer Count")
            ax.set_xlabel("Churn Status")
            plt.tight_layout()
            fig.savefig(figures_dir / "target_distribution.png", dpi=150)
            plt.close(fig)

        # Plot 2: Numerical distributions by Churn
        if has_target and len(num_cols) >= 2:
            fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
            for i, col in enumerate(num_cols):
                sns.boxplot(x=target_col, y=col, data=df, palette="Set2", ax=axes[i])
                axes[i].set_title(f"{col} by Churn", fontweight="bold")
            plt.tight_layout()
            fig.savefig(figures_dir / "numerical_distributions.png", dpi=150)
            plt.close(fig)

        # Plot 3: Churn by Contract Type
        if has_target and "Contract" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            contract_churn = df.groupby(["Contract", target_col], observed=True).size().unstack(fill_value=0)
            contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
            contract_churn_pct.plot(kind="bar", stacked=True, color=["#3498db", "#e74c3c"], ax=ax)
            ax.set_title("Churn Rate by Contract Type (%)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Percentage")
            ax.set_xlabel("Contract")
            plt.xticks(rotation=0)
            plt.tight_layout()
            fig.savefig(figures_dir / "churn_by_contract.png", dpi=150)
            plt.close(fig)

        logger.info("Saved EDA figures to: %s", figures_dir)
    except Exception as e:
        logger.warning("Visualizations generation encountered warning: %s", e)

    eda_summary = {
        "dataset_rows": len(df),
        "features_count": len(df.columns),
        "target_distribution": target_summary,
        "numerical_statistics": num_stats,
        "churn_rate_by_category": churn_by_category,
    }

    summary_path = reports_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    logger.info("Exported EDA summary JSON to: %s", summary_path)

    return eda_summary
