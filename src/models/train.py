"""Master Model Training, Tuning, Evaluation, and Serialization Pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.common.config import AppConfig, get_config, find_project_root
from src.common.exceptions import ModelTrainingError
from src.common.logger import get_logger
from src.data.eda import run_eda
from src.data.loader import ingest_dataset, split_data
from src.data.preprocessor import build_full_pipeline, build_preprocessor
from src.data.validator import validate_dataframe
from src.models.churn_models import (
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
    tune_xgboost,
)
from src.models.evaluate import (
    evaluate_model,
    find_optimal_f1_threshold,
    plot_confusion_matrix,
    plot_evaluation_curves,
)
from src.models.explainability import generate_shap_explanations

logger = get_logger(__name__)


def prepare_features_and_target(
    df: pd.DataFrame,
    config: AppConfig | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Separates feature matrix X from binary target vector y.

    Args:
        df: Input DataFrame containing features and target.
        config: Application configuration.

    Returns:
        Tuple of (X_features, y_binary).
    """
    cfg = config or get_config()
    target_col = cfg.data.target_column
    id_col = cfg.data.id_column

    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
    X = df[feature_cols].copy()

    if target_col in df.columns:
        y = (df[target_col] == cfg.data.positive_target_value).astype(int).values
    else:
        y = np.zeros(len(df), dtype=int)

    return X, y


def generate_markdown_report(
    eval_results: Dict[str, Dict[str, Any]],
    champion_name: str,
    shap_results: Dict[str, Any],
    config: AppConfig,
    report_path: Path,
) -> None:
    """Generates an executive markdown training and evaluation report."""
    champ_metrics = eval_results[champion_name]

    lines = [
        "# Enterprise MLOps Platform — Model Training & Evaluation Report",
        "",
        f"**Project**: {config.project.name}  ",
        f"**Random Seed**: `{config.project.random_seed}`  ",
        f"**Selected Champion Model**: `{champion_name}`  ",
        "",
        "---",
        "",
        "## 1. Candidate Model Benchmark Comparison (Holdout Test Set)",
        "",
        "| Model Name | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Brier Score | Latency p95 (ms) | Total Cost ($) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, m in eval_results.items():
        is_champ = " *(Champion)*" if model_name == champion_name else ""
        cost_val = m["business_cost"]["total_cost"]
        lat_val = m["latency_benchmark"]["p95_ms"]
        lines.append(
            f"| **{model_name}**{is_champ} | **{m['roc_auc']:.4f}** | **{m['pr_auc']:.4f}** | "
            f"{m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['brier_score']:.4f} | {lat_val:.2f} ms | ${cost_val:,.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        f"## 2. Champion Model Deep-Dive: `{champion_name}`",
        "",
        f"- **Optimal Decision Cutoff**: `{champ_metrics['optimal_threshold']}` (Yields Calibrated F1: `{champ_metrics['optimal_f1']:.4f}`)",
        f"- **Confusion Matrix**: TN=`{champ_metrics['confusion_matrix']['tn']}`, FP=`{champ_metrics['confusion_matrix']['fp']}`, FN=`{champ_metrics['confusion_matrix']['fn']}`, TP=`{champ_metrics['confusion_matrix']['tp']}`",
        f"- **Business Cost Breakdown**: ${champ_metrics['business_cost']['total_cost']:,.2f} total across test partition (${champ_metrics['business_cost']['avg_cost_per_customer']:.2f} / customer)",
        f"- **Actual Inference Latency**: Mean=`{champ_metrics['latency_benchmark']['mean_ms']:.2f} ms`, p50=`{champ_metrics['latency_benchmark']['p50_ms']:.2f} ms`, p95=`{champ_metrics['latency_benchmark']['p95_ms']:.2f} ms`",
        "",
        "---",
        "",
        "## 3. SHAP Explainability & Top Churn Risk Factors",
        "",
        "Top features driving customer churn risk based on mean absolute SHAP values:",
        "",
        "| Rank | Feature | Mean |SHAP Value| Impact on Churn Risk |",
        "| :---: | :--- | :---: | :--- |",
    ])

    top_feats = shap_results.get("top_features", [])[:8]
    for idx, item in enumerate(top_feats, start=1):
        lines.append(f"| {idx} | `{item['feature']}` | {item['mean_abs_shap']:.4f} | High influence on predicted probability |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Model Artifacts & Diagnostics Generated",
        "",
        "- Serialized Champion Pipeline: `models/saved/best_model_pipeline.joblib`",
        "- Reference Drift Baseline: `data/processed/reference_dataset.parquet`",
        "- Diagnostic Curves: `reports/figures/roc_curve_comparison.png`, `pr_curve_comparison.png`",
        "- Confusion Matrix Heatmap: `reports/figures/confusion_matrix.png`",
        "- SHAP Visualizations: `reports/figures/shap_summary.png`, `reports/figures/shap_importance_bar.png`",
        "- Metrics JSON Database: `reports/evaluations/training_metrics.json`",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Exported comprehensive training markdown report to: %s", report_path)


def run_training_pipeline(config: AppConfig | None = None) -> Dict[str, Any]:
    """Executes the full end-to-end training pipeline.

    Steps:
    1. Ingestion & Validation
    2. Exploratory Data Analysis
    3. Stratified Train / Val / Test Partitioning
    4. Reference Dataset Export (for Phase 5 Drift)
    5. Baseline & XGBoost Model Training
    6. Hyperparameter Optimization (RandomizedSearchCV)
    7. Multi-Metric Evaluation & Actual Latency Benchmarking
    8. Diagnostic Plotting
    9. Champion Selection & Serialization
    10. SHAP Explainability & Final Reporting
    """
    cfg = config or get_config()
    root = find_project_root()

    models_dir = root / cfg.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = root / cfg.paths.evaluations_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("STARTING PHASE 1 TRAINING PIPELINE: %s", cfg.project.name)
    logger.info("================================================================================")

    # 1. Ingestion
    raw_df = ingest_dataset(cfg)

    # 2. Validation
    clean_df, val_report = validate_dataframe(raw_df, cfg)

    # 3. EDA
    eda_summary = run_eda(clean_df, cfg)

    # 4. Stratified Split (Zero leakage)
    train_df, val_df, test_df = split_data(clean_df, cfg)

    # 5. Feature extraction
    X_train, y_train = prepare_features_and_target(train_df, cfg)
    X_val, y_val = prepare_features_and_target(val_df, cfg)
    X_test, y_test = prepare_features_and_target(test_df, cfg)

    # Export sample test payload for API integration
    sample_path = root / cfg.paths.sample_data
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    X_test.head(10).to_csv(sample_path, index=False)
    logger.info("Exported sample test payload (10 records) to %s", sample_path)

    # Export reference dataset (parquet) for future drift monitoring
    ref_path = root / cfg.paths.reference_data
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(ref_path, index=False)
    logger.info("Exported baseline reference dataset (%d records) to %s", len(X_train), ref_path)

    # 6. Build Preprocessor and fit strictly on X_train
    preprocessor = build_preprocessor(cfg)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # 7. Model Suite Training
    models_to_evaluate: Dict[str, Pipeline] = {}

    # Model 1: Logistic Regression Baseline
    logger.info("Training Logistic Regression Baseline...")
    lr_est = get_logistic_regression(cfg)
    lr_est.fit(X_train_trans, y_train)
    lr_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", lr_est)])
    models_to_evaluate["Logistic_Regression"] = lr_pipe

    # Model 2: Random Forest Classifier
    logger.info("Training Random Forest Classifier...")
    rf_est = get_random_forest(cfg)
    rf_est.fit(X_train_trans, y_train)
    rf_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", rf_est)])
    models_to_evaluate["Random_Forest"] = rf_pipe

    # Model 3: XGBoost Classifier Baseline
    logger.info("Training XGBoost Baseline...")
    xgb_baseline = get_xgboost(cfg)
    xgb_baseline.fit(X_train_trans, y_train)
    xgb_base_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", xgb_baseline)])
    models_to_evaluate["XGBoost_Baseline"] = xgb_base_pipe

    # Model 4: XGBoost with Hyperparameter Optimization
    if cfg.models.tuning.enabled:
        logger.info("Tuning XGBoost Hyperparameters via RandomizedSearchCV...")
        best_xgb_est, best_params = tune_xgboost(X_train_trans, y_train, cfg)
        xgb_tuned_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", best_xgb_est)])
        models_to_evaluate["XGBoost_Tuned"] = xgb_tuned_pipe
    else:
        best_params = {}

    # 8. Evaluation on Validation Set and Test Set
    logger.info("Evaluating models on Holdout Test Set (%d samples)...", len(X_test))
    eval_results: Dict[str, Dict[str, Any]] = {}
    pipe_dict_for_plots: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

    for name, pipe in models_to_evaluate.items():
        # Evaluate on validation set to find optimal threshold
        val_probs = pipe.predict_proba(X_val)[:, 1]
        opt_thresh, _ = find_optimal_f1_threshold(y_val, val_probs)

        # Evaluate on holdout test set using both 0.5 and optimal threshold
        metrics = evaluate_model(pipe, X_test, y_test, model_name=name, config=cfg, optimal_threshold=opt_thresh)
        eval_results[name] = metrics
        pipe_dict_for_plots[name] = (pipe, metrics)

    # 9. Diagnostic Curves
    plot_evaluation_curves(pipe_dict_for_plots, X_test, y_test, cfg)

    # 10. Select Champion Model
    # Champion criterion: highest PR-AUC and lowest Business Cost on Holdout Test
    ranked_models = sorted(
        eval_results.items(),
        key=lambda item: (item[1]["pr_auc"], -item[1]["business_cost"]["total_cost"]),
        reverse=True,
    )
    champion_name, champion_metrics = ranked_models[0]
    champion_pipeline = models_to_evaluate[champion_name]
    logger.info("Selected Champion Model: [%s] (PR-AUC: %.4f, ROC-AUC: %.4f)", champion_name, champion_metrics["pr_auc"], champion_metrics["roc_auc"])

    # Plot Confusion Matrix for Champion
    plot_confusion_matrix(champion_metrics["confusion_matrix"], champion_name, cfg)

    # 11. SHAP Explainability
    shap_results = generate_shap_explanations(champion_pipeline, X_test, cfg, sample_size=300)

    # 12. Save Serialized Champion Artifact
    best_model_file = root / cfg.paths.best_model_path
    best_model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, best_model_file)
    logger.info("Saved serialized Champion model pipeline to: %s", best_model_file)

    # 13. Export Training Metrics & Markdown Report
    summary_output = {
        "project": cfg.project.name,
        "champion_model": champion_name,
        "champion_metrics": champion_metrics,
        "all_models_evaluated": eval_results,
        "shap_summary": shap_results,
    }

    metrics_json_path = eval_dir / "training_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)

    report_md_path = eval_dir / "training_report.md"
    generate_markdown_report(eval_results, champion_name, shap_results, cfg, report_md_path)

    logger.info("================================================================================")
    logger.info("PHASE 1 TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Champion Model: %s", champion_name)
    logger.info("Holdout PR-AUC: %.4f | ROC-AUC: %.4f | F1: %.4f", champion_metrics["pr_auc"], champion_metrics["roc_auc"], champion_metrics["f1"])
    logger.info("Report: %s", report_md_path)
    logger.info("================================================================================")

    return summary_output


if __name__ == "__main__":
    run_training_pipeline()
