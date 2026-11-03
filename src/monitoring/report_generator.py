"""Report Generator producing interactive HTML dashboards and machine-readable JSON reports."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.common.config import find_project_root
from src.common.logger import get_logger
from src.monitoring.data_quality import DataQualityResult
from src.monitoring.drift_detector import DatasetDriftResult
from src.monitoring.monitoring_config import MonitoringConfig
from src.monitoring.performance_monitor import PerformanceResult
from src.monitoring.prediction_monitor import PredictionDistributionResult
from src.monitoring.retraining_recommendations import RetrainingRecommendation

logger = get_logger(__name__)

# Safe Evidently imports
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


class ReportGenerator:
    """Produces interactive HTML reports and comprehensive JSON summaries."""

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or MonitoringConfig()
        self.root = find_project_root()
        self.reports_dir = (self.root / self.config.reports_dir).resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(
        self,
        report_id: str,
        timestamp: datetime,
        model_name: str,
        model_version: str,
        drift_result: DatasetDriftResult,
        quality_result: DataQualityResult,
        prediction_result: PredictionDistributionResult,
        performance_result: PerformanceResult,
        recommendation: RetrainingRecommendation,
    ) -> Path:
        """Constructs and saves the machine-readable JSON monitoring report."""
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_dir / f"monitoring_report_{ts_str}_{report_id[:8]}.json"

        report_payload = {
            "metadata": {
                "report_id": report_id,
                "timestamp": timestamp.isoformat(),
                "model_name": model_name,
                "model_version": model_version,
                "config": {
                    "drift_share_threshold": self.config.drift_share_threshold,
                    "significance_level": self.config.significance_level,
                    "min_sample_size": self.config.min_sample_size,
                    "cooldown_hours": self.config.cooldown_hours,
                },
            },
            "data_quality": quality_result.to_dict(),
            "feature_drift": drift_result.to_dict(),
            "prediction_drift": prediction_result.to_dict(),
            "model_performance": performance_result.to_dict(),
            "retraining_recommendation": recommendation.to_dict(),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        logger.info("Saved machine-readable monitoring JSON report to: %s", json_path)
        return json_path

    def generate_html_report(
        self,
        report_id: str,
        timestamp: datetime,
        model_name: str,
        model_version: str,
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        drift_result: DatasetDriftResult,
        quality_result: DataQualityResult,
        prediction_result: PredictionDistributionResult,
        performance_result: PerformanceResult,
        recommendation: RetrainingRecommendation,
    ) -> Path:
        """Constructs and saves the interactive HTML monitoring dashboard."""
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        html_path = self.reports_dir / f"monitoring_report_{ts_str}_{report_id[:8]}.html"

        # 1. Attempt to generate full Evidently HTML report
        evidently_success = False
        if EVIDENTLY_AVAILABLE and not curr_df.empty and len(curr_df) >= 5:
            try:
                # Keep common features between reference and current
                common_cols = [c for c in ref_df.columns if c in curr_df.columns]
                if common_cols:
                    rep = EvidentlyReport(metrics=[DataDriftPreset(), DataQualityPreset()])
                    rep.run(reference_data=ref_df[common_cols], current_data=curr_df[common_cols])
                    rep.save_html(str(html_path))
                    evidently_success = True
                    logger.info("Generated Evidently AI native HTML report at: %s", html_path)
            except Exception as e:
                logger.warning("Evidently HTML generation failed (%s); falling back to custom HTML dashboard", e)

        # 2. Fallback to self-contained responsive HTML dashboard
        if not evidently_success:
            html_content = self._build_custom_html_dashboard(
                report_id=report_id,
                timestamp=timestamp,
                model_name=model_name,
                model_version=model_version,
                drift_result=drift_result,
                quality_result=quality_result,
                prediction_result=prediction_result,
                performance_result=performance_result,
                recommendation=recommendation,
            )
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("Generated responsive monitoring HTML dashboard at: %s", html_path)

        return html_path

    def _build_custom_html_dashboard(
        self,
        report_id: str,
        timestamp: datetime,
        model_name: str,
        model_version: str,
        drift_result: DatasetDriftResult,
        quality_result: DataQualityResult,
        prediction_result: PredictionDistributionResult,
        performance_result: PerformanceResult,
        recommendation: RetrainingRecommendation,
    ) -> str:
        """Renders a production-grade, self-contained HTML monitoring dashboard."""
        drift_badge_color = "#e74c3c" if drift_result.dataset_drift_detected else "#2ecc71"
        drift_badge_text = "DRIFT DETECTED" if drift_result.dataset_drift_detected else "NO DRIFT"

        rec_color_map = {
            "RECOMMENDED": "#e74c3c",
            "COOLDOWN_ACTIVE": "#f39c12",
            "NO_ACTION": "#2ecc71",
            "INSUFFICIENT_EVIDENCE": "#95a5a6",
        }
        rec_color = rec_color_map.get(recommendation.status, "#34495e")

        feature_rows_html = ""
        for feat_name, res in drift_result.feature_results.items():
            status_badge = (
                '<span style="color:#e74c3c;font-weight:bold;">DRIFT</span>'
                if res.drift_detected
                else '<span style="color:#2ecc71;">STABLE</span>'
            )
            p_val_str = f"{res.p_value:.4f}" if res.p_value is not None else "N/A (TVD)"
            score_str = f"{res.drift_score:.4f}"
            feature_rows_html += f"""
            <tr>
                <td><strong>{feat_name}</strong></td>
                <td>{res.feature_type}</td>
                <td>{res.metric_name.upper()}</td>
                <td>{score_str}</td>
                <td>{p_val_str}</td>
                <td>{status_badge}</td>
            </tr>
            """

        remediation_html = "".join([f"<li>{step}</li>" for step in recommendation.remediation_steps])

        perf_html = (
            f"""
            <div class="card">
                <h3>Supervised Performance Evaluation (Ground Truth)</h3>
                <p><strong>Status:</strong> {performance_result.status}</p>
                <p><strong>ROC-AUC:</strong> {performance_result.roc_auc:.4f if performance_result.roc_auc else 'N/A'}</p>
                <p><strong>PR-AUC:</strong> {performance_result.pr_auc:.4f if performance_result.pr_auc else 'N/A'}</p>
                <p><strong>Recall:</strong> {performance_result.recall:.4f if performance_result.recall else 'N/A'}</p>
                <p><strong>F1 Score:</strong> {performance_result.f1_score:.4f if performance_result.f1_score else 'N/A'}</p>
            </div>
            """
            if performance_result.status == "EVALUATED"
            else f"""
            <div class="card">
                <h3>Supervised Performance Evaluation (Ground Truth)</h3>
                <p><strong>Status:</strong> {performance_result.status}</p>
                <p style="color:#7f8c8d;">Ground truth labels unavailable in this monitoring window. Model accuracy is not claimed to be degraded.</p>
            </div>
            """
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MLOps Monitoring Report - {model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; color: #2c3e50; margin: 0; padding: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 25px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 24px; }}
        .header p {{ margin: 0; opacity: 0.8; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .card h3 {{ margin-top: 0; font-size: 16px; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; color: white; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background-color: #f1f2f6; color: #34495e; }}
        ul {{ margin: 0; padding-left: 20px; }}
        li {{ margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Enterprise MLOps Platform — Statistical Monitoring & Drift Report</h1>
        <p>Model: <strong>{model_name}</strong> (v{model_version}) | Report ID: <code>{report_id}</code> | Timestamp: {timestamp.isoformat()}</p>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Dataset Drift Status</h3>
            <p>Status: <span class="badge" style="background-color: {drift_badge_color};">{drift_badge_text}</span></p>
            <p>Drift Share: <strong>{drift_result.drift_share * 100:.1f}%</strong> (Threshold: {drift_result.drift_share_threshold * 100:.1f}%)</p>
            <p>Drifted Features: <strong>{drift_result.drifted_features_count} / {drift_result.total_features}</strong></p>
            <p>Sample Observations: <strong>{drift_result.current_rows}</strong> (Ref: {drift_result.reference_rows})</p>
        </div>

        <div class="card">
            <h3>Prediction Output Drift</h3>
            <p>Mean Churn Probability: <strong>{prediction_result.mean_probability:.3f}</strong></p>
            <p>Predicted Churn Rate: <strong>{prediction_result.predicted_churn_rate * 100:.1f}%</strong> (Ref: {prediction_result.reference_churn_rate * 100:.1f}%)</p>
            <p>KS Statistic: <strong>{prediction_result.ks_statistic:.4f}</strong></p>
            <p>Probability Drift: <strong>{"DETECTED" if prediction_result.probability_drift_detected else "NOMINAL"}</strong></p>
        </div>

        <div class="card">
            <h3>Data Quality & Schema</h3>
            <p>Status: <strong>{quality_result.status}</strong></p>
            <p>Missing Value Rate: <strong>{quality_result.missing_values_share * 100:.2f}%</strong></p>
            <p>Out-of-Range Violations: <strong>{quality_result.out_of_range_count}</strong></p>
            <p>Unexpected Categories: <strong>{quality_result.unexpected_categories_count}</strong></p>
        </div>

        <div class="card">
            <h3>Retraining Recommendation</h3>
            <p>Status: <span class="badge" style="background-color: {rec_color};">{recommendation.status}</span></p>
            <p>Trigger Reason: <strong>{recommendation.trigger_reason or 'None'}</strong></p>
            <p>Cooldown Until: <strong>{recommendation.cooldown_until.isoformat() if recommendation.cooldown_until else 'Inactive'}</strong></p>
        </div>
    </div>

    {perf_html}

    <div class="card" style="margin-bottom: 20px;">
        <h3>Governance & Remediation Guidance</h3>
        <ul>{remediation_html}</ul>
    </div>

    <div class="card">
        <h3>Feature Drift Breakdown</h3>
        <table>
            <thead>
                <tr>
                    <th>Feature Name</th>
                    <th>Type</th>
                    <th>Metric</th>
                    <th>Score / Statistic</th>
                    <th>P-Value</th>
                    <th>Drift Status</th>
                </tr>
            </thead>
            <tbody>
                {feature_rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
