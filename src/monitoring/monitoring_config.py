"""Configuration schema and defaults for statistical drift detection and model monitoring."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.config import AppConfig, find_project_root, get_config


class MonitoringConfig(BaseModel):
    """Configuration for Evidently-based drift detection and performance monitoring."""

    reference_data_path: str = "data/processed/reference_dataset.parquet"
    monitoring_db_uri: str = "sqlite:///predictions.db"
    reports_dir: str = "reports/monitoring"
    min_sample_size: int = 50
    drift_share_threshold: float = 0.30
    significance_level: float = 0.05
    numerical_method: str = "ks"
    categorical_method: str = "chisquare"
    prediction_drift_threshold: float = 0.10
    data_quality_missing_threshold: float = 0.05
    performance_degradation_recall_threshold: float = 0.60
    performance_degradation_pr_auc_threshold: float = 0.55
    cooldown_hours: int = 24
    retention_days: int = 30
    buffer_max_size: int = 5000
    buffer_retention_hours: int = 72

    @classmethod
    def from_app_config(cls, app_config: Optional[AppConfig] = None) -> "MonitoringConfig":
        """Constructs MonitoringConfig from the centralized AppConfig."""
        cfg = app_config or get_config()
        m_cfg = cfg.monitoring
        return cls(
            reference_data_path=m_cfg.reference_data_path,
            monitoring_db_uri=m_cfg.monitoring_db_uri,
            reports_dir=m_cfg.reports_dir,
            min_sample_size=m_cfg.min_sample_size,
            drift_share_threshold=m_cfg.drift_share_threshold,
            significance_level=m_cfg.significance_level,
            numerical_method=m_cfg.numerical_method,
            categorical_method=m_cfg.categorical_method,
            prediction_drift_threshold=m_cfg.prediction_drift_threshold,
            data_quality_missing_threshold=m_cfg.data_quality_missing_threshold,
            performance_degradation_recall_threshold=m_cfg.performance_degradation_recall_threshold,
            performance_degradation_pr_auc_threshold=m_cfg.performance_degradation_pr_auc_threshold,
            cooldown_hours=m_cfg.cooldown_hours,
            retention_days=m_cfg.retention_days,
            buffer_max_size=m_cfg.buffer_max_size,
            buffer_retention_hours=m_cfg.buffer_retention_hours,
        )
