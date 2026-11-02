"""Centralized Configuration Manager using Pydantic and PyYAML."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

from src.common.exceptions import ConfigurationError


class ProjectConfig(BaseModel):
    name: str = "Enterprise MLOps Platform"
    version: str = "1.0.0"
    random_seed: int = 42


class PathsConfig(BaseModel):
    raw_data: str = "data/raw/telco_customer_churn.csv"
    processed_dir: str = "data/processed"
    reference_data: str = "data/processed/reference_dataset.parquet"
    sample_data: str = "data/samples/sample_churn_input.csv"
    models_dir: str = "models/saved"
    best_model_path: str = "models/saved/best_model_pipeline.joblib"
    reports_dir: str = "reports"
    evaluations_dir: str = "reports/evaluations"
    figures_dir: str = "reports/figures"


class SplitRatiosConfig(BaseModel):
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


class DataConfig(BaseModel):
    dataset_type: str = "real"
    allow_synthetic_fallback: bool = False
    expected_sha256: Optional[str] = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
    expected_rows: Optional[int] = 7043
    expected_cols: Optional[int] = 21
    target_column: str = "Churn"
    positive_target_value: str = "Yes"
    negative_target_value: str = "No"
    id_column: str = "customerID"
    split_ratios: SplitRatiosConfig = Field(default_factory=SplitRatiosConfig)
    stratify: bool = True


class FeaturesConfig(BaseModel):
    numerical: List[str]
    categorical: List[str]


class PreprocessingConfig(BaseModel):
    numerical_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    scaler: str = "StandardScaler"
    one_hot_drop: str = "first"
    handle_unknown: str = "ignore"


class LogisticRegressionConfig(BaseModel):
    max_iter: int = 1000
    class_weight: str = "balanced"
    solver: str = "lbfgs"
    C: float = 1.0


class RandomForestConfig(BaseModel):
    n_estimators: int = 150
    max_depth: Optional[int] = 8
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    n_jobs: int = -1


class XGBoostConfig(BaseModel):
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "logloss"
    scale_pos_weight: float = 2.76


class TuningConfig(BaseModel):
    enabled: bool = True
    n_iter: int = 15
    cv_folds: int = 5
    param_distributions: Dict[str, List[Any]]


class ModelsConfig(BaseModel):
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    tuning: TuningConfig


class TargetThresholdsConfig(BaseModel):
    roc_auc_target: float = 0.830
    pr_auc_target: float = 0.650
    f1_target: float = 0.580
    latency_p95_target_ms: float = 30.0


class BusinessCostConfig(BaseModel):
    customer_lifetime_value: float = 500.0
    intervention_cost: float = 50.0


class EvaluationConfig(BaseModel):
    target_thresholds: TargetThresholdsConfig
    business_cost: BusinessCostConfig


class MLflowAliasesConfig(BaseModel):
    champion: str = "champion"
    challenger: str = "challenger"
    candidate: str = "candidate"
    archived: str = "archived"


class MLflowGatingConfig(BaseModel):
    min_roc_auc: float = 0.830
    min_pr_auc: float = 0.650
    min_f1: float = 0.580
    max_latency_p95_ms: float = 30.0
    min_improvement_pr_auc: float = 0.005


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "mlruns"
    experiment_name: str = "customer-churn-prediction"
    registered_model_name: str = "customer-churn-classifier"
    enable_autolog: bool = False
    aliases: MLflowAliasesConfig = Field(default_factory=MLflowAliasesConfig)
    gating: MLflowGatingConfig = Field(default_factory=MLflowGatingConfig)


class ServingConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    model_alias: str = "champion"
    decision_threshold: float = 0.571
    reload_token: str = "dev-admin-secret-key-phase3"
    enable_reload_endpoint: bool = True
    auto_reload_interval_seconds: int = 0
    prediction_db_uri: str = "sqlite:///predictions.db"


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig
    preprocessing: PreprocessingConfig
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    models: ModelsConfig
    evaluation: EvaluationConfig
    serving: ServingConfig = Field(default_factory=ServingConfig)


_cached_config: Optional[AppConfig] = None


def find_project_root() -> Path:
    """Finds the root directory containing configs/ or .git."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads and validates application configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, defaults to `configs/config.yaml`.

    Returns:
        AppConfig: Validated strongly typed configuration object.
    """
    global _cached_config
    if _cached_config is not None and config_path is None:
        return _cached_config

    root = find_project_root()
    path = Path(config_path) if config_path else root / "configs" / "config.yaml"

    if not path.is_file():
        raise ConfigurationError(
            f"Configuration file not found at: {path}",
            details={"searched_path": str(path), "project_root": str(root)},
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        config = AppConfig(**raw_data)
        if config_path is None:
            _cached_config = config
        return config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML config: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error validating config schema: {e}") from e
