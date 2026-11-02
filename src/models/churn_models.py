"""Model Architectures, Baselines, and Hyperparameter Optimization Module."""

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.common.config import AppConfig, get_config
from src.common.exceptions import ModelTrainingError
from src.common.logger import get_logger

logger = get_logger(__name__)


def get_logistic_regression(config: AppConfig | None = None) -> LogisticRegression:
    """Instantiates the Logistic Regression baseline model.

    Args:
        config: Application configuration.

    Returns:
        LogisticRegression: Configured baseline estimator.
    """
    cfg = config or get_config()
    lr_cfg = cfg.models.logistic_regression
    seed = cfg.project.random_seed

    return LogisticRegression(
        max_iter=lr_cfg.max_iter,
        class_weight=lr_cfg.class_weight,
        solver=lr_cfg.solver,
        C=lr_cfg.C,
        random_state=seed,
    )


def get_random_forest(config: AppConfig | None = None) -> RandomForestClassifier:
    """Instantiates the Random Forest ensemble baseline model.

    Args:
        config: Application configuration.

    Returns:
        RandomForestClassifier: Configured ensemble estimator.
    """
    cfg = config or get_config()
    rf_cfg = cfg.models.random_forest
    seed = cfg.project.random_seed

    return RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_split=rf_cfg.min_samples_split,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        class_weight=rf_cfg.class_weight,
        n_jobs=rf_cfg.n_jobs,
        random_state=seed,
    )


def get_xgboost(config: AppConfig | None = None) -> XGBClassifier:
    """Instantiates the primary production candidate: XGBoost Classifier.

    Configured with scale_pos_weight to compensate for class imbalance.

    Args:
        config: Application configuration.

    Returns:
        XGBClassifier: Configured gradient boosted decision tree estimator.
    """
    cfg = config or get_config()
    xgb_cfg = cfg.models.xgboost
    seed = cfg.project.random_seed

    return XGBClassifier(
        n_estimators=xgb_cfg.n_estimators,
        max_depth=xgb_cfg.max_depth,
        learning_rate=xgb_cfg.learning_rate,
        subsample=xgb_cfg.subsample,
        colsample_bytree=xgb_cfg.colsample_bytree,
        scale_pos_weight=xgb_cfg.scale_pos_weight,
        eval_metric=xgb_cfg.eval_metric,
        random_state=seed,
        n_jobs=-1,
    )


def tune_xgboost(
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    config: AppConfig | None = None,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Performs Hyperparameter Optimization for XGBoost using Stratified K-Fold CV.

    Optimizes on average_precision (PR-AUC) to prioritize detection of minority churn class.

    Args:
        X_train_trans: Transformed training feature matrix.
        y_train: Training binary target array.
        config: Application configuration.

    Returns:
        Tuple of (best_estimator, best_parameters).
    """
    cfg = config or get_config()
    tune_cfg = cfg.models.tuning
    seed = cfg.project.random_seed

    logger.info(
        "Starting XGBoost Hyperparameter Optimization (%d iterations, %d-fold Stratified CV)...",
        tune_cfg.n_iter,
        tune_cfg.cv_folds,
    )

    base_estimator = get_xgboost(cfg)
    base_estimator.set_params(n_jobs=1)
    param_dist = tune_cfg.param_distributions

    cv = StratifiedKFold(n_splits=tune_cfg.cv_folds, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=tune_cfg.n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=seed,
        n_jobs=2,
        verbose=0,
    )

    try:
        search.fit(X_train_trans, y_train)
    except Exception as e:
        raise ModelTrainingError(f"Hyperparameter optimization failed: {e}") from e

    logger.info("Best CV PR-AUC Score: %.4f", search.best_score_)
    logger.info("Best Hyperparameters: %s", search.best_params_)

    return search.best_estimator_, search.best_params_
