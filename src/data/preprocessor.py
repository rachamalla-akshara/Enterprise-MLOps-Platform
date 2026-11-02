"""Feature Preprocessing and Transformation Pipeline Module."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import AppConfig, get_config
from src.common.exceptions import PreprocessingError
from src.common.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor(config: AppConfig | None = None) -> ColumnTransformer:
    """Builds a Scikit-Learn ColumnTransformer for numerical and categorical features.

    Numerical pipeline: SimpleImputer(median) -> StandardScaler
    Categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(drop='first', handle_unknown='ignore')

    Args:
        config: Application configuration.

    Returns:
        ColumnTransformer: Configured preprocessor object.
    """
    cfg = config or get_config()
    num_cols = cfg.features.numerical
    cat_cols = cfg.features.categorical

    # Numerical Transformer Pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.preprocessing.numerical_imputer_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Transformer Pipeline
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.preprocessing.categorical_imputer_strategy)),
            (
                "encoder",
                OneHotEncoder(
                    drop=cfg.preprocessing.one_hot_drop,
                    handle_unknown=cfg.preprocessing.handle_unknown,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Constructed ColumnTransformer: %d numerical features, %d categorical features.",
        len(num_cols),
        len(cat_cols),
    )
    return preprocessor


def build_full_pipeline(estimator: object, config: AppConfig | None = None) -> Pipeline:
    """Encapsulates the preprocessor and estimator into an end-to-end inference Pipeline.

    This guarantees that during serving, the API can accept raw feature dictionaries/JSON
    and produce calibrated prediction probabilities without disparate preprocessing code.

    Args:
        estimator: A Scikit-learn or XGBoost classifier instance.
        config: Application configuration.

    Returns:
        Pipeline: Unified Pipeline(steps=[('preprocessor', ...), ('classifier', ...)]).
    """
    preprocessor = build_preprocessor(config)
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    return full_pipeline


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-readable transformed feature names from a fitted ColumnTransformer.

    Args:
        fitted_preprocessor: Preprocessor that has already executed .fit().

    Returns:
        List[str]: Transformed output column names.
    """
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception as e:
        logger.warning("Could not extract feature names via get_feature_names_out: %s", e)
        # Fallback manual reconstruction
        names = []
        try:
            num_cols = fitted_preprocessor.transformers_[0][2]
            names.extend(num_cols)
            cat_encoder = fitted_preprocessor.transformers_[1][1].named_steps["encoder"]
            cat_cols = fitted_preprocessor.transformers_[1][2]
            cat_names = cat_encoder.get_feature_names_out(cat_cols)
            names.extend(cat_names)
        except Exception:
            pass
        return names
