"""Machine Learning Models, Training, Evaluation, and Explainability Package."""

from src.models.churn_models import (
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
    tune_xgboost,
)
from src.models.evaluate import (
    evaluate_model,
    benchmark_inference_latency,
    find_optimal_f1_threshold,
    plot_evaluation_curves,
    plot_confusion_matrix,
)
from src.models.explainability import generate_shap_explanations
from src.models.train import run_training_pipeline, prepare_features_and_target

__all__ = [
    "get_logistic_regression",
    "get_random_forest",
    "get_xgboost",
    "tune_xgboost",
    "evaluate_model",
    "benchmark_inference_latency",
    "find_optimal_f1_threshold",
    "plot_evaluation_curves",
    "plot_confusion_matrix",
    "generate_shap_explanations",
    "run_training_pipeline",
    "prepare_features_and_target",
]
