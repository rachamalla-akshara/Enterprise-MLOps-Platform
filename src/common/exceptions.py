"""Custom Exception Hierarchy for Enterprise MLOps Platform."""


class MLOpsPlatformError(Exception):
    """Base exception for all domain errors within the Enterprise MLOps Platform."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(MLOpsPlatformError):
    """Raised when configuration validation or file loading fails."""
    pass


class DataIngestionError(MLOpsPlatformError):
    """Raised when dataset loading, extraction, or raw source access fails."""
    pass


class DataValidationError(MLOpsPlatformError):
    """Raised when dataset fails contract, schema, or integrity constraints."""
    pass


class PreprocessingError(MLOpsPlatformError):
    """Raised when data transformation, imputation, or encoding fails."""
    pass


class ModelTrainingError(MLOpsPlatformError):
    """Raised when model training, hyperparameter tuning, or fitting encounters a failure."""
    pass


class EvaluationError(MLOpsPlatformError):
    """Raised when model evaluation, metric computation, or benchmark checking fails."""
    pass


class ModelArtifactError(MLOpsPlatformError):
    """Raised when saving or loading serialized model artifacts fails."""
    pass
