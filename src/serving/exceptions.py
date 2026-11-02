"""Custom Exceptions for Model Serving and Inference Service."""

from src.common.exceptions import MLOpsPlatformError


class ServingError(MLOpsPlatformError):
    """Base exception for all errors within the model serving subsystem."""
    pass


class ModelNotLoadedError(ServingError):
    """Raised when inference is requested but no model has been loaded or resolved."""
    pass


class ModelReloadError(ServingError):
    """Raised when loading or validating a new model during reload fails."""
    pass


class PredictionError(ServingError):
    """Raised when model inference execution encounters an unrecoverable failure."""
    pass


class AuthenticationError(ServingError):
    """Raised when an administrative or protected endpoint is accessed without valid authorization."""
    pass


class InvalidInputError(ServingError):
    """Raised when prediction input payloads fail validation constraints."""
    pass
