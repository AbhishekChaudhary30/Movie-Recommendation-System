"""Custom exceptions for Movie Recommendation System."""

class MovieRecommendationError(Exception):
    """Base exception for all domain errors."""


class DataLoadError(MovieRecommendationError):
    """Raised when file loading fails or is invalid."""


class ValidationError(MovieRecommendationError):
    """Raised when dataset schema or essential data fails validation."""


class DataCleaningError(MovieRecommendationError):
    """Raised when an error occurs during data cleaning pipeline."""


class RecommendationEngineError(MovieRecommendationError):
    """Raised when recommendation computation or query fails."""


class InsufficientFeaturesError(RecommendationEngineError):
    """Raised when a dataset or selected movie lacks content features."""
