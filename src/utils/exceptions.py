"""Custom exceptions for Movie Recommendation System."""

class MovieRecommendationError(Exception):
    """Base exception for all domain errors."""
    pass


class DataLoadError(MovieRecommendationError):
    """Raised when file loading fails or is invalid."""
    pass


class ValidationError(MovieRecommendationError):
    """Raised when dataset schema or essential data fails validation."""
    pass


class DataCleaningError(MovieRecommendationError):
    """Raised when an error occurs during data cleaning pipeline."""
    pass


class RecommendationEngineError(MovieRecommendationError):
    """Raised when recommendation computation or query fails."""
    pass


class InsufficientFeaturesError(RecommendationEngineError):
    """Raised when a dataset or selected movie lacks content features."""
    pass
