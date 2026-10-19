"""Recommendation engine package."""

from src.recommendation.content_based import (
    ContentBasedRecommender,
    RecommendationItem,
)
from src.recommendation.similarity import (
    compute_full_similarity_matrix,
    compute_query_similarity,
)

__all__ = [
    "ContentBasedRecommender",
    "RecommendationItem",
    "compute_full_similarity_matrix",
    "compute_query_similarity",
]
