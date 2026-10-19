"""Recommendation engine package."""

from src.recommendation.similarity import (
    compute_query_similarity,
    compute_full_similarity_matrix,
)
from src.recommendation.content_based import (
    ContentBasedRecommender,
    RecommendationItem,
)

__all__ = [
    "compute_query_similarity",
    "compute_full_similarity_matrix",
    "ContentBasedRecommender",
    "RecommendationItem",
]
