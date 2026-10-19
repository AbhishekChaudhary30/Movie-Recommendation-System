"""Analytics and visualization package."""

from src.analytics.metrics import (
    compute_overview_metrics,
    get_unique_genres_list,
    get_genre_counts,
    plot_genre_distribution,
    plot_release_year_distribution,
    plot_rating_distribution,
    plot_recommendation_similarity_scores,
)

__all__ = [
    "compute_overview_metrics",
    "get_unique_genres_list",
    "get_genre_counts",
    "plot_genre_distribution",
    "plot_release_year_distribution",
    "plot_rating_distribution",
    "plot_recommendation_similarity_scores",
]
