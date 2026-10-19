"""Preprocessing package: text cleaning, entity extraction, and feature engineering."""

from src.preprocessing.feature_engineering import (
    build_movie_soup,
    vectorize_features,
)
from src.preprocessing.text_processing import (
    clean_text,
    entity_to_token,
    process_entities_string,
    process_genres_string,
)

__all__ = [
    "build_movie_soup",
    "clean_text",
    "entity_to_token",
    "process_entities_string",
    "process_genres_string",
    "vectorize_features",
]
