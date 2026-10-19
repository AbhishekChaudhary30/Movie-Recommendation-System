"""Application configuration and constants."""

from pathlib import Path
from typing import Dict, List

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Application Metadata
APP_TITLE = "Movie Recommendation System"
APP_SUBTITLE = "Intelligent Movie Discovery & Recommendation Analytics"
APP_DESCRIPTION = (
    "Discover movies you'll love through intelligent content-based filtering, "
    "similarity scoring, movie metadata analysis, and personalized recommendations."
)
APP_VERSION = "1.0.0"

# Limits & Guardrails
MAX_FILE_SIZE_MB = 50
MAX_RECOMMENDATIONS_DEFAULT = 10
MAX_RECOMMENDATIONS_LIMIT = 30
MAX_TFIDF_FEATURES = 10000

# Canonical Schema Definitions & Common Synonyms
SCHEMA_SYNONYMS: Dict[str, List[str]] = {
    "title": [
        "title",
        "movie_title",
        "original_title",
        "name",
        "film_title",
        "track_name",
    ],
    "genres": [
        "genres",
        "genre",
        "categories",
        "category",
        "genre_names",
    ],
    "overview": [
        "overview",
        "description",
        "plot",
        "summary",
        "storyline",
        "synopsis",
    ],
    "keywords": [
        "keywords",
        "keyword",
        "tags",
        "plot_keywords",
    ],
    "cast": [
        "cast",
        "actors",
        "starring",
        "stars",
        "top_cast",
        "lead_actors",
    ],
    "director": [
        "director",
        "directors",
        "directed_by",
        "crew",
    ],
    "release_date": [
        "release_date",
        "release_year",
        "year",
        "date",
        "released",
        "first_air_date",
    ],
    "rating": [
        "vote_average",
        "rating",
        "imdb_rating",
        "score",
        "user_rating",
        "average_rating",
    ],
    "vote_count": [
        "vote_count",
        "num_votes",
        "votes",
        "total_votes",
        "rating_count",
    ],
    "runtime": [
        "runtime",
        "duration",
        "length",
        "duration_min",
    ],
    "popularity": [
        "popularity",
        "pop_score",
        "trending_score",
    ],
    "language": [
        "original_language",
        "language",
        "lang",
    ],
    "poster_path": [
        "poster_path",
        "poster_url",
        "poster",
        "image_url",
    ],
    "id": [
        "id",
        "movie_id",
        "imdb_id",
        "tmdb_id",
    ],
}

# Recommendation Feature Weights
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "genres": 2.0,
    "keywords": 1.5,
    "overview": 1.0,
    "cast": 1.2,
    "director": 1.0,
}

# Theme Palette Tokens
THEME = {
    "bg_main": "#FBFBFD",
    "bg_surface": "#FFFFFF",
    "primary_accent": "#6C5CE7",
    "secondary_accent": "#00B894",
    "highlight_amber": "#E17055",
    "text_primary": "#1E272E",
    "text_secondary": "#636E72",
    "border_subtle": "#E2E8F0",
    "card_shadow": "0 4px 14px rgba(0, 0, 0, 0.05)",
}
