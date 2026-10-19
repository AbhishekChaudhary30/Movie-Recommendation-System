"""Application configuration and constants."""

from pathlib import Path

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Application Metadata
APP_TITLE = "Movie Recommendation System"
APP_NAME = "CineScope"
APP_SUBTITLE = "Intelligent Movie Discovery & Recommendation Analytics"
APP_DESCRIPTION = (
    "CineScope transforms movie metadata into meaningful insights and relevant movie recommendations "
    "through content-based machine learning, interactive analytics, and explainable similarity scoring."
)
APP_VERSION = "1.0.0"

# Limits & Guardrails
MAX_FILE_SIZE_MB = 50
MAX_RECOMMENDATIONS_DEFAULT = 10
MAX_RECOMMENDATIONS_LIMIT = 30
MAX_TFIDF_FEATURES = 10000

# Canonical Schema Definitions & Common Synonyms
SCHEMA_SYNONYMS: dict[str, list[str]] = {
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
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "genres": 2.0,
    "keywords": 1.5,
    "overview": 1.0,
    "cast": 1.2,
    "director": 1.0,
}

# Cinematic Neutral Design Tokens
# Main background: warm stone (#E8E3DA)
# Main surface: soft parchment (#F5F1E9)
# Secondary surface: muted sage-gray (#D5D8CE)
# Primary text: deep ink (#252B2B)
# Secondary text: muted slate (#656B68)
# Primary accent: restrained cinematic plum (#76566E)
# Secondary accent: muted petrol teal (#477E79)
# Highlight: subdued ochre (#C39A5B)
# Borders: warm gray (#CDC8BE)
THEME = {
    "bg_main": "#E8E3DA",
    "bg_surface": "#F5F1E9",
    "surface_secondary": "#D5D8CE",
    "text_primary": "#252B2B",
    "text_secondary": "#656B68",
    "primary_accent": "#76566E",
    "secondary_accent": "#477E79",
    "highlight_ochre": "#C39A5B",
    "border_subtle": "#CDC8BE",
    "card_shadow": "0 2px 10px rgba(37, 43, 43, 0.05)",
}
