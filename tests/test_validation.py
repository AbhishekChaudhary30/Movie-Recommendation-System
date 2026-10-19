"""Unit tests for schema detection and dataset validation."""

import pandas as pd

from src.data.schema import detect_schema, normalize_column_name
from src.data.validation import validate_dataset


def test_normalize_column_name():
    """Test column normalization rules."""
    assert normalize_column_name("Movie Title") == "movie_title"
    assert normalize_column_name("  Release-Date. ") == "release_date"
    assert normalize_column_name("IMDB_Score#1") == "imdb_score1"


def test_detect_schema_standard():
    """Test schema detection on standard column names."""
    cols = ["title", "genres", "overview", "release_date", "vote_average", "runtime"]
    mapping = detect_schema(cols)

    assert mapping.has_title is True
    assert mapping.get_original("title") == "title"
    assert mapping.get_original("genres") == "genres"
    assert mapping.get_original("overview") == "overview"
    assert mapping.get_original("rating") == "vote_average"
    assert "genres" in mapping.available_features
    assert "overview" in mapping.available_features


def test_detect_schema_alternate_synonyms():
    """Test detection on diverse synonym column names."""
    cols = ["movie_title", "plot", "actors", "directors", "year", "score"]
    mapping = detect_schema(cols)

    assert mapping.has_title is True
    assert mapping.get_original("title") == "movie_title"
    assert mapping.get_original("overview") == "plot"
    assert mapping.get_original("cast") == "actors"
    assert mapping.get_original("director") == "directors"
    assert mapping.get_original("release_date") == "year"
    assert mapping.get_original("rating") == "score"


def test_detect_schema_user_overrides():
    """Test applying explicit user overrides."""
    cols = ["custom_film_name", "genre_list"]
    overrides = {"title": "custom_film_name", "genres": "genre_list"}
    mapping = detect_schema(cols, user_overrides=overrides)

    assert mapping.has_title is True
    assert mapping.get_original("title") == "custom_film_name"
    assert mapping.get_original("genres") == "genre_list"


def test_validate_dataset_missing_title():
    """Validation must fail when no title column is present."""
    df = pd.DataFrame({"overview": ["Great movie"], "rating": [8.0]})
    mapping = detect_schema(list(df.columns))
    report = validate_dataset(df, mapping)

    assert report.is_valid is False
    assert any("title" in err.lower() for err in report.critical_errors)


def test_validate_dataset_valid():
    """Validation succeeds when title is present."""
    df = pd.DataFrame({
        "title": ["Inception", "Interstellar"],
        "genres": ["Action", "Sci-Fi"],
        "vote_average": [8.4, 8.4],
    })
    mapping = detect_schema(list(df.columns))
    report = validate_dataset(df, mapping)

    assert report.is_valid is True
    assert len(report.critical_errors) == 0


def test_validate_dataset_negative_ratings_warning():
    """Validation detects negative values as warnings."""
    df = pd.DataFrame({
        "title": ["Movie A", "Movie B"],
        "rating": [-1.0, 7.5],
        "runtime": [-90, 120],
    })
    mapping = detect_schema(list(df.columns))
    report = validate_dataset(df, mapping)

    assert report.is_valid is True  # Non-blocking, but warned
    assert any("negative" in w.lower() for w in report.warnings)
