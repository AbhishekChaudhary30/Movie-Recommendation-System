"""Unit tests for automatic data cleaning and quality reporting."""

import pandas as pd

from src.data.cleaning import (
    clean_dataset,
    extract_year_from_value,
    parse_json_or_delimited,
)
from src.data.schema import detect_schema


def test_parse_json_or_delimited():
    """Test parsing TMDB style JSON arrays, pipe-separated, and list strings."""
    # TMDB JSON
    tmdb_json = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    assert parse_json_or_delimited(tmdb_json) == "Action, Adventure"

    # Pipe separated
    pipe_str = "Drama|Mystery|Sci-Fi"
    assert parse_json_or_delimited(pipe_str) == "Drama, Mystery, Sci-Fi"

    # Plain list of strings
    list_str = "['Brad Pitt', 'Edward Norton']"
    assert parse_json_or_delimited(list_str) == "Brad Pitt, Edward Norton"

    # Normal text
    assert parse_json_or_delimited("Christopher Nolan") == "Christopher Nolan"
    assert parse_json_or_delimited(None) == ""


def test_extract_year_from_value():
    """Test extracting 4-digit release years."""
    assert extract_year_from_value("2010-07-16") == 2010
    assert extract_year_from_value("1994") == 1994
    assert extract_year_from_value(2008) == 2008
    assert extract_year_from_value("Invalid Date") is None
    assert extract_year_from_value(None) is None


def test_cleaning_drops_exact_duplicates():
    """Exact duplicate rows must be removed."""
    df = pd.DataFrame({
        "title": ["Inception", "Inception", "Interstellar"],
        "genres": ["Action", "Action", "Sci-Fi"],
        "vote_average": [8.4, 8.4, 8.4],
    })
    schema = detect_schema(list(df.columns))
    cleaned_df, report = clean_dataset(df, schema)

    assert len(cleaned_df) == 2
    assert report.exact_duplicates_removed == 1


def test_cleaning_disambiguates_duplicate_movie_titles():
    """Movies with same title but different metadata/years are disambiguated."""
    df = pd.DataFrame({
        "title": ["Halloween", "Halloween"],
        "release_date": ["1978-10-25", "2018-10-19"],
        "genres": ["Horror", "Horror"],
    })
    schema = detect_schema(list(df.columns))
    cleaned_df, report = clean_dataset(df, schema)

    assert len(cleaned_df) == 2
    assert report.potential_duplicate_titles_count == 2
    assert "Halloween (1978)" in cleaned_df["display_title"].values
    assert "Halloween (2018)" in cleaned_df["display_title"].values


def test_cleaning_excludes_empty_titles():
    """Rows with missing/empty title must be excluded."""
    df = pd.DataFrame({
        "title": ["Inception", None, "", "nan"],
        "genres": ["Action", "Comedy", "Drama", "Sci-Fi"],
    })
    schema = detect_schema(list(df.columns))
    cleaned_df, report = clean_dataset(df, schema)

    assert len(cleaned_df) == 1
    assert cleaned_df["raw_title"].iloc[0] == "Inception"
    assert report.missing_titles_removed == 3


def test_cleaning_sanitizes_negative_numerics():
    """Negative runtime or rating must be converted to NaN and reported."""
    df = pd.DataFrame({
        "title": ["Movie A", "Movie B"],
        "runtime": [-100, 120],
        "vote_average": [-5.0, 8.0],
    })
    schema = detect_schema(list(df.columns))
    cleaned_df, report = clean_dataset(df, schema)

    assert pd.isna(cleaned_df.loc[0, "cleaned_runtime"])
    assert cleaned_df.loc[1, "cleaned_runtime"] == 120
    assert pd.isna(cleaned_df.loc[0, "cleaned_rating"])
    assert cleaned_df.loc[1, "cleaned_rating"] == 8.0
    assert report.anomalies_handled.get("negative_runtimes") == 1
    assert report.anomalies_handled.get("negative_ratings") == 1


def test_cleaning_report_exports():
    """Verify markdown and dict exports of CleaningReport."""
    df = pd.DataFrame({"title": ["Test Movie"], "genres": ["Action"]})
    schema = detect_schema(list(df.columns))
    _, report = clean_dataset(df, schema)

    md = report.to_markdown()
    assert "Data Quality & Cleaning Audit Report" in md
    assert "Original Shape" in md

    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["cleaned_rows"] == 1
