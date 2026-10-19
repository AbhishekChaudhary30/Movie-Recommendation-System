"""Comprehensive edge-case and boundary verification test suite."""

import pandas as pd
import pytest

from src.analytics.metrics import compute_overview_metrics
from src.analytics.visualizations import CHART_REGISTRY
from src.config import MAX_FILE_SIZE_MB
from src.data.cleaning import clean_dataset
from src.data.loader import check_file_size
from src.data.schema import detect_schema
from src.data.validation import validate_dataset
from src.recommendation.content_based import ContentBasedRecommender
from src.utils.exceptions import DataLoadError, RecommendationEngineError


def test_single_movie_catalog_recommendation():
    """Verify that a single-movie catalog fits cleanly and returns empty recommendations without crashing."""
    df_single = pd.DataFrame([{
        "title": "Solo Movie",
        "genres": "Drama",
        "overview": "A character alone in a room.",
        "release_year": 2020,
        "rating": 7.5,
    }])
    mapping = detect_schema(list(df_single.columns))
    cleaned_df, report = clean_dataset(df_single, mapping)
    assert len(cleaned_df) == 1

    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, mapping)

    # Calling recommend on the only movie should return empty DataFrame due to self-exclusion
    recs = recommender.recommend("Solo Movie", top_n=5)
    assert isinstance(recs, pd.DataFrame)
    assert recs.empty


def test_all_optional_columns_null():
    """Verify system stability when every column except the title is null."""
    df_sparse = pd.DataFrame([
        {"title": "Mystery Film A", "genres": None, "overview": None, "cast": None, "director": None, "rating": None},
        {"title": "Mystery Film B", "genres": None, "overview": None, "cast": None, "director": None, "rating": None},
    ])
    mapping = detect_schema(list(df_sparse.columns))
    cleaned_df, report = clean_dataset(df_sparse, mapping)
    assert len(cleaned_df) == 2

    # Overview metrics handle missing columns/nulls
    metrics = compute_overview_metrics(cleaned_df)
    assert metrics["total_movies"] == 2
    assert metrics["avg_rating"] is None

    # Recommender still fits and produces recommendations
    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, mapping)
    recs = recommender.recommend("Mystery Film A", top_n=1)
    assert len(recs) == 1
    assert recs.iloc[0]["display_title"] == "Mystery Film B"


def test_extreme_and_dirty_numeric_values():
    """Verify cleaning and parsing handles strings, negative numbers, and out-of-bound values."""
    df_dirty = pd.DataFrame([
        {"title": "Film A", "rating": "8.5", "runtime": "120 min", "release_date": "2021-06-15", "vote_count": "1,500"},
        {"title": "Film B", "rating": -5.0, "runtime": -90, "release_date": "Unknown Date", "vote_count": -10},
        {"title": "Film C", "rating": 15.0, "runtime": "N/A", "release_date": "1880", "vote_count": "invalid"},
    ])
    mapping = detect_schema(list(df_dirty.columns))
    cleaned_df, report = clean_dataset(df_dirty, mapping)

    # Film A parsed numeric strings
    row_a = cleaned_df[cleaned_df["raw_title"] == "Film A"].iloc[0]
    assert row_a["cleaned_rating"] == 8.5
    assert row_a["cleaned_runtime"] == 120.0
    assert row_a["cleaned_release_year"] == 2021
    assert row_a["cleaned_vote_count"] == 1500

    # Film B sanitized negative values
    row_b = cleaned_df[cleaned_df["raw_title"] == "Film B"].iloc[0]
    assert pd.isna(row_b["cleaned_rating"])
    assert pd.isna(row_b["cleaned_runtime"])
    assert pd.isna(row_b["cleaned_release_year"])
    assert pd.isna(row_b["cleaned_vote_count"])


def test_unicode_and_accented_movie_titles():
    """Verify correct handling of foreign characters and accents in titles."""
    df_unicode = pd.DataFrame([
        {"title": "Amélie", "genres": "Romance, Comedy", "overview": "A whimsical Parisian woman."},
        {"title": "千と千尋の神隠し", "genres": "Animation, Fantasy", "overview": "Spirited Away magical journey."},
        {"title": "La vita è bella", "genres": "Drama, War", "overview": "A father shields his son in a camp."},
    ])
    mapping = detect_schema(list(df_unicode.columns))
    cleaned_df, report = clean_dataset(df_unicode, mapping)
    assert len(cleaned_df) == 3

    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, mapping)

    # Lookup by unicode titles
    idx_jp = recommender.find_movie_index("千と千尋の神隠し")
    assert idx_jp is not None

    recs = recommender.recommend("Amélie", top_n=2)
    assert not recs.empty
    assert "Amélie" not in recs["display_title"].values


def test_extreme_feature_weights():
    """Verify recommender functions with boundary weights (zeros and large weights)."""
    df = pd.DataFrame([
        {"title": "Space Odyssey", "genres": "Sci-Fi", "overview": "A voyage to Jupiter.", "cast": "Keir Dullea"},
        {"title": "Interstellar", "genres": "Sci-Fi", "overview": "A voyage through wormhole.", "cast": "Matthew McConaughey"},
        {"title": "Dallas Buyers Club", "genres": "Drama", "overview": "An AIDS patient story.", "cast": "Matthew McConaughey"},
    ])
    mapping = detect_schema(list(df.columns))
    cleaned_df, _ = clean_dataset(df, mapping)

    # Weight heavily on Cast (10x) vs Genre (0.1x)
    weights_cast = {"genres": 0.1, "keywords": 1.0, "overview": 1.0, "cast": 10.0, "director": 1.0}
    rec_cast = ContentBasedRecommender(feature_weights=weights_cast)
    rec_cast.fit(cleaned_df, mapping)

    recs = rec_cast.recommend("Interstellar", top_n=2)
    # Dallas Buyers Club should rank high due to shared actor Matthew McConaughey
    assert recs.iloc[0]["display_title"] == "Dallas Buyers Club"


def test_empty_filtered_recommendation_results():
    """Verify recommender returns empty DataFrame when filters match zero candidates."""
    df = pd.DataFrame([
        {"title": "Movie 1", "genres": "Comedy", "rating": 6.0},
        {"title": "Movie 2", "genres": "Comedy", "rating": 5.5},
        {"title": "Movie 3", "genres": "Horror", "rating": 6.5},
    ])
    mapping = detect_schema(list(df.columns))
    cleaned_df, _ = clean_dataset(df, mapping)

    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, mapping)

    # Filter with min_rating=9.0 (none qualify)
    recs = recommender.recommend("Movie 1", min_rating=9.0)
    assert isinstance(recs, pd.DataFrame)
    assert recs.empty

    # Filter with non-existent genre
    recs_genre = recommender.recommend("Movie 1", genre_filter="Documentary")
    assert recs_genre.empty


def test_nonexistent_movie_recommendation_raises_informative_error():
    """Verify asking recommendations for an unindexed movie raises a clean RecommendationEngineError."""
    df = pd.DataFrame([{"title": "The Matrix", "genres": "Sci-Fi"}, {"title": "Inception", "genres": "Sci-Fi"}])
    mapping = detect_schema(list(df.columns))
    cleaned_df, _ = clean_dataset(df, mapping)

    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, mapping)

    with pytest.raises(RecommendationEngineError) as exc_info:
        recommender.recommend("Gladiator", top_n=5)
    assert "not found in the dataset" in str(exc_info.value)


def test_missing_title_column_validation():
    """Verify validation flags dataset without any title column as invalid."""
    df_no_title = pd.DataFrame({"release_year": [2010, 2012], "rating": [7.0, 8.0]})
    mapping = detect_schema(list(df_no_title.columns))
    assert mapping.has_title is False

    val_rep = validate_dataset(df_no_title, mapping)
    assert val_rep.is_valid is False
    assert any("title" in err.lower() for err in val_rep.critical_errors)


def test_oversized_file_limit_guard():
    """Verify file size guard rejects files exceeding MAX_FILE_SIZE_MB."""
    # 51 MB buffer
    large_size = (MAX_FILE_SIZE_MB + 1) * 1024 * 1024
    with pytest.raises(DataLoadError) as exc_info:
        check_file_size(large_size, MAX_FILE_SIZE_MB)
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_chart_registry_handles_empty_dataframe():
    """Verify all registry charts return None safely when evaluated on an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["cleaned_genres", "cleaned_rating", "cleaned_release_year", "cleaned_runtime", "cleaned_vote_count"])
    for chart_def in CHART_REGISTRY:
        fig = chart_def.builder_func(empty_df)
        assert fig is None
