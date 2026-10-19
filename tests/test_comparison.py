"""Unit tests for movie comparison and pairwise similarity calculation."""

import pytest

from src.analytics.reports import build_movie_comparison_matrix
from src.data.cleaning import clean_dataset
from src.data.loader import load_csv
from src.data.schema import detect_schema
from src.recommendation.content_based import ContentBasedRecommender


@pytest.fixture
def recommender_sample():
    """Fitted recommender on sample movies."""
    df = load_csv("data/sample/sample_movies.csv")
    schema = detect_schema(list(df.columns))
    cleaned_df, _ = clean_dataset(df, schema)

    rec = ContentBasedRecommender()
    rec.fit(cleaned_df, schema)
    return rec, cleaned_df


def test_compute_pair_similarity(recommender_sample):
    """Test computing direct pairwise similarity between two movies."""
    rec, _ = recommender_sample
    m1 = "Inception"
    m2 = "Interstellar"

    sim_res = rec.compute_pair_similarity(m1, m2)
    assert isinstance(sim_res, dict)
    assert sim_res["movie1_title"] == m1
    assert sim_res["movie2_title"] == m2
    assert 0.0 <= sim_res["similarity_score"] <= 1.0
    assert 0.0 <= sim_res["similarity_pct"] <= 100.0
    # Both are directed by Christopher Nolan with Sci-Fi themes
    assert "Christopher Nolan" in str(sim_res["shared_director"])
    assert len(sim_res["explanation"]) > 0


def test_compute_pair_similarity_same_movie(recommender_sample):
    """Pairwise similarity of a movie with itself must be 1.0 (100%)."""
    rec, _ = recommender_sample
    sim_res = rec.compute_pair_similarity("Inception", "Inception")

    assert sim_res["similarity_score"] == 1.0
    assert sim_res["similarity_pct"] == 100.0


def test_build_movie_comparison_matrix(recommender_sample):
    """Test building tabular comparison matrix for multiple movies."""
    rec, df = recommender_sample
    titles = ["The Dark Knight", "The Dark Knight Rises", "Inception"]

    matrix_df = build_movie_comparison_matrix(df, titles, recommender=rec)
    assert not matrix_df.empty
    assert len(matrix_df) == 3
    assert "Movie Title" in matrix_df.columns
    assert "Genres" in matrix_df.columns
    assert "Release Year" in matrix_df.columns
    assert any("Match vs." in col for col in matrix_df.columns)
