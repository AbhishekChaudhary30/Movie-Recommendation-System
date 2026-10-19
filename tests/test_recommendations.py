"""Unit tests for content-based recommendation engine."""

import pandas as pd
import pytest

from src.data.cleaning import clean_dataset
from src.data.loader import load_csv
from src.data.schema import detect_schema
from src.recommendation.content_based import ContentBasedRecommender
from src.utils.exceptions import RecommendationEngineError


@pytest.fixture
def recommender_sample():
    """Fixture providing a fitted recommender on sample movies."""
    df = load_csv("data/sample/sample_movies.csv")
    schema = detect_schema(list(df.columns))
    cleaned_df, _ = clean_dataset(df, schema)

    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, schema)
    return recommender


def test_recommendation_self_exclusion(recommender_sample):
    """The queried movie must never be recommended to itself."""
    query = "Inception"
    recs = recommender_sample.recommend(query, top_n=5)

    assert not recs.empty
    assert "Inception" not in recs["display_title"].values
    assert len(recs) <= 5


def test_recommendation_ranking_and_scores(recommender_sample):
    """Similarity scores must be sorted in descending order between 0.0 and 1.0."""
    recs = recommender_sample.recommend("The Dark Knight", top_n=5)

    assert not recs.empty
    scores = recs["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # The Dark Knight Rises should rank highly due to Nolan, Bale, Batman theme
    assert any("Dark Knight" in title for title in recs["display_title"].values)


def test_recommendation_genre_filter(recommender_sample):
    """Recommendations must respect selected genre filter."""
    recs = recommender_sample.recommend("Inception", top_n=10, genre_filter="Crime")

    assert not recs.empty
    for _, row in recs.iterrows():
        assert "crime" in str(row["cleaned_genres"]).lower()


def test_recommendation_rating_filter(recommender_sample):
    """Recommendations must respect minimum rating filter."""
    min_rating = 8.3
    recs = recommender_sample.recommend("Inception", top_n=10, min_rating=min_rating)

    assert not recs.empty
    for _, row in recs.iterrows():
        assert row["cleaned_rating"] >= min_rating


def test_recommendation_year_filter(recommender_sample):
    """Recommendations must respect release year bounds."""
    recs = recommender_sample.recommend("Inception", top_n=10, year_min=2015, year_max=2024)

    assert not recs.empty
    for _, row in recs.iterrows():
        if pd.notna(row["cleaned_release_year"]):
            assert 2015 <= row["cleaned_release_year"] <= 2024


def test_recommendation_explanations_present(recommender_sample):
    """Recommendations must include explanation fields."""
    recs = recommender_sample.recommend("Pulp Fiction", top_n=3)

    assert not recs.empty
    first_rec = recs.iloc[0]
    assert "explanation" in first_rec
    assert len(first_rec["explanation"]) > 0
    assert "shared_genres" in first_rec
    assert "shared_cast" in first_rec


def test_recommendation_unknown_movie_raises_error(recommender_sample):
    """Querying an unknown movie must raise RecommendationEngineError."""
    with pytest.raises(RecommendationEngineError, match="not found"):
        recommender_sample.recommend("NonExistentMovieTitle999")


def test_recommender_handles_missing_metadata():
    """Recommender must handle movies with missing genres/overviews/directors gracefully."""
    df = pd.DataFrame({
        "title": ["Bare Movie 1", "Bare Movie 2", "Bare Movie 3"],
        "genres": [None, "", "Drama"],
        "overview": [None, "Some story", ""],
    })
    schema = detect_schema(list(df.columns))
    cleaned_df, _ = clean_dataset(df, schema)

    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, schema)

    recs = recommender.recommend("Bare Movie 1", top_n=2)
    assert len(recs) <= 2
    assert "Bare Movie 1" not in recs["display_title"].values


def test_recommendation_min_similarity_filter(recommender_sample):
    """Recommendations must filter candidates below the minimum similarity threshold."""
    # With a high threshold, only highly similar movies should appear
    recs_high = recommender_sample.recommend("The Dark Knight", top_n=10, min_similarity=0.20)
    assert not recs_high.empty
    assert all(r["similarity_score"] >= 0.20 for _, r in recs_high.iterrows())

    # An unrealistically high threshold should return an empty dataframe without error
    recs_none = recommender_sample.recommend("The Dark Knight", top_n=10, min_similarity=0.99)
    assert recs_none.empty
