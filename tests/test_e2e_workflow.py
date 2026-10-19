"""End-to-end workflow verification test suite."""

from src.analytics.metrics import (
    compute_overview_metrics,
    get_unique_genres_list,
    plot_genre_distribution,
    plot_rating_distribution,
    plot_recommendation_similarity_scores,
    plot_release_year_distribution,
)
from src.data.cleaning import clean_dataset
from src.data.loader import load_csv
from src.data.schema import detect_schema
from src.data.validation import validate_dataset
from src.recommendation.content_based import ContentBasedRecommender


def test_complete_e2e_workflow():
    """Verify complete progression: Upload -> Validate -> Clean -> Recommend -> Analytics."""
    # Stage 1: Load sample CSV
    df_raw = load_csv("data/sample/sample_movies.csv")
    assert len(df_raw) >= 40
    assert "title" in df_raw.columns

    # Stage 2: Schema Detection and Validation
    mapping = detect_schema(list(df_raw.columns))
    assert mapping.has_title is True
    val_report = validate_dataset(df_raw, mapping)
    assert val_report.is_valid is True
    assert len(val_report.critical_errors) == 0

    # Stage 3: Automatic Cleaning & Quality Audit
    cleaned_df, clean_report = clean_dataset(df_raw, mapping)
    assert len(cleaned_df) > 0
    assert "display_title" in cleaned_df.columns
    assert clean_report.cleaned_rows == len(cleaned_df)
    assert clean_report.to_markdown().startswith("### Data Quality & Cleaning Audit Report")

    # Stage 4: Recommendation Engine
    recommender = ContentBasedRecommender()
    recommender.fit(cleaned_df, mapping)

    # 4a. Recommend for Inception
    recs_inception = recommender.recommend("Inception", top_n=5)
    assert len(recs_inception) == 5
    assert "Inception" not in recs_inception["display_title"].values
    assert all("explanation" in row for _, row in recs_inception.iterrows())

    # 4b. Recommend with Genre and Rating Filters
    recs_filtered = recommender.recommend(
        "The Godfather",
        top_n=5,
        genre_filter="Drama",
        min_rating=7.5,
    )
    assert not recs_filtered.empty
    assert "The Godfather" not in recs_filtered["display_title"].values
    for _, row in recs_filtered.iterrows():
        assert "drama" in str(row["cleaned_genres"]).lower()
        assert row["cleaned_rating"] >= 7.5

    # 4c. Analytics Metrics & Plots
    overview = compute_overview_metrics(cleaned_df)
    assert overview["total_movies"] == len(cleaned_df)
    assert overview["unique_genres"] > 0

    fig_genres = plot_genre_distribution(cleaned_df)
    assert fig_genres is not None

    fig_years = plot_release_year_distribution(cleaned_df)
    assert fig_years is not None

    fig_ratings = plot_rating_distribution(cleaned_df)
    assert fig_ratings is not None

    fig_matches = plot_recommendation_similarity_scores(recs_inception)
    assert fig_matches is not None
