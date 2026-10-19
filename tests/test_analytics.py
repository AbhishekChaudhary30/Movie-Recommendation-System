"""Unit tests for advanced analytics metrics, insights, and report generators."""

import pandas as pd
import pytest

from src.analytics.metrics import (
    compute_overview_metrics,
    generate_executive_insights,
    get_decade_counts,
    get_metadata_completeness,
)
from src.analytics.reports import generate_executive_summary_markdown
from src.data.cleaning import clean_dataset
from src.data.loader import load_csv
from src.data.schema import detect_schema


@pytest.fixture
def sample_cleaned_df():
    """Fixture providing a cleaned version of sample movies."""
    df = load_csv("data/sample/sample_movies.csv")
    schema = detect_schema(list(df.columns))
    cleaned_df, report = clean_dataset(df, schema)
    return cleaned_df, report


def test_compute_overview_metrics(sample_cleaned_df):
    """Test KPI calculations on sample dataset."""
    df, _ = sample_cleaned_df
    kpis = compute_overview_metrics(df)

    assert kpis["total_movies"] == len(df)
    assert kpis["unique_titles"] > 0
    assert kpis["unique_genres"] > 5
    assert 0.0 <= kpis["avg_rating"] <= 10.0
    assert 0.0 <= kpis["median_rating"] <= 10.0
    assert kpis["avg_runtime"] > 60
    assert 1900 <= kpis["min_year"] <= 2030
    assert 1900 <= kpis["max_year"] <= 2030
    assert kpis["most_common_year"] is not None
    assert isinstance(kpis["missing_desc_pct"], float)
    assert isinstance(kpis["duplicate_pct"], float)


def test_generate_executive_insights(sample_cleaned_df):
    """Test generating evidence-based executive insights."""
    df, _ = sample_cleaned_df
    insights = generate_executive_insights(df)

    assert isinstance(insights, list)
    assert len(insights) >= 3
    # Verify presence of key analytical topics
    insights_str = " ".join(insights).lower()
    assert "genre" in insights_str
    assert "rating" in insights_str or "benchmark" in insights_str


def test_get_decade_counts(sample_cleaned_df):
    """Test movie aggregation by decade."""
    df, _ = sample_cleaned_df
    decade_df = get_decade_counts(df)

    assert not decade_df.empty
    assert "Decade" in decade_df.columns
    assert "Count" in decade_df.columns
    assert decade_df["Count"].sum() == df["cleaned_release_year"].notna().sum()


def test_get_metadata_completeness(sample_cleaned_df):
    """Test metadata completeness percentages."""
    df, _ = sample_cleaned_df
    comp_df = get_metadata_completeness(df)

    assert not comp_df.empty
    assert "Attribute" in comp_df.columns
    assert "Completeness (%)" in comp_df.columns
    for _, row in comp_df.iterrows():
        assert 0.0 <= row["Completeness (%)"] <= 100.0


def test_generate_executive_summary_markdown(sample_cleaned_df):
    """Test markdown report generation."""
    df, report = sample_cleaned_df
    md = generate_executive_summary_markdown(df, "sample_movies.csv", report)

    assert isinstance(md, str)
    assert "Executive Movie Analytics & Discovery Report" in md
    assert "Key Performance Indicators" in md
    assert "Dynamic Analytical Insights" in md
    assert "Top Genre Representation" in md


def test_empty_dataset_analytics():
    """Verify safe handling of empty DataFrames."""
    empty_df = pd.DataFrame()
    kpis = compute_overview_metrics(empty_df)
    assert kpis["total_movies"] == 0

    insights = generate_executive_insights(empty_df)
    assert len(insights) == 1
    assert "empty" in insights[0].lower()

    decades = get_decade_counts(empty_df)
    assert decades.empty
