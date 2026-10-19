"""Unit tests for Plotly visualization builders and ChartRegistry."""

import pandas as pd
import pytest

from src.analytics.visualizations import (
    CHART_REGISTRY,
    get_available_charts,
    plot_genre_bar,
    plot_genre_cooccurrence,
    plot_genre_share_donut,
    plot_genre_treemap,
    plot_metadata_completeness_chart,
    plot_movies_by_decade,
    plot_overview_word_count,
    plot_rating_box,
    plot_rating_by_genre,
    plot_rating_histogram,
    plot_rating_trend_by_decade,
    plot_rating_vs_votes_scatter,
    plot_release_year_histogram,
    plot_runtime_distribution,
    plot_runtime_vs_rating_scatter,
)
from src.data.cleaning import clean_dataset
from src.data.loader import load_csv
from src.data.schema import detect_schema


@pytest.fixture
def cleaned_sample():
    """Fixture providing cleaned movie dataset."""
    df = load_csv("data/sample/sample_movies.csv")
    schema = detect_schema(list(df.columns))
    cleaned_df, _ = clean_dataset(df, schema)
    return cleaned_df


def test_chart_registry_availability(cleaned_sample):
    """Test that chart registry identifies available charts based on schema."""
    available = get_available_charts(cleaned_sample)
    assert len(available) >= 10
    # Sample dataset has all canonical columns so all registry charts should be available
    assert len(available) == len(CHART_REGISTRY)


def test_genre_visualizations(cleaned_sample):
    """Test genre analysis charts render valid Plotly figures."""
    fig_bar = plot_genre_bar(cleaned_sample, top_n=8)
    assert fig_bar is not None

    fig_donut = plot_genre_share_donut(cleaned_sample, top_n=6)
    assert fig_donut is not None

    fig_treemap = plot_genre_treemap(cleaned_sample, top_n=10)
    assert fig_treemap is not None

    fig_cooc = plot_genre_cooccurrence(cleaned_sample, top_n=5)
    assert fig_cooc is not None


def test_rating_visualizations(cleaned_sample):
    """Test rating charts render valid Plotly figures."""
    fig_hist = plot_rating_histogram(cleaned_sample)
    assert fig_hist is not None

    fig_box = plot_rating_box(cleaned_sample)
    assert fig_box is not None

    fig_genre_r = plot_rating_by_genre(cleaned_sample, top_n=6)
    assert fig_genre_r is not None

    fig_scatter = plot_rating_vs_votes_scatter(cleaned_sample, log_votes=True)
    assert fig_scatter is not None


def test_temporal_and_runtime_visualizations(cleaned_sample):
    """Test release year and runtime charts render valid Plotly figures."""
    fig_y_hist = plot_release_year_histogram(cleaned_sample)
    assert fig_y_hist is not None

    fig_decades = plot_movies_by_decade(cleaned_sample)
    assert fig_decades is not None

    fig_trend = plot_rating_trend_by_decade(cleaned_sample)
    assert fig_trend is not None

    fig_rt = plot_runtime_distribution(cleaned_sample)
    assert fig_rt is not None

    fig_rt_scatter = plot_runtime_vs_rating_scatter(cleaned_sample)
    assert fig_rt_scatter is not None


def test_metadata_and_text_visualizations(cleaned_sample):
    """Test metadata completeness and synopsis length charts."""
    fig_comp = plot_metadata_completeness_chart(cleaned_sample)
    assert fig_comp is not None

    fig_ov = plot_overview_word_count(cleaned_sample)
    assert fig_ov is not None


def test_visualizations_handle_missing_columns():
    """Verify that charts gracefully return None when required columns are absent."""
    bare_df = pd.DataFrame({"display_title": ["Movie 1", "Movie 2"]})

    assert plot_genre_bar(bare_df) is None
    assert plot_rating_histogram(bare_df) is None
    assert plot_release_year_histogram(bare_df) is None
    assert plot_runtime_distribution(bare_df) is None
    assert plot_genre_cooccurrence(bare_df) is None
