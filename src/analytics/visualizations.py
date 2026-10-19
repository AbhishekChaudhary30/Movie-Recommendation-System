"""Advanced visualization library and chart registry using Plotly."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.analytics.metrics import (
    get_decade_counts,
    get_genre_counts,
    get_metadata_completeness,
    get_unique_genres_list,
)
from src.config import THEME

# Standard styling defaults for Plotly Cinematic Neutral theme
CHART_LAYOUT = dict(
    plot_bgcolor=THEME["bg_surface"],
    paper_bgcolor=THEME["bg_surface"],
    font=dict(color=THEME["text_primary"], family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
    margin=dict(l=24, r=24, t=44, b=24),
)


@dataclass
class ChartDefinition:
    """Metadata describing an analytics chart in the visualization registry."""
    id: str
    title: str
    category: str
    description: str
    required_columns: List[str]
    builder_func: Callable[..., Optional[go.Figure]]

    def is_available(self, df: pd.DataFrame) -> bool:
        """Check whether the dataset contains the necessary columns for this chart."""
        return all(col in df.columns for col in self.required_columns)


# ==============================================================================
# GENRE ANALYSIS CHARTS
# ==============================================================================

def plot_genre_bar(df: pd.DataFrame, top_n: int = 12) -> Optional[go.Figure]:
    """Horizontal bar chart of most common genres."""
    df_genres = get_genre_counts(df, top_n=top_n)
    if df_genres.empty:
        return None

    fig = px.bar(
        df_genres,
        x="Count",
        y="Genre",
        orientation="h",
        color="Count",
        color_continuous_scale=[THEME["surface_secondary"], THEME["primary_accent"]],
        title=f"Top {len(df_genres)} Movie Genres",
        labels={"Count": "Number of Movies", "Genre": "Genre"},
    )
    fig.update_layout(
        **CHART_LAYOUT,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=max(320, len(df_genres) * 26),
    )
    return fig


def plot_genre_share_donut(df: pd.DataFrame, top_n: int = 8) -> Optional[go.Figure]:
    """Donut chart illustrating proportional genre breakdown."""
    df_genres = get_genre_counts(df, top_n=top_n)
    if df_genres.empty:
        return None

    cinematic_colors = [
        "#76566E",  # Restrained plum
        "#477E79",  # Petrol teal
        "#C39A5B",  # Subdued ochre
        "#656B68",  # Muted slate
        "#937B8F",  # Soft plum
        "#6D9A95",  # Soft teal
        "#8E6D38",  # Deep ochre
        "#A48F7B",  # Warm taupe
    ]
    fig = px.pie(
        df_genres,
        names="Genre",
        values="Count",
        hole=0.45,
        title=f"Genre Share (Top {len(df_genres)})",
        color_discrete_sequence=cinematic_colors,
    )
    fig.update_layout(**CHART_LAYOUT, height=360)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_genre_treemap(df: pd.DataFrame, top_n: int = 15) -> Optional[go.Figure]:
    """Treemap of genre hierarchy and representation."""
    df_genres = get_genre_counts(df, top_n=top_n)
    if df_genres.empty:
        return None

    fig = px.treemap(
        df_genres,
        path=["Genre"],
        values="Count",
        color="Count",
        color_continuous_scale=[THEME["surface_secondary"], THEME["primary_accent"]],
        title="Genre Proportions Treemap",
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, height=360)
    return fig


def plot_genre_cooccurrence(df: pd.DataFrame, top_n: int = 8) -> Optional[go.Figure]:
    """Heatmap showing frequency of genres paired together in movies."""
    if "cleaned_genres" not in df.columns:
        return None

    top_genres_df = get_genre_counts(df, top_n=top_n)
    top_genres = top_genres_df["Genre"].tolist()
    if len(top_genres) < 2:
        return None

    # Compute co-occurrence matrix
    matrix = pd.DataFrame(0, index=top_genres, columns=top_genres)
    for val in df["cleaned_genres"].dropna():
        row_genres = [g.strip().title() for g in str(val).split(",") if g.strip()]
        for g1 in row_genres:
            if g1 in matrix.index:
                for g2 in row_genres:
                    if g2 in matrix.columns:
                        matrix.loc[g1, g2] += 1

    fig = px.imshow(
        matrix,
        labels=dict(x="Genre", y="Co-occurring Genre", color="Movies"),
        x=top_genres,
        y=top_genres,
        color_continuous_scale=[[0, THEME["bg_surface"]], [0.5, THEME["surface_secondary"]], [1.0, THEME["primary_accent"]]],
        title="Genre Co-Occurrence Matrix",
    )
    fig.update_layout(**CHART_LAYOUT, height=380)
    return fig


# ==============================================================================
# RATING ANALYSIS CHARTS
# ==============================================================================

def plot_rating_histogram(df: pd.DataFrame) -> Optional[go.Figure]:
    """Rating distribution histogram with mean line."""
    if "cleaned_rating" not in df.columns:
        return None
    ratings = df["cleaned_rating"].dropna()
    if ratings.empty:
        return None

    mean_val = ratings.mean()
    fig = px.histogram(
        ratings,
        x="cleaned_rating",
        nbins=20,
        title=f"Movie Rating Distribution (Mean: {mean_val:.2f}/10)",
        color_discrete_sequence=[THEME["primary_accent"]],
        labels={"cleaned_rating": "Rating Score", "count": "Count"},
    )
    fig.add_vline(x=mean_val, line_dash="dash", line_color=THEME["highlight_ochre"], annotation_text=f"Mean: {mean_val:.2f}")
    fig.update_layout(**CHART_LAYOUT, height=340)
    return fig


def plot_rating_box(df: pd.DataFrame) -> Optional[go.Figure]:
    """Box plot of movie ratings showing quartiles and outliers."""
    if "cleaned_rating" not in df.columns:
        return None
    ratings = df["cleaned_rating"].dropna()
    if ratings.empty:
        return None

    fig = px.box(
        ratings,
        y="cleaned_rating",
        points="all",
        title="Rating Summary & Quartiles",
        color_discrete_sequence=[THEME["secondary_accent"]],
        labels={"cleaned_rating": "Rating"},
    )
    fig.update_layout(**CHART_LAYOUT, height=340)
    return fig


def plot_rating_by_genre(df: pd.DataFrame, top_n: int = 8) -> Optional[go.Figure]:
    """Average rating grouped by top genres with movie counts."""
    if "cleaned_rating" not in df.columns or "cleaned_genres" not in df.columns:
        return None

    top_genres_df = get_genre_counts(df, top_n=top_n)
    top_genres = top_genres_df["Genre"].tolist()
    if not top_genres:
        return None

    records = []
    for g in top_genres:
        mask = df["cleaned_genres"].astype(str).str.contains(g, case=False, na=False)
        sub_ratings = df.loc[mask, "cleaned_rating"].dropna()
        if not sub_ratings.empty:
            records.append({
                "Genre": g,
                "Average Rating": round(float(sub_ratings.mean()), 2),
                "Movie Count": len(sub_ratings),
            })

    if not records:
        return None

    df_plot = pd.DataFrame(records).sort_values(by="Average Rating", ascending=False)
    fig = px.bar(
        df_plot,
        x="Genre",
        y="Average Rating",
        color="Average Rating",
        color_continuous_scale=[THEME["surface_secondary"], THEME["highlight_ochre"]],
        title="Average Rating by Major Genre",
        hover_data=["Movie Count"],
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_range=[0, 10], height=340)
    return fig


def plot_rating_vs_votes_scatter(df: pd.DataFrame, log_votes: bool = True) -> Optional[go.Figure]:
    """Scatter plot comparing movie ratings against vote counts."""
    if "cleaned_rating" not in df.columns or "cleaned_vote_count" not in df.columns:
        return None

    valid = df.dropna(subset=["cleaned_rating", "cleaned_vote_count"]).copy()
    if valid.empty:
        return None

    fig = px.scatter(
        valid,
        x="cleaned_vote_count",
        y="cleaned_rating",
        hover_name="display_title" if "display_title" in valid.columns else None,
        color="cleaned_rating",
        color_continuous_scale=[THEME["surface_secondary"], THEME["secondary_accent"], THEME["primary_accent"]],
        title="Rating vs. Vote Count Engagement",
        labels={"cleaned_vote_count": "Vote Count", "cleaned_rating": "Rating (0-10)"},
        log_x=log_votes,
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, height=360)
    return fig


# ==============================================================================
# RELEASE YEAR ANALYSIS CHARTS
# ==============================================================================

def plot_release_year_histogram(df: pd.DataFrame) -> Optional[go.Figure]:
    """Histogram of movie releases over time."""
    if "cleaned_release_year" not in df.columns:
        return None

    valid_years = df["cleaned_release_year"].dropna().astype(int)
    if valid_years.empty:
        return None

    fig = px.histogram(
        valid_years,
        x="cleaned_release_year",
        nbins=min(30, max(6, valid_years.nunique())),
        title="Release Year Distribution",
        color_discrete_sequence=[THEME["secondary_accent"]],
        labels={"cleaned_release_year": "Release Year", "count": "Movie Count"},
    )
    fig.update_layout(**CHART_LAYOUT, height=340)
    return fig


def plot_movies_by_decade(df: pd.DataFrame) -> Optional[go.Figure]:
    """Bar chart aggregating movies produced across decades."""
    df_decades = get_decade_counts(df)
    if df_decades.empty:
        return None

    fig = px.bar(
        df_decades,
        x="Decade",
        y="Count",
        color="Count",
        color_continuous_scale=[THEME["surface_secondary"], THEME["secondary_accent"]],
        title="Movie Catalog by Decade",
        labels={"Count": "Total Movies"},
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, height=340)
    return fig


def plot_rating_trend_by_decade(df: pd.DataFrame) -> Optional[go.Figure]:
    """Line chart tracking average movie ratings across decades."""
    if "cleaned_release_year" not in df.columns or "cleaned_rating" not in df.columns:
        return None

    valid = df.dropna(subset=["cleaned_release_year", "cleaned_rating"]).copy()
    if valid.empty:
        return None

    valid["Decade"] = (valid["cleaned_release_year"].astype(int) // 10) * 10
    trend = valid.groupby("Decade")["cleaned_rating"].agg(["mean", "count"]).reset_index()
    trend["Decade_Label"] = trend["Decade"].astype(str) + "s"

    fig = px.line(
        trend,
        x="Decade_Label",
        y="mean",
        markers=True,
        title="Average Rating Trend Across Decades",
        labels={"mean": "Average Rating", "Decade_Label": "Decade"},
        hover_data={"count": True},
    )
    fig.update_traces(line_color=THEME["primary_accent"], marker=dict(size=8, color=THEME["highlight_ochre"]))
    fig.update_layout(**CHART_LAYOUT, yaxis_range=[max(0, trend["mean"].min() - 1), 10], height=340)
    return fig


# ==============================================================================
# RUNTIME & POPULARITY CHARTS
# ==============================================================================

def plot_runtime_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Runtime distribution histogram with median marker."""
    if "cleaned_runtime" not in df.columns:
        return None
    runtimes = df["cleaned_runtime"].dropna()
    if runtimes.empty:
        return None

    med_rt = runtimes.median()
    fig = px.histogram(
        runtimes,
        x="cleaned_runtime",
        nbins=20,
        title=f"Movie Runtime Distribution (Median: {med_rt:.0f} mins)",
        color_discrete_sequence=[THEME["secondary_accent"]],
        labels={"cleaned_runtime": "Runtime (Minutes)"},
    )
    fig.add_vline(x=med_rt, line_dash="dash", line_color=THEME["highlight_ochre"], annotation_text=f"Median: {med_rt:.0f}m")
    fig.update_layout(**CHART_LAYOUT, height=340)
    return fig


def plot_runtime_vs_rating_scatter(df: pd.DataFrame) -> Optional[go.Figure]:
    """Scatter plot exploring relationship between movie length and rating."""
    if "cleaned_runtime" not in df.columns or "cleaned_rating" not in df.columns:
        return None
    valid = df.dropna(subset=["cleaned_runtime", "cleaned_rating"])
    if valid.empty:
        return None

    fig = px.scatter(
        valid,
        x="cleaned_runtime",
        y="cleaned_rating",
        hover_name="display_title" if "display_title" in valid.columns else None,
        color="cleaned_rating",
        color_continuous_scale=[THEME["surface_secondary"], THEME["secondary_accent"], THEME["primary_accent"]],
        title="Runtime vs. Rating Correlation",
        labels={"cleaned_runtime": "Runtime (Minutes)", "cleaned_rating": "Rating (0-10)"},
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, height=340)
    return fig


# ==============================================================================
# METADATA & DATA QUALITY CHARTS
# ==============================================================================

def plot_metadata_completeness_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Horizontal bar chart indicating percentage completeness of key metadata fields."""
    if df.empty:
        return None
    comp_df = get_metadata_completeness(df)
    if comp_df.empty:
        return None

    fig = px.bar(
        comp_df,
        x="Completeness (%)",
        y="Attribute",
        orientation="h",
        color="Completeness (%)",
        color_continuous_scale=[THEME["surface_secondary"], THEME["secondary_accent"]],
        title="Dataset Metadata Attribute Completeness (%)",
        range_x=[0, 100],
    )
    fig.update_layout(
        **CHART_LAYOUT,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=340,
    )
    return fig


def plot_overview_word_count(df: pd.DataFrame) -> Optional[go.Figure]:
    """Distribution of word counts across movie overviews/descriptions."""
    if "cleaned_overview" not in df.columns:
        return None
    lengths = df["cleaned_overview"].dropna().apply(lambda s: len(str(s).split()))
    if lengths.empty:
        return None

    fig = px.histogram(
        lengths,
        x="cleaned_overview",
        nbins=20,
        title="Overview Synopsis Word Count Distribution",
        color_discrete_sequence=[THEME["primary_accent"]],
        labels={"cleaned_overview": "Word Count", "count": "Frequency"},
    )
    fig.update_layout(**CHART_LAYOUT, height=340)
    return fig


def plot_recommendation_similarity_scores(rec_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot similarity match scores for recommendations."""
    if rec_df.empty or "similarity_pct" not in rec_df.columns:
        return None

    df_plot = rec_df.sort_values(by="similarity_pct", ascending=True)
    fig = px.bar(
        df_plot,
        x="similarity_pct",
        y="display_title",
        orientation="h",
        color="similarity_pct",
        color_continuous_scale=[THEME["surface_secondary"], THEME["primary_accent"]],
        title="Recommendation Match Strength (%)",
        labels={"similarity_pct": "Similarity Match (%)", "display_title": "Movie"},
    )
    fig.update_layout(
        **CHART_LAYOUT,
        coloraxis_showscale=False,
        height=max(280, len(rec_df) * 32),
    )
    return fig


# ==============================================================================
# CENTRAL CHART REGISTRY
# ==============================================================================

CHART_REGISTRY: List[ChartDefinition] = [
    ChartDefinition(
        id="genre_bar",
        title="Top Movie Genres",
        category="Genre Analysis",
        description="Frequency count of major genres across the catalog.",
        required_columns=["cleaned_genres"],
        builder_func=plot_genre_bar,
    ),
    ChartDefinition(
        id="genre_share",
        title="Genre Proportions (Donut)",
        category="Genre Analysis",
        description="Proportional share of top movie genres.",
        required_columns=["cleaned_genres"],
        builder_func=plot_genre_share_donut,
    ),
    ChartDefinition(
        id="genre_treemap",
        title="Genre Representation Treemap",
        category="Genre Analysis",
        description="Hierarchical area visualization of catalog genres.",
        required_columns=["cleaned_genres"],
        builder_func=plot_genre_treemap,
    ),
    ChartDefinition(
        id="genre_cooccurrence",
        title="Genre Co-Occurrence Matrix",
        category="Genre Analysis",
        description="Frequency heatmap of genres co-appearing in the same film.",
        required_columns=["cleaned_genres"],
        builder_func=plot_genre_cooccurrence,
    ),
    ChartDefinition(
        id="rating_histogram",
        title="Rating Distribution & Mean",
        category="Rating Analysis",
        description="Histogram of movie ratings with average benchmark line.",
        required_columns=["cleaned_rating"],
        builder_func=plot_rating_histogram,
    ),
    ChartDefinition(
        id="rating_box",
        title="Rating Box Plot & Quartiles",
        category="Rating Analysis",
        description="Quartiles, median, and outlier detection for rating scores.",
        required_columns=["cleaned_rating"],
        builder_func=plot_rating_box,
    ),
    ChartDefinition(
        id="rating_by_genre",
        title="Average Rating by Major Genre",
        category="Rating Analysis",
        description="Comparison of mean user ratings grouped across top genres.",
        required_columns=["cleaned_rating", "cleaned_genres"],
        builder_func=plot_rating_by_genre,
    ),
    ChartDefinition(
        id="rating_vs_votes",
        title="Rating vs. Vote Count Engagement",
        category="Rating Analysis",
        description="Scatter plot comparing rating scores against audience vote counts.",
        required_columns=["cleaned_rating", "cleaned_vote_count"],
        builder_func=plot_rating_vs_votes_scatter,
    ),
    ChartDefinition(
        id="release_year_hist",
        title="Release Year Distribution",
        category="Release Year Trends",
        description="Chronological distribution of movie release dates.",
        required_columns=["cleaned_release_year"],
        builder_func=plot_release_year_histogram,
    ),
    ChartDefinition(
        id="movies_by_decade",
        title="Movies Released by Decade",
        category="Release Year Trends",
        description="Catalog volume grouped across historical cinema decades.",
        required_columns=["cleaned_release_year"],
        builder_func=plot_movies_by_decade,
    ),
    ChartDefinition(
        id="rating_trend_decade",
        title="Rating Trends Over Decades",
        category="Release Year Trends",
        description="Temporal evolution of average movie ratings across decades.",
        required_columns=["cleaned_release_year", "cleaned_rating"],
        builder_func=plot_rating_trend_by_decade,
    ),
    ChartDefinition(
        id="runtime_dist",
        title="Runtime Distribution",
        category="Runtime & Popularity",
        description="Duration spread with median feature runtime.",
        required_columns=["cleaned_runtime"],
        builder_func=plot_runtime_distribution,
    ),
    ChartDefinition(
        id="runtime_vs_rating",
        title="Runtime vs. Rating Correlation",
        category="Runtime & Popularity",
        description="Relationship between movie duration and critical rating.",
        required_columns=["cleaned_runtime", "cleaned_rating"],
        builder_func=plot_runtime_vs_rating_scatter,
    ),
    ChartDefinition(
        id="metadata_completeness",
        title="Metadata Attribute Completeness",
        category="Data Quality",
        description="Attribute-by-attribute coverage percentage for dataset features.",
        required_columns=["display_title"],
        builder_func=plot_metadata_completeness_chart,
    ),
    ChartDefinition(
        id="overview_word_count",
        title="Synopsis Word Count Distribution",
        category="Data Quality",
        description="Length distribution of movie narrative descriptions.",
        required_columns=["cleaned_overview"],
        builder_func=plot_overview_word_count,
    ),
]


def get_available_charts(df: pd.DataFrame) -> List[ChartDefinition]:
    """Return all chart definitions whose required columns are present in the dataset."""
    return [chart for chart in CHART_REGISTRY if chart.is_available(df)]
