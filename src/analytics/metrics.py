"""Analytics foundation: summary statistics and Plotly visualization builders."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import THEME


def compute_overview_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute top-level summary metrics from cleaned dataset."""
    total_movies = len(df)
    unique_titles = df["raw_title"].nunique() if "raw_title" in df.columns else total_movies

    # Ratings
    has_ratings = "cleaned_rating" in df.columns and df["cleaned_rating"].notna().sum() > 0
    avg_rating = round(float(df["cleaned_rating"].mean()), 2) if has_ratings else None

    # Runtimes
    has_runtime = "cleaned_runtime" in df.columns and df["cleaned_runtime"].notna().sum() > 0
    avg_runtime = round(float(df["cleaned_runtime"].mean()), 1) if has_runtime else None

    # Release Years
    has_years = "cleaned_release_year" in df.columns and df["cleaned_release_year"].notna().sum() > 0
    min_year = int(df["cleaned_release_year"].min()) if has_years else None
    max_year = int(df["cleaned_release_year"].max()) if has_years else None

    # Genres count
    unique_genres = len(get_unique_genres_list(df))

    return {
        "total_movies": total_movies,
        "unique_titles": unique_titles,
        "avg_rating": avg_rating,
        "avg_runtime": avg_runtime,
        "min_year": min_year,
        "max_year": max_year,
        "unique_genres": unique_genres,
    }


def get_unique_genres_list(df: pd.DataFrame) -> List[str]:
    """Extract sorted list of unique genres present in cleaned dataset."""
    if "cleaned_genres" not in df.columns:
        return []
    genres_set = set()
    for val in df["cleaned_genres"].dropna():
        for g in str(val).split(","):
            g_clean = g.strip().title()
            if g_clean:
                genres_set.add(g_clean)
    return sorted(list(genres_set))


def get_genre_counts(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Compute frequency counts for all genres."""
    if "cleaned_genres" not in df.columns:
        return pd.DataFrame(columns=["Genre", "Count"])

    counts: Dict[str, int] = {}
    for val in df["cleaned_genres"].dropna():
        for g in str(val).split(","):
            g_clean = g.strip().title()
            if g_clean:
                counts[g_clean] = counts.get(g_clean, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["Genre", "Count"])

    df_counts = pd.DataFrame(list(counts.items()), columns=["Genre", "Count"])
    df_counts.sort_values(by="Count", ascending=False, inplace=True)
    return df_counts.head(top_n).reset_index(drop=True)


def plot_genre_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot interactive bar chart of movie genres."""
    df_genres = get_genre_counts(df, top_n=12)
    if df_genres.empty:
        return None

    fig = px.bar(
        df_genres,
        x="Count",
        y="Genre",
        orientation="h",
        color="Count",
        color_continuous_scale=["#A29BFE", THEME["primary_accent"]],
        title="Top Movie Genres in Dataset",
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=THEME["text_primary"]),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
    return fig


def plot_release_year_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot interactive histogram of movie release years."""
    if "cleaned_release_year" not in df.columns:
        return None

    valid_years = df["cleaned_release_year"].dropna().astype(int)
    if len(valid_years) == 0:
        return None

    fig = px.histogram(
        valid_years,
        x="cleaned_release_year",
        nbins=min(30, max(5, valid_years.nunique())),
        title="Release Year Distribution",
        color_discrete_sequence=[THEME["secondary_accent"]],
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=THEME["text_primary"]),
        xaxis_title="Release Year",
        yaxis_title="Number of Movies",
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
    return fig


def plot_rating_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot interactive distribution of movie ratings."""
    if "cleaned_rating" not in df.columns:
        return None

    valid_ratings = df["cleaned_rating"].dropna()
    if len(valid_ratings) == 0:
        return None

    fig = px.histogram(
        valid_ratings,
        x="cleaned_rating",
        nbins=20,
        title="Movie Rating Distribution",
        color_discrete_sequence=[THEME["highlight_amber"]],
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=THEME["text_primary"]),
        xaxis_title="Rating Score",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
    return fig


def plot_recommendation_similarity_scores(rec_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot similarity score comparison for generated recommendations."""
    if rec_df.empty or "similarity_pct" not in rec_df.columns:
        return None

    df_plot = rec_df.sort_values(by="similarity_pct", ascending=True)
    fig = px.bar(
        df_plot,
        x="similarity_pct",
        y="display_title",
        orientation="h",
        color="similarity_pct",
        color_continuous_scale=["#81ECEC", THEME["primary_accent"]],
        title="Recommendation Match Strength (%)",
        labels={"similarity_pct": "Similarity Score (%)", "display_title": "Movie"},
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=THEME["text_primary"]),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=max(280, len(rec_df) * 32),
    )
    return fig
