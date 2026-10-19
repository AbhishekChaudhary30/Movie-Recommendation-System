"""Analytics foundation: summary statistics and Plotly visualization builders."""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import THEME


def compute_overview_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute top-level summary metrics from cleaned dataset."""
    total_movies = len(df)
    unique_titles = df["raw_title"].nunique() if "raw_title" in df.columns else total_movies

    # Ratings
    has_ratings = "cleaned_rating" in df.columns and df["cleaned_rating"].notna().sum() > 0
    avg_rating = round(float(df["cleaned_rating"].mean()), 2) if has_ratings else None
    median_rating = round(float(df["cleaned_rating"].median()), 2) if has_ratings else None

    # Runtimes
    has_runtime = "cleaned_runtime" in df.columns and df["cleaned_runtime"].notna().sum() > 0
    avg_runtime = round(float(df["cleaned_runtime"].mean()), 1) if has_runtime else None

    # Release Years
    has_years = "cleaned_release_year" in df.columns and df["cleaned_release_year"].notna().sum() > 0
    min_year = int(df["cleaned_release_year"].min()) if has_years else None
    max_year = int(df["cleaned_release_year"].max()) if has_years else None
    if has_years:
        mode_year_series = df["cleaned_release_year"].dropna().mode()
        most_common_year = int(mode_year_series.iloc[0]) if not mode_year_series.empty else None
    else:
        most_common_year = None

    # Overview completeness
    missing_desc_pct = 0.0
    if "cleaned_overview" in df.columns and total_movies > 0:
        missing_desc = (df["cleaned_overview"].isna() | (df["cleaned_overview"].str.strip() == "")).sum()
        missing_desc_pct = round((missing_desc / total_movies) * 100, 1)

    # Duplicate titles pct
    dup_pct = 0.0
    if total_movies > 0 and unique_titles < total_movies:
        dup_pct = round(((total_movies - unique_titles) / total_movies) * 100, 1)

    # Genres count
    unique_genres = len(get_unique_genres_list(df))

    return {
        "total_movies": total_movies,
        "unique_titles": unique_titles,
        "avg_rating": avg_rating,
        "median_rating": median_rating,
        "avg_runtime": avg_runtime,
        "min_year": min_year,
        "max_year": max_year,
        "most_common_year": most_common_year,
        "missing_desc_pct": missing_desc_pct,
        "duplicate_pct": dup_pct,
        "unique_genres": unique_genres,
    }


def generate_executive_insights(df: pd.DataFrame) -> list[str]:
    """Generate concise, evidence-based observations computed dynamically from the dataset."""
    insights: list[str] = []
    total = len(df)
    if total == 0:
        return ["Dataset is empty. No analytical observations can be drawn."]

    # 1. Dominant Genre Insight
    genre_counts = get_genre_counts(df, top_n=3)
    if not genre_counts.empty:
        top_g = genre_counts.iloc[0]["Genre"]
        top_count = int(genre_counts.iloc[0]["Count"])
        pct = round((top_count / total) * 100, 1)
        insights.append(
            f"**Genre Leadership**: **{top_g}** is the predominant genre with {top_count} movies "
            f"representing {pct}% of the catalog."
        )

    # 2. Rating Concentration
    if "cleaned_rating" in df.columns and df["cleaned_rating"].notna().sum() > 0:
        valid_ratings = df["cleaned_rating"].dropna()
        avg_r = valid_ratings.mean()
        med_r = valid_ratings.median()
        high_rated = (valid_ratings >= 8.0).sum()
        pct_high = round((high_rated / len(valid_ratings)) * 100, 1)
        insights.append(
            f"**Quality Benchmark**: Average rating is **{avg_r:.2f}/10** (median: {med_r:.2f}). "
            f"{pct_high}% of catalog titles ({high_rated} movies) hold an exceptional rating of 8.0 or higher."
        )

    # 3. Era Representation
    if "cleaned_release_year" in df.columns and df["cleaned_release_year"].notna().sum() > 0:
        valid_years = df["cleaned_release_year"].dropna().astype(int)
        min_y, max_y = valid_years.min(), valid_years.max()
        decades = (valid_years // 10) * 10
        top_decade = decades.mode().iloc[0]
        decade_count = (decades == top_decade).sum()
        decade_pct = round((decade_count / len(valid_years)) * 100, 1)
        insights.append(
            f"**Temporal Distribution**: Spanning releases from **{min_y} to {max_y}**, the "
            f"**{top_decade}s** represents the peak cinematic era with {decade_count} titles ({decade_pct}%)."
        )

    # 4. Runtime Profiles
    if "cleaned_runtime" in df.columns and df["cleaned_runtime"].notna().sum() > 0:
        runtimes = df["cleaned_runtime"].dropna()
        avg_rt = runtimes.mean()
        standard_length = ((runtimes >= 90) & (runtimes <= 150)).sum()
        std_pct = round((standard_length / len(runtimes)) * 100, 1)
        insights.append(
            f"**Runtime Profile**: Mean duration is **{avg_rt:.0f} minutes**; {std_pct}% of movies fall within "
            f"the standard feature runtime window of 90–150 minutes."
        )

    # 5. Metadata Integrity
    if "cleaned_overview" in df.columns:
        valid_ov = (df["cleaned_overview"].str.strip() != "").sum()
        ov_pct = round((valid_ov / total) * 100, 1)
        insights.append(
            f"**Content-Based Readiness**: Narrative synopsis coverage is **{ov_pct}%**, providing strong "
            f"semantic signals for TF-IDF feature vectorization."
        )

    return insights


def get_decade_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Group movies by decade."""
    if "cleaned_release_year" not in df.columns:
        return pd.DataFrame(columns=["Decade", "Count"])
    years = df["cleaned_release_year"].dropna().astype(int)
    if years.empty:
        return pd.DataFrame(columns=["Decade", "Count"])
    decades = (years // 10) * 10
    counts = decades.value_counts().sort_index()
    df_decades = pd.DataFrame({
        "Decade": [f"{int(d)}s" for d in counts.index],
        "Count": counts.values,
    })
    return df_decades


def get_metadata_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate completeness percentage for standard movie attributes."""
    attributes = [
        ("Title", "display_title"),
        ("Genres", "cleaned_genres"),
        ("Overview", "cleaned_overview"),
        ("Cast", "cleaned_cast"),
        ("Director", "cleaned_director"),
        ("Rating", "cleaned_rating"),
        ("Release Year", "cleaned_release_year"),
        ("Runtime", "cleaned_runtime"),
        ("Vote Count", "cleaned_vote_count"),
    ]
    total = len(df)
    records = []
    for label, col in attributes:
        if col in df.columns:
            if df[col].dtype == object:
                valid = (df[col].notna() & (df[col].astype(str).str.strip() != "")).sum()
            else:
                valid = df[col].notna().sum()
            pct = round((valid / total) * 100, 1) if total > 0 else 0.0
            records.append({"Attribute": label, "Available": valid, "Completeness (%)": pct})
        else:
            records.append({"Attribute": label, "Available": 0, "Completeness (%)": 0.0})
    return pd.DataFrame(records)


def get_unique_genres_list(df: pd.DataFrame) -> list[str]:
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

    counts: dict[str, int] = {}
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


def plot_genre_distribution(df: pd.DataFrame) -> go.Figure | None:
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
        color_continuous_scale=[THEME["surface_secondary"], THEME["primary_accent"]],
        title="Top Movie Genres in Dataset",
    )
    fig.update_layout(
        plot_bgcolor=THEME["bg_surface"],
        paper_bgcolor=THEME["bg_surface"],
        font=dict(color=THEME["text_primary"]),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
    return fig


def plot_release_year_distribution(df: pd.DataFrame) -> go.Figure | None:
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
        plot_bgcolor=THEME["bg_surface"],
        paper_bgcolor=THEME["bg_surface"],
        font=dict(color=THEME["text_primary"]),
        xaxis_title="Release Year",
        yaxis_title="Number of Movies",
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
    return fig


def plot_rating_distribution(df: pd.DataFrame) -> go.Figure | None:
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
        color_discrete_sequence=[THEME["primary_accent"]],
    )
    fig.update_layout(
        plot_bgcolor=THEME["bg_surface"],
        paper_bgcolor=THEME["bg_surface"],
        font=dict(color=THEME["text_primary"]),
        xaxis_title="Rating Score",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
    return fig


def plot_recommendation_similarity_scores(rec_df: pd.DataFrame) -> go.Figure | None:
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
        color_continuous_scale=[THEME["surface_secondary"], THEME["primary_accent"]],
        title="Recommendation Match Strength (%)",
        labels={"similarity_pct": "Similarity Score (%)", "display_title": "Movie"},
    )
    fig.update_layout(
        plot_bgcolor=THEME["bg_surface"],
        paper_bgcolor=THEME["bg_surface"],
        font=dict(color=THEME["text_primary"]),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=max(280, len(rec_df) * 32),
    )
    return fig
