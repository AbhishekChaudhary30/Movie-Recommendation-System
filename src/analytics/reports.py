"""Reporting and export generation utilities for analytics and comparisons."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from src.analytics.metrics import (
    compute_overview_metrics,
    generate_executive_insights,
    get_genre_counts,
    get_metadata_completeness,
)


def generate_executive_summary_markdown(
    df: pd.DataFrame,
    dataset_name: str = "Uploaded Dataset",
    cleaning_report: Any = None,
    recs_df: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a comprehensive, publication-ready Executive Analytics Report in Markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = compute_overview_metrics(df)
    insights = generate_executive_insights(df)
    genre_df = get_genre_counts(df, top_n=5)
    comp_df = get_metadata_completeness(df)

    avg_r = f"{metrics.get('avg_rating')}/10" if metrics.get('avg_rating') else "N/A"
    med_r = f"{metrics.get('median_rating')}/10" if metrics.get('median_rating') else "N/A"
    avg_rt = f"{metrics.get('avg_runtime')} minutes" if metrics.get('avg_runtime') else "N/A"
    y_span = f"{metrics.get('min_year')} – {metrics.get('max_year')}" if metrics.get('min_year') else "N/A"
    peak_y = str(metrics.get('most_common_year') or "N/A")
    missing_desc = f"{metrics.get('missing_desc_pct', 0.0)}%"

    lines: List[str] = [
        f"# Executive Movie Analytics & Discovery Report",
        f"**Generated:** {timestamp} | **Dataset:** `{dataset_name}`",
        "",
        "---",
        "",
        "## 1. Key Performance Indicators (KPIs)",
        f"- **Total Catalog Records**: {metrics['total_movies']:,}",
        f"- **Unique Movie Titles**: {metrics['unique_titles']:,}",
        f"- **Distinct Genres Represented**: {metrics['unique_genres']}",
        f"- **Average Rating**: {avg_r} (Median: {med_r})",
        f"- **Average Runtime**: {avg_rt}",
        f"- **Release Year Horizon**: {y_span}",
        f"- **Peak Release Year**: {peak_y}",
        f"- **Missing Synopsis Rate**: {missing_desc}",
        "",
        "## 2. Dynamic Analytical Insights",
    ]

    for item in insights:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## 3. Top Genre Representation",
        "| Genre | Movies Count | Percentage |",
        "| :--- | :--- | :--- |",
    ])

    total_m = metrics["total_movies"]
    for _, row in genre_df.iterrows():
        cnt = row["Count"]
        pct = round((cnt / total_m) * 100, 1) if total_m > 0 else 0
        lines.append(f"| {row['Genre']} | {cnt:,} | {pct}% |")

    lines.extend([
        "",
        "## 4. Metadata Completeness Audit",
        "| Attribute | Available Records | Completeness (%) |",
        "| :--- | :--- | :--- |",
    ])
    for _, row in comp_df.iterrows():
        lines.append(f"| {row['Attribute']} | {row['Available']:,} | {row['Completeness (%)']}% |")

    if cleaning_report is not None:
        lines.extend([
            "",
            "## 5. Cleaning & Quality Audit Summary",
            f"- Original Shape: {cleaning_report.original_rows} rows × {cleaning_report.original_cols} cols",
            f"- Cleaned Shape: {cleaning_report.cleaned_rows} rows × {cleaning_report.cleaned_cols} cols",
            f"- Exact Duplicates Pruned: {cleaning_report.exact_duplicates_removed}",
            f"- Disambiguated Movie Titles: {cleaning_report.potential_duplicate_titles_count}",
        ])

    if recs_df is not None and not recs_df.empty:
        lines.extend([
            "",
            "## 6. Latest Recommendation Query Results",
            "| Rank | Movie Title | Match (%) | Rating | Year | Rationale |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for idx, row in recs_df.head(10).iterrows():
            lines.append(
                f"| #{idx+1} | {row['display_title']} | {row['similarity_pct']}% | "
                f"{row.get('cleaned_rating', 'N/A')} | {row.get('cleaned_release_year', 'N/A')} | "
                f"{row.get('explanation', 'Content similarity')} |"
            )

    lines.extend([
        "",
        "---",
        "*Report compiled automatically by the Intelligent Movie Discovery & Recommendation Analytics Engine.*",
    ])

    return "\n".join(lines)


def build_movie_comparison_matrix(
    df: pd.DataFrame,
    movie_titles: List[str],
    recommender: Any = None,
) -> pd.DataFrame:
    """
    Construct a side-by-side comparison table for a list of movie titles.
    If recommender is provided, computes pairwise similarity against the first movie.
    """
    if not movie_titles or df.empty:
        return pd.DataFrame()

    matched_rows = []
    first_title = movie_titles[0]

    for title in movie_titles:
        # Match by display_title or raw_title
        mask = (df["display_title"].str.lower() == title.lower()) | (df["raw_title"].str.lower() == title.lower())
        sub = df[mask]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            # Compute pairwise similarity against first movie
            if recommender is not None and title != first_title:
                try:
                    sim_info = recommender.compute_pair_similarity(first_title, title)
                    row["similarity_to_seed"] = f"{sim_info['similarity_pct']}%"
                except Exception:
                    row["similarity_to_seed"] = "N/A"
            elif title == first_title:
                row["similarity_to_seed"] = "100.0% (Seed)"
            else:
                row["similarity_to_seed"] = "N/A"
            matched_rows.append(row)

    if not matched_rows:
        return pd.DataFrame()

    comp_df = pd.DataFrame(matched_rows)
    display_cols = [
        "display_title",
        "similarity_to_seed",
        "cleaned_release_year",
        "cleaned_rating",
        "cleaned_runtime",
        "cleaned_genres",
        "cleaned_director",
        "cleaned_cast",
        "cleaned_overview",
    ]
    avail_cols = [c for c in display_cols if c in comp_df.columns]
    rename_map = {
        "display_title": "Movie Title",
        "similarity_to_seed": f"Match vs. '{first_title}'",
        "cleaned_release_year": "Release Year",
        "cleaned_rating": "Rating (0-10)",
        "cleaned_runtime": "Runtime (Min)",
        "cleaned_genres": "Genres",
        "cleaned_director": "Director",
        "cleaned_cast": "Lead Cast",
        "cleaned_overview": "Overview / Plot",
    }
    return comp_df[avail_cols].rename(columns=rename_map)
