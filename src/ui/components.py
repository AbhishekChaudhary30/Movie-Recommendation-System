"""Reusable light-themed UI presentation components for Streamlit."""

from typing import Any, Dict, Optional
import streamlit as st
from src.config import APP_DESCRIPTION, APP_NAME, APP_SUBTITLE, APP_TITLE, APP_VERSION, THEME


def render_hero():
    """Render editorial cinema hero header in Cinematic Neutral aesthetic."""
    st.markdown(
        f"""
        <div style="background-color: {THEME['bg_surface']};
                    border: 1px solid {THEME['border_subtle']};
                    border-radius: 14px;
                    padding: 26px 30px;
                    margin-bottom: 22px;
                    box-shadow: {THEME['card_shadow']};">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <span style="background-color: {THEME['surface_secondary']}; color: {THEME['primary_accent']}; font-size: 0.8rem; font-weight: 700; padding: 4px 12px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.05em;">
                    🎬 {APP_NAME} v{APP_VERSION} &bull; {APP_SUBTITLE}
                </span>
                <span style="font-size: 0.8rem; color: {THEME['text_secondary']}; font-weight: 500;">
                    Content-Based Catalog Modeling &bull; No Private User Tracking
                </span>
            </div>
            <div style="color: {THEME['primary_accent']}; font-size: 2.35rem; font-weight: 800; margin: 4px 0 6px 0; letter-spacing: -0.025em; line-height: 1.15;">
                {APP_NAME}
            </div>
            <div style="color: {THEME['text_secondary']}; font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">
                {APP_SUBTITLE}
            </div>
            <div style="color: {THEME['text_primary']}; font-size: 0.98rem; line-height: 1.55; max-width: 880px;">
                {APP_DESCRIPTION}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: Any, subtext: str = "", border_color: Optional[str] = None):
    """Render styled KPI card using Cinematic Neutral palette."""
    accent = border_color or THEME["border_subtle"]
    st.markdown(
        f"""
        <div style="background-color: {THEME['bg_surface']};
                    border: 1px solid {accent};
                    border-radius: 10px;
                    padding: 16px 18px;
                    box-shadow: {THEME['card_shadow']};
                    margin-bottom: 12px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: {THEME['text_secondary']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                {title}
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; color: {THEME['primary_accent']}; line-height: 1.1;">
                {value}
            </div>
            {f'<div style="font-size: 0.82rem; color: {THEME["text_secondary"]}; margin-top: 4px;">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(icon: str, title: str, description: str, suggestion: str = ""):
    """Render informative empty state when data or results are unavailable."""
    st.markdown(
        f"""
        <div style="background-color: {THEME['bg_surface']};
                    border: 2px dashed {THEME['border_subtle']};
                    border-radius: 12px;
                    padding: 34px 24px;
                    text-align: center;
                    margin: 18px 0;">
            <div style="font-size: 2.6rem; margin-bottom: 10px;">{icon}</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {THEME['text_primary']}; margin-bottom: 6px;">
                {title}
            </div>
            <div style="font-size: 0.94rem; color: {THEME['text_secondary']}; max-width: 580px; margin: 0 auto; line-height: 1.5;">
                {description}
            </div>
            {f'<div style="font-size: 0.88rem; color: {THEME["primary_accent"]}; font-weight: 600; margin-top: 12px;">💡 Suggestion: {suggestion}</div>' if suggestion else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_movie_card(movie: Dict[str, Any], show_overview: bool = True):
    """Render structured movie detail card in Cinematic Neutral style."""
    title = movie.get("display_title", movie.get("raw_title", "Unknown Title"))
    year = movie.get("cleaned_release_year")
    rating = movie.get("cleaned_rating")
    runtime = movie.get("cleaned_runtime")
    genres = movie.get("cleaned_genres", "N/A")
    director = movie.get("cleaned_director", "N/A")
    cast = movie.get("cleaned_cast", "N/A")
    overview = movie.get("cleaned_overview", "No synopsis available.")

    year_str = f"📅 {int(year)}" if pd_notna(year) else "📅 Year N/A"
    rating_str = f"⭐ {rating}/10" if pd_notna(rating) else "⭐ N/A"
    runtime_str = f"⏱️ {int(runtime)}m" if pd_notna(runtime) else "⏱️ N/A"

    st.markdown(
        f"""
        <div style="background-color: {THEME['bg_surface']};
                    border: 1px solid {THEME['border_subtle']};
                    border-radius: 10px;
                    padding: 18px 20px;
                    margin-bottom: 14px;
                    box-shadow: {THEME['card_shadow']};">
            <div style="font-size: 1.25rem; font-weight: 800; color: {THEME['text_primary']}; margin-bottom: 6px;">
                {title}
            </div>
            <div style="margin-bottom: 8px;">
                <span style="background-color: {THEME['surface_secondary']}; color: {THEME['text_primary']}; font-size: 0.82rem; font-weight: 600; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">{year_str}</span>
                <span style="background-color: #EDE8D0; color: #8C6A2E; font-size: 0.82rem; font-weight: 700; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">{rating_str}</span>
                <span style="background-color: {THEME['surface_secondary']}; color: {THEME['text_secondary']}; font-size: 0.82rem; font-weight: 600; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">{runtime_str}</span>
                <span style="background-color: #EFE9EC; color: {THEME['primary_accent']}; font-size: 0.82rem; font-weight: 600; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">🎭 {genres}</span>
            </div>
            <div style="font-size: 0.88rem; color: {THEME['text_secondary']}; margin-bottom: 6px;">
                <b>Director:</b> {director} &nbsp;|&nbsp; <b>Cast:</b> {cast}
            </div>
            {f'<div style="font-size: 0.9rem; color: {THEME["text_primary"]}; line-height: 1.45; margin-top: 8px;">{overview}</div>' if show_overview else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works():
    """Render educational methodology breakdown for recommendation systems."""
    st.markdown(
        f"""
        <div style="background-color: {THEME['bg_surface']}; border: 1px solid {THEME['border_subtle']}; border-radius: 12px; padding: 22px 26px; margin: 20px 0; box-shadow: {THEME['card_shadow']};">
            <div style="font-size: 1.25rem; font-weight: 800; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                💡 How CineScope Content-Based Recommendations Work
            </div>
            <div style="font-size: 0.92rem; color: {THEME['text_secondary']}; margin-bottom: 16px;">
                CineScope uses genuine <b>Content-Based Filtering</b> with mathematical vector space modeling — zero synthetic scores or commercial streaming claims.
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 16px;">
                <div style="background-color: {THEME['bg_main']}; border: 1px solid {THEME['border_subtle']}; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']}; font-size: 0.92rem;">1. Weighted Metadata Soup</b><br>
                    <span style="font-size: 0.85rem; color: {THEME['text_secondary']};">Extracts genres, synopsis, cast, director, and keywords, applying verified weights (e.g., Genre 2.0x, Cast 1.2x).</span>
                </div>
                <div style="background-color: {THEME['bg_main']}; border: 1px solid {THEME['border_subtle']}; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']}; font-size: 0.92rem;">2. Entity Tokenization</b><br>
                    <span style="font-size: 0.85rem; color: {THEME['text_secondary']};">Converts multi-word names like <i>Christopher Nolan</i> into atomic tokens (<code>christopher_nolan</code>) to eliminate surname collisions.</span>
                </div>
                <div style="background-color: {THEME['bg_main']}; border: 1px solid {THEME['border_subtle']}; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']}; font-size: 0.92rem;">3. TF-IDF & Sparse Cosine</b><br>
                    <span style="font-size: 0.85rem; color: {THEME['text_secondary']};">Calculates term frequencies with sublinear scaling and computes exact query similarity via sparse dot products in O(N) memory.</span>
                </div>
                <div style="background-color: {THEME['bg_main']}; border: 1px solid {THEME['border_subtle']}; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']}; font-size: 0.92rem;">4. Explainable Rationale</b><br>
                    <span style="font-size: 0.85rem; color: {THEME['text_secondary']};">Excludes the seed movie, applies user criteria (genre, rating, year), and isolates exact shared genres, cast, and thematic terms.</span>
                </div>
            </div>
            <div style="font-size: 0.84rem; color: {THEME['text_secondary']}; border-top: 1px solid {THEME['border_subtle']}; padding-top: 10px;">
                <b>Note on Similarity vs. User Enjoyment:</b> Similarity scores indicate descriptive proximity in feature space, not a probability that a particular user will like the movie. Content-based systems can overemphasize similar metadata; collaborative filtering (which analyzes user-item interaction histories) can be combined in hybrid architectures.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pd_notna(val: Any) -> bool:
    """Helper to check notna safely without importing pandas everywhere."""
    if val is None:
        return False
    try:
        import numpy as np
        return not (np.isnan(val) if isinstance(val, (int, float, np.number)) else False)
    except Exception:
        return True
