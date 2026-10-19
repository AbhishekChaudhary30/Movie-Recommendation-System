"""Reusable light-themed UI presentation components for Streamlit."""

from typing import Any, Dict, Optional
import streamlit as st
from src.config import APP_DESCRIPTION, APP_SUBTITLE, APP_TITLE, APP_VERSION, THEME


def render_hero():
    """Render editorial cinema hero header."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%);
                    border: 1px solid {THEME['border_subtle']};
                    border-radius: 16px;
                    padding: 28px 32px;
                    margin-bottom: 24px;
                    box-shadow: {THEME['card_shadow']};">
            <span style="background-color: #EDE9FE; color: {THEME['primary_accent']}; font-size: 0.8rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase;">
                🎬 Cinema Analytics & ML Engine v{APP_VERSION}
            </span>
            <div style="color: {THEME['primary_accent']}; font-size: 2.3rem; font-weight: 800; margin: 10px 0 4px 0; letter-spacing: -0.02em;">
                Discover Your Next Favourite Movie
            </div>
            <div style="color: {THEME['text_secondary']}; font-size: 1.1rem; font-weight: 500; margin-bottom: 12px;">
                {APP_SUBTITLE}
            </div>
            <div style="color: {THEME['text_primary']}; font-size: 0.98rem; line-height: 1.55; max-width: 860px;">
                {APP_DESCRIPTION}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: Any, subtext: str = "", border_color: Optional[str] = None):
    """Render styled KPI card."""
    accent = border_color or THEME["border_subtle"]
    st.markdown(
        f"""
        <div style="background-color: {THEME['bg_surface']};
                    border: 1px solid {accent};
                    border-radius: 12px;
                    padding: 16px 20px;
                    box-shadow: {THEME['card_shadow']};
                    margin-bottom: 12px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: {THEME['text_secondary']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
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
        <div style="background-color: #FFFFFF;
                    border: 2px dashed {THEME['border_subtle']};
                    border-radius: 14px;
                    padding: 36px 24px;
                    text-align: center;
                    margin: 20px 0;">
            <div style="font-size: 2.8rem; margin-bottom: 10px;">{icon}</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {THEME['text_primary']}; margin-bottom: 6px;">
                {title}
            </div>
            <div style="font-size: 0.95rem; color: {THEME['text_secondary']}; max-width: 580px; margin: 0 auto; line-height: 1.5;">
                {description}
            </div>
            {f'<div style="font-size: 0.88rem; color: {THEME["primary_accent"]}; font-weight: 600; margin-top: 12px;">💡 Suggestion: {suggestion}</div>' if suggestion else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_movie_card(movie: Dict[str, Any], show_overview: bool = True):
    """Render structured movie detail card."""
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
        <div style="background-color: #FFFFFF;
                    border: 1px solid {THEME['border_subtle']};
                    border-radius: 12px;
                    padding: 18px 22px;
                    margin-bottom: 16px;
                    box-shadow: {THEME['card_shadow']};">
            <div style="font-size: 1.3rem; font-weight: 800; color: {THEME['text_primary']}; margin-bottom: 6px;">
                {title}
            </div>
            <div style="margin-bottom: 8px;">
                <span style="background-color: #F1F5F9; color: {THEME['text_secondary']}; font-size: 0.82rem; font-weight: 600; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">{year_str}</span>
                <span style="background-color: #FEF3C7; color: #92400E; font-size: 0.82rem; font-weight: 700; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">{rating_str}</span>
                <span style="background-color: #F1F5F9; color: {THEME['text_secondary']}; font-size: 0.82rem; font-weight: 600; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">{runtime_str}</span>
                <span style="background-color: #EDE9FE; color: {THEME['primary_accent']}; font-size: 0.82rem; font-weight: 600; padding: 3px 8px; border-radius: 6px; margin-right: 6px;">🎭 {genres}</span>
            </div>
            <div style="font-size: 0.88rem; color: {THEME['text_secondary']}; margin-bottom: 6px;">
                <b>Director:</b> {director} &nbsp;|&nbsp; <b>Cast:</b> {cast}
            </div>
            {f'<div style="font-size: 0.9rem; color: #334155; line-height: 1.45; margin-top: 8px;">{overview}</div>' if show_overview else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works():
    """Render educational methodology breakdown for recommendation systems."""
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid {THEME['border_subtle']}; border-radius: 14px; padding: 24px 28px; margin: 20px 0; box-shadow: {THEME['card_shadow']};">
            <div style="font-size: 1.3rem; font-weight: 800; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                💡 How Recommendations Work
            </div>
            <div style="font-size: 0.95rem; color: {THEME['text_secondary']}; margin-bottom: 16px;">
                Our recommendation engine uses genuine <b>Content-Based Filtering</b> with mathematical vector space modeling.
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-bottom: 18px;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']};">1. Metadata Soup</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Extracts genres, synopsis, cast, director, and keywords, repeating weighted attributes (e.g. Genre 2x, Cast 1.2x).</span>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']};">2. Entity Tokenization</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Converts multi-word names like <i>Christopher Nolan</i> into joined tokens (<code>christopher_nolan</code>) to prevent accidental name collisions.</span>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']};">3. TF-IDF & Cosine Math</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Calculates term frequencies with sublinear scaling and computes exact cosine similarity via sparse vector dot products in O(N) memory.</span>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;">
                    <b style="color: {THEME['primary_accent']};">4. Multi-Factor Rationale</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Excludes the seed movie, applies user criteria (genre, rating, year), and isolates exact shared genres, cast, and thematic terms.</span>
                </div>
            </div>
            <div style="font-size: 0.86rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 12px;">
                <b>Note on Similarity vs. User Enjoyment:</b> Similarity scores indicate descriptive proximity in feature space, not a probability that a particular user will like the movie. Collaborative filtering (which analyzes user-item interaction histories) can be combined in hybrid architectures.
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
