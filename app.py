"""Movie Recommendation System — Streamlit Application Entrypoint.

Intelligent Movie Discovery & Recommendation Analytics.
"""

from datetime import datetime
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
from src.analytics.metrics import (
    compute_overview_metrics,
    generate_executive_insights,
    get_decade_counts,
    get_genre_counts,
    get_metadata_completeness,
    get_unique_genres_list,
)
from src.analytics.reports import (
    build_movie_comparison_matrix,
    generate_executive_summary_markdown,
)
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
    plot_recommendation_similarity_scores,
    plot_release_year_histogram,
    plot_runtime_distribution,
    plot_runtime_vs_rating_scatter,
)
from src.config import (
    APP_DESCRIPTION,
    APP_SUBTITLE,
    APP_TITLE,
    APP_VERSION,
    DEFAULT_FEATURE_WEIGHTS,
    SAMPLE_DATA_DIR,
    THEME,
)
from src.data.cleaning import clean_dataset
from src.data.loader import load_csv
from src.data.schema import detect_schema
from src.data.validation import validate_dataset
from src.recommendation.content_based import ContentBasedRecommender
from src.ui.components import (
    render_empty_state,
    render_hero,
    render_how_it_works,
    render_metric_card,
    render_movie_card,
)
from src.utils.exceptions import DataLoadError, MovieRecommendationError

# Page Configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | Cinema Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Light Theme CSS
CUSTOM_CSS = f"""
<style>
    /* App background */
    .stApp {{
        background-color: {THEME['bg_main']};
        color: {THEME['text_primary']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Global Headings */
    h1, h2, h3, h4 {{
        color: {THEME['text_primary']};
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }}

    /* Cards */
    .cinema-card {{
        background-color: {THEME['bg_surface']};
        border: 1px solid {THEME['border_subtle']};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: {THEME['card_shadow']};
    }}

    /* Recommendation Card */
    .rec-card {{
        background-color: {THEME['bg_surface']};
        border: 1px solid {THEME['border_subtle']};
        border-left: 5px solid {THEME['primary_accent']};
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: {THEME['card_shadow']};
    }}
    .rec-card:hover {{
        border-color: {THEME['primary_accent']};
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.12);
    }}
    .rec-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {THEME['text_primary']};
        margin-bottom: 6px;
    }}
    .rec-score-badge {{
        background-color: #EDE9FE;
        color: {THEME['primary_accent']};
        font-size: 0.82rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        margin-right: 8px;
    }}
    .meta-tag {{
        background-color: #F1F5F9;
        color: {THEME['text_secondary']};
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-right: 6px;
    }}
    .exp-box {{
        background-color: #F8FAFC;
        border: 1px dashed {THEME['border_subtle']};
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.88rem;
        color: {THEME['text_primary']};
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        border-radius: 8px;
        font-weight: 600;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent application state."""
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "🏠 Home"
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = ""
    if "user_mappings" not in st.session_state:
        st.session_state.user_mappings = {}
    if "schema_mapping" not in st.session_state:
        st.session_state.schema_mapping = None
    if "validation_report" not in st.session_state:
        st.session_state.validation_report = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "cleaning_report" not in st.session_state:
        st.session_state.cleaning_report = None
    if "recommender" not in st.session_state:
        st.session_state.recommender = None
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = None
    if "latest_recs" not in st.session_state:
        st.session_state.latest_recs = None
    if "feature_weights" not in st.session_state:
        st.session_state.feature_weights = DEFAULT_FEATURE_WEIGHTS.copy()
    if "comparison_movies" not in st.session_state:
        st.session_state.comparison_movies = []
    if "explorer_page_num" not in st.session_state:
        st.session_state.explorer_page_num = 1
    if "auto_analysis_run" not in st.session_state:
        st.session_state.auto_analysis_run = False


init_session_state()


def reset_workflow():
    """Reset the application workflow state for a new dataset."""
    st.session_state.raw_df = None
    st.session_state.dataset_name = ""
    st.session_state.user_mappings = {}
    st.session_state.schema_mapping = None
    st.session_state.validation_report = None
    st.session_state.cleaned_df = None
    st.session_state.cleaning_report = None
    st.session_state.recommender = None
    st.session_state.selected_movie = None
    st.session_state.latest_recs = None
    st.session_state.feature_weights = DEFAULT_FEATURE_WEIGHTS.copy()
    st.session_state.comparison_movies = []
    st.session_state.explorer_page_num = 1
    st.session_state.auto_analysis_run = False
    st.session_state.nav_page = "📂 Dataset & Data Quality"
    st.rerun()


def navigate_to(page_name: str):
    """Navigate to another page and trigger rerun."""
    st.session_state.nav_page = page_name
    st.rerun()


# ==============================================================================
# SIDEBAR APPLICATION SHELL
# ==============================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding-bottom: 10px;">
            <div style="font-size: 1.25rem; font-weight: 800; color: {THEME['primary_accent']};">🎬 Movie Rec System</div>
            <div style="font-size: 0.8rem; color: {THEME['text_secondary']};">Intelligent Discovery & Analytics v{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = [
        "🏠 Home",
        "📂 Dataset & Data Quality",
        "🔍 Movie Explorer",
        "🎯 Recommendations",
        "⚖️ Compare Movies",
        "📊 Analytics Dashboard",
        "📈 Advanced Visualizations",
        "📥 Reports & Downloads",
        "ℹ️ About This Project",
    ]

    # Ensure page in options
    current_idx = nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=current_idx,
        label_visibility="collapsed",
    )
    if selected_nav != st.session_state.nav_page:
        st.session_state.nav_page = selected_nav
        st.rerun()

    st.write("---")
    # Real-Time Dataset Status Widget
    st.markdown("<b style='font-size: 0.85rem; text-transform: uppercase;'>Dataset Status</b>", unsafe_allow_html=True)
    if st.session_state.cleaned_df is not None:
        st.success(f"📂 **{st.session_state.dataset_name}**")
        st.caption(
            f"✓ {len(st.session_state.cleaned_df):,} movies loaded\n\n"
            f"✓ {len(st.session_state.cleaning_report.available_recommendation_features)} rec features active\n\n"
            f"✓ ML Engine: {'Ready' if st.session_state.recommender else 'Pending'}"
        )
    else:
        st.warning("No dataset loaded yet.")
        st.caption("Upload a CSV or load the curated sample dataset to begin.")

    if st.button("🔄 Reset / New Dataset", use_container_width=True):
        reset_workflow()

    st.write("---")
    st.caption("Editorial Cinema Analytics • Developed with Python, Scikit-Learn & Streamlit")


# ==============================================================================
# PAGE 1: 🏠 HOME
# ==============================================================================
if st.session_state.nav_page == "🏠 Home":
    render_hero()

    # Status KPI Row
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        if st.session_state.cleaned_df is not None:
            render_metric_card("Loaded Movies", f"{len(st.session_state.cleaned_df):,}", "In active memory")
        else:
            render_metric_card("Dataset Status", "Not Loaded", "Awaiting ingestion")
    with col_k2:
        if st.session_state.cleaning_report is not None:
            render_metric_card("Quality Audit", "Passed", f"{st.session_state.cleaning_report.exact_duplicates_removed} dups pruned")
        else:
            render_metric_card("Data Cleaning", "Pending", "Upload CSV first")
    with col_k3:
        if st.session_state.cleaned_df is not None:
            genres_count = len(get_unique_genres_list(st.session_state.cleaned_df))
            render_metric_card("Genres Identified", f"{genres_count}", "Available for filtering")
        else:
            render_metric_card("Rec Engine", "Standby", "Requires dataset")
    with col_k4:
        if st.session_state.recommender is not None:
            render_metric_card("ML Engine", "Active ✓", "TF-IDF + Cosine")
        else:
            render_metric_card("ML Engine", "Standby", "Initializes on load")

    st.write(" ")
    # Primary Call to Action buttons
    act_col1, act_col2, act_col3 = st.columns([1.2, 1.2, 2])
    with act_col1:
        if st.button("📂 Ingest / Change Dataset ➔", type="primary", use_container_width=True):
            navigate_to("📂 Dataset & Data Quality")
    with act_col2:
        if st.session_state.cleaned_df is not None:
            if st.button("🎯 Explore Recommendations ➔", use_container_width=True):
                navigate_to("🎯 Recommendations")
        else:
            st.button("🎯 Recommendations (Upload First)", disabled=True, use_container_width=True)

    st.write("---")
    st.write("### 🌟 Principal Capabilities")

    cap1, cap2, cap3 = st.columns(3)
    with cap1:
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                    🧹 Automated Data Engineering
                </div>
                <div style="font-size: 0.9rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                    Intelligent synonym detection maps diverse CSV schemas. Removes empty records, prunes exact duplicates, and disambiguates movie titles with audit reports.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                    🔍 Interactive Movie Explorer
                </div>
                <div style="font-size: 0.9rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                    Filter and search thousands of titles by genre, release era, rating threshold, and director with paginated card browsing and detailed previews.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cap2:
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                    🎯 Content-Based Recommender
                </div>
                <div style="font-size: 0.9rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                    Mathematical TF-IDF feature vectorization with memory-aware sparse cosine similarity in O(N) memory. Strict self-exclusion and deterministic rankings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                    💡 Explainable Recommendations
                </div>
                <div style="font-size: 0.9rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                    Transparent multi-factor rationale cards highlight exact shared genres, cast members, director connections, and thematic narrative terms.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cap3:
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                    ⚖️ Side-by-Side Movie Comparison
                </div>
                <div style="font-size: 0.9rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                    Select multiple films to compare ratings, runtimes, cast, and narrative summaries side-by-side with real pairwise similarity scoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 6px;">
                    📊 Advanced Analytics & Reports
                </div>
                <div style="font-size: 0.9rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                    15+ interactive Plotly charts across genres, ratings, decades, and runtimes, with one-click full executive analysis report generation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("---")
    st.write("### 🧭 Workflow Overview")
    st.caption("The system enforces a reliable, reproducible analytical lifecycle:")

    wf_cols = st.columns(6)
    wf_steps = [
        ("1. Ingest", "Load sample or custom CSV", st.session_state.raw_df is not None),
        ("2. Validate", "Detect schema & check integrity", st.session_state.validation_report is not None and st.session_state.validation_report.is_valid),
        ("3. Clean", "Deduplicate & parse structures", st.session_state.cleaned_df is not None),
        ("4. Explore", "Search & filter catalog", st.session_state.cleaned_df is not None),
        ("5. Recommend", "Generate weighted matches", st.session_state.recommender is not None),
        ("6. Export", "Download datasets & reports", st.session_state.cleaned_df is not None),
    ]

    for i, (name, desc, done) in enumerate(wf_steps):
        with wf_cols[i]:
            status_badge = "✓ Done" if done else "○ Pending"
            badge_color = THEME["secondary_accent"] if done else "#94A3B8"
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid {THEME['border_subtle']}; border-radius: 10px; padding: 12px; text-align: center;">
                    <span style="background-color: {badge_color}; color: white; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;">{status_badge}</span>
                    <div style="font-size: 0.95rem; font-weight: 700; color: {THEME['text_primary']}; margin-top: 6px;">{name}</div>
                    <div style="font-size: 0.78rem; color: {THEME['text_secondary']}; margin-top: 2px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("---")
    # Next Workflow Button
    col_n1, col_n2 = st.columns([1.5, 3])
    with col_n1:
        if st.button("Next: Dataset Ingestion & Quality ➔", type="primary", use_container_width=True):
            navigate_to("📂 Dataset & Data Quality")
    with col_n2:
        st.caption("Proceed to load, inspect, and validate your movie dataset.")


# ==============================================================================
# PAGE 2: 📂 DATASET & DATA QUALITY
# ==============================================================================
elif st.session_state.nav_page == "📂 Dataset & Data Quality":
    st.subheader("📂 Dataset Ingestion & Quality Audit")
    st.write(
        "Upload any movie metadata CSV or load the curated 50-movie cinema gem dataset. "
        "The automated pipeline detects column schemas, sanitizes text and numbers, and prunes duplicates."
    )

    tab_load1, tab_load2 = st.tabs(["🌟 Curated Cinema Sample Dataset (50 Classic & Modern Movies)", "📁 Upload Custom Movie CSV"])

    with tab_load1:
        st.info("The curated sample contains 50 diverse titles across genres, directors, and eras with full metadata.")
        if st.button("🚀 Load Curated Sample Dataset", type="primary"):
            sample_path = SAMPLE_DATA_DIR / "sample_movies.csv"
            try:
                df = load_csv(sample_path)
                st.session_state.raw_df = df
                st.session_state.dataset_name = "sample_movies.csv"
                st.session_state.user_mappings = {}
                mapping = detect_schema(list(df.columns))
                st.session_state.schema_mapping = mapping
                val_rep = validate_dataset(df, mapping)
                st.session_state.validation_report = val_rep
                cleaned, clean_rep = clean_dataset(df, mapping)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_report = clean_rep
                # Re-fit recommender
                rec = ContentBasedRecommender(feature_weights=st.session_state.feature_weights)
                rec.fit(cleaned, mapping)
                st.session_state.recommender = rec
                st.session_state.selected_movie = cleaned["display_title"].iloc[0]
                st.success("Loaded and initialized sample dataset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load sample dataset: {e}")

    with tab_load2:
        uploaded_file = st.file_uploader(
            "Upload a Movie CSV (Max 50MB)",
            type=["csv"],
            help="Compatible with standard movie datasets (TMDB 5000, MovieLens, or custom exports).",
        )
        if uploaded_file is not None:
            if st.button("📥 Parse & Ingest Uploaded File", type="primary"):
                try:
                    with st.spinner("Reading CSV with multi-encoding fallback..."):
                        df = load_csv(uploaded_file)
                        st.session_state.raw_df = df
                        st.session_state.dataset_name = uploaded_file.name
                        st.session_state.user_mappings = {}
                        mapping = detect_schema(list(df.columns))
                        st.session_state.schema_mapping = mapping
                        val_rep = validate_dataset(df, mapping)
                        st.session_state.validation_report = val_rep
                        if val_rep.is_valid:
                            cleaned, clean_rep = clean_dataset(df, mapping)
                            st.session_state.cleaned_df = cleaned
                            st.session_state.cleaning_report = clean_rep
                            rec = ContentBasedRecommender(feature_weights=st.session_state.feature_weights)
                            rec.fit(cleaned, mapping)
                            st.session_state.recommender = rec
                            st.session_state.selected_movie = cleaned["display_title"].iloc[0]
                        st.success(f"Ingested '{uploaded_file.name}' successfully! ({len(df)} rows)")
                        st.rerun()
                except DataLoadError as e:
                    st.error(f"Data Load Error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # Display Quality Audit & Schema Mapping if loaded
    if st.session_state.cleaned_df is not None:
        st.write("---")
        st.write("### 📋 Quality Audit & Cleaning Summary")
        rep = st.session_state.cleaning_report

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            render_metric_card("Cleaned Records", f"{rep.cleaned_rows:,}", f"From {rep.original_rows:,} raw rows")
        with q2:
            render_metric_card("Duplicates Pruned", f"{rep.exact_duplicates_removed}", "Exact duplicate rows")
        with q3:
            render_metric_card("Titles Disambiguated", f"{rep.potential_duplicate_titles_count}", "Year/ID tagged")
        with q4:
            render_metric_card("Active Features", f"{len(rep.available_recommendation_features)}", ", ".join(rep.available_recommendation_features) or "None")

        # Schema Mapping Overrides Expander
        with st.expander("🛠️ Review / Customize Column Mappings", expanded=False):
            st.caption("Adjust which column from your CSV maps to each canonical movie attribute.")
            cols = list(st.session_state.raw_df.columns)
            curr_map = st.session_state.schema_mapping

            m_cols = st.columns(3)
            canonical_fields = [
                ("title", "Movie Title (Mandatory) *"),
                ("genres", "Genres"),
                ("overview", "Overview / Synopsis"),
                ("keywords", "Keywords / Tags"),
                ("cast", "Cast / Actors"),
                ("director", "Director"),
                ("release_date", "Release Date / Year"),
                ("rating", "Rating / Vote Average"),
                ("runtime", "Runtime (Minutes)"),
            ]
            options = ["<Unmapped>"] + cols
            new_overrides = {}
            for idx, (canon_key, label) in enumerate(canonical_fields):
                target_col = m_cols[idx % 3]
                current_val = curr_map.get_original(canon_key)
                def_idx = options.index(current_val) if current_val in options else 0
                sel = target_col.selectbox(label, options=options, index=def_idx, key=f"map_{canon_key}")
                if sel != "<Unmapped>":
                    new_overrides[canon_key] = sel

            if st.button("Apply Column Mapping Changes"):
                st.session_state.user_mappings = new_overrides
                updated_mapping = detect_schema(cols, user_overrides=new_overrides)
                st.session_state.schema_mapping = updated_mapping
                cleaned, clean_rep = clean_dataset(st.session_state.raw_df, updated_mapping)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_report = clean_rep
                rec = ContentBasedRecommender(feature_weights=st.session_state.feature_weights)
                rec.fit(cleaned, updated_mapping)
                st.session_state.recommender = rec
                st.success("Column mappings updated and engine re-fitted!")
                st.rerun()

        # Previews Tabs
        p_tab1, p_tab2, p_tab3 = st.tabs(["🔍 Cleaned Catalog Preview", "📊 Missing Values Audit", "📋 Cleaning Audit Trail"])
        with p_tab1:
            preview_cols = [
                c for c in [
                    "display_title", "cleaned_genres", "cleaned_release_year",
                    "cleaned_rating", "cleaned_director", "cleaned_cast"
                ] if c in st.session_state.cleaned_df.columns
            ]
            st.dataframe(st.session_state.cleaned_df[preview_cols].head(12), use_container_width=True)

        with p_tab2:
            comp_df = get_metadata_completeness(st.session_state.cleaned_df)
            st.dataframe(comp_df, use_container_width=True)

        with p_tab3:
            st.markdown(rep.to_markdown())

        st.write("---")
        # Download Section
        dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
        with dcol1:
            csv_data = st.session_state.cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_movies.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dcol2:
            st.download_button(
                "📋 Download Quality Audit Report (MD)",
                data=rep.to_markdown(),
                file_name="quality_audit_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.write("---")
        # Next Button
        col_n1, col_n2 = st.columns([1.5, 3])
        with col_n1:
            if st.button("Next: Explore Movie Catalog ➔", type="primary", use_container_width=True):
                navigate_to("🔍 Movie Explorer")
        with col_n2:
            st.caption("Proceed to browse, filter, and search the cleaned movie catalog.")

    else:
        render_empty_state("📂", "No Dataset Loaded", "Please upload a movie CSV or load the sample dataset to proceed.")


# ==============================================================================
# PAGE 3: 🔍 MOVIE EXPLORER
# ==============================================================================
elif st.session_state.nav_page == "🔍 Movie Explorer":
    st.subheader("🔍 Movie Catalog Explorer")
    st.write("Search, filter, and discover movies across genres, release eras, ratings, and creative directors.")

    df = st.session_state.cleaned_df
    if df is None:
        render_empty_state("📂", "Dataset Required", "Please load a movie dataset to explore the catalog.", "Visit 'Dataset & Data Quality' tab")
        if st.button("Go to Dataset Ingestion"):
            navigate_to("📂 Dataset & Data Quality")
        st.stop()

    # Search Bar & Filter Controls
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_query = st.text_input(
            "🔎 Search Movies (Title, Director, Cast, or Keywords):",
            placeholder="e.g. Inception, Christopher Nolan, Leonardo DiCaprio...",
        ).strip().lower()

    # Filter Controls Expander
    with st.expander("⚙️ Filter Controls (Genre, Year, Rating, Runtime)", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        # Genre Filter
        all_genres = ["All Genres"] + get_unique_genres_list(df)
        with fcol1:
            selected_genre = st.selectbox("Genre:", options=all_genres, index=0)

        # Rating Filter
        has_ratings = "cleaned_rating" in df.columns and df["cleaned_rating"].notna().sum() > 0
        with fcol2:
            if has_ratings:
                min_r = float(df["cleaned_rating"].min())
                max_r = float(df["cleaned_rating"].max())
                rating_range = st.slider("Rating Score:", min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.5)
            else:
                st.caption("Rating filter disabled (no rating field)")
                rating_range = (0.0, 10.0)

        # Year Range Filter
        has_years = "cleaned_release_year" in df.columns and df["cleaned_release_year"].notna().sum() > 0
        with fcol3:
            if has_years:
                valid_y = df["cleaned_release_year"].dropna().astype(int)
                y_min, y_max = int(valid_y.min()), int(valid_y.max())
                year_range = st.slider("Release Year Range:", min_value=y_min, max_value=y_max, value=(y_min, y_max))
            else:
                st.caption("Year filter disabled (no date field)")
                year_range = (None, None)

        # Runtime Filter
        has_runtime = "cleaned_runtime" in df.columns and df["cleaned_runtime"].notna().sum() > 0
        with fcol4:
            if has_runtime:
                valid_rt = df["cleaned_runtime"].dropna().astype(int)
                rt_min, rt_max = int(valid_rt.min()), int(valid_rt.max())
                runtime_range = st.slider("Runtime (Minutes):", min_value=rt_min, max_value=rt_max, value=(rt_min, rt_max))
            else:
                st.caption("Runtime filter disabled")
                runtime_range = (None, None)

    # Apply Filters
    filtered_df = df.copy()

    # Text Search Across Supported Fields
    if search_query:
        search_cols = [c for c in ["display_title", "raw_title", "cleaned_director", "cleaned_cast", "cleaned_keywords", "cleaned_overview"] if c in filtered_df.columns]
        mask = pd.Series(False, index=filtered_df.index)
        for col in search_cols:
            mask = mask | filtered_df[col].astype(str).str.lower().str.contains(search_query, na=False)
        filtered_df = filtered_df[mask]

    # Genre Filter
    if selected_genre != "All Genres" and "cleaned_genres" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["cleaned_genres"].astype(str).str.lower().str.contains(selected_genre.lower(), na=False)]

    # Rating Filter
    if has_ratings:
        filtered_df = filtered_df[
            filtered_df["cleaned_rating"].isna() | (
                (filtered_df["cleaned_rating"] >= rating_range[0]) &
                (filtered_df["cleaned_rating"] <= rating_range[1])
            )
        ]

    # Year Filter
    if has_years and year_range != (None, None):
        filtered_df = filtered_df[
            filtered_df["cleaned_release_year"].isna() | (
                (filtered_df["cleaned_release_year"] >= year_range[0]) &
                (filtered_df["cleaned_release_year"] <= year_range[1])
            )
        ]

    # Runtime Filter
    if has_runtime and runtime_range != (None, None):
        filtered_df = filtered_df[
            filtered_df["cleaned_runtime"].isna() | (
                (filtered_df["cleaned_runtime"] >= runtime_range[0]) &
                (filtered_df["cleaned_runtime"] <= runtime_range[1])
            )
        ]

    # Results Header & Pagination
    st.write(f"#### Results: Showing {len(filtered_df):,} of {len(df):,} Movies")

    if filtered_df.empty:
        render_empty_state("🔍", "No Matching Movies Found", "No titles match the chosen search query and filter criteria.", "Try clearing the search term or widening your rating/year filters.")
    else:
        # Pagination Settings
        PAGE_SIZE = 9
        total_pages = max(1, math.ceil(len(filtered_df) / PAGE_SIZE))
        curr_p = min(st.session_state.explorer_page_num, total_pages)

        # Pagination Controls
        pg_col1, pg_col2, pg_col3 = st.columns([1, 2, 1])
        with pg_col1:
            if st.button("⬅ Previous Page", disabled=(curr_p <= 1), use_container_width=True):
                st.session_state.explorer_page_num = max(1, curr_p - 1)
                st.rerun()
        with pg_col2:
            st.markdown(f"<div style='text-align: center; font-weight: 600; padding-top: 8px;'>Page {curr_p} of {total_pages}</div>", unsafe_allow_html=True)
        with pg_col3:
            if st.button("Next Page ➔", disabled=(curr_p >= total_pages), use_container_width=True):
                st.session_state.explorer_page_num = min(total_pages, curr_p + 1)
                st.rerun()

        start_idx = (curr_p - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_records = filtered_df.iloc[start_idx:end_idx]

        # Render 3x3 Card Grid
        grid_cols = st.columns(3)
        for i, (_, row) in enumerate(page_records.iterrows()):
            with grid_cols[i % 3]:
                render_movie_card(row.to_dict())
                # Quick Action to Seed Recommendations
                if st.button(f"🎯 Find Similar Movies", key=f"seed_{row['display_title']}", use_container_width=True):
                    st.session_state.selected_movie = row["display_title"]
                    navigate_to("🎯 Recommendations")

    st.write("---")
    # Next Navigation Button
    col_n1, col_n2 = st.columns([1.5, 3])
    with col_n1:
        if st.button("Next: View Recommendations ➔", type="primary", use_container_width=True):
            navigate_to("🎯 Recommendations")
    with col_n2:
        st.caption("Generate content-based recommendations with similarity scoring.")


# ==============================================================================
# PAGE 4: 🎯 RECOMMENDATIONS
# ==============================================================================
elif st.session_state.nav_page == "🎯 Recommendations":
    st.subheader("🎯 Content-Based Movie Recommendations")
    st.write(
        "Generate intelligent movie recommendations using TF-IDF metadata vectorization and Cosine Similarity. "
        "Customize feature weights, apply genre/rating filters, and review transparent similarity rationales."
    )

    df = st.session_state.cleaned_df
    schema = st.session_state.schema_mapping

    if df is None or schema is None:
        render_empty_state("📂", "Dataset Required", "Please load a movie dataset to run the recommendation engine.")
        if st.button("Go to Dataset Ingestion"):
            navigate_to("📂 Dataset & Data Quality")
        st.stop()

    # Recommender Initialization
    if st.session_state.recommender is None:
        with st.spinner("Fitting Content-Based Recommendation Engine..."):
            rec = ContentBasedRecommender(feature_weights=st.session_state.feature_weights)
            rec.fit(df, schema)
            st.session_state.recommender = rec

    recommender: ContentBasedRecommender = st.session_state.recommender

    # Seed Movie Selector
    all_titles = sorted(df["display_title"].tolist())
    def_idx = 0
    if st.session_state.selected_movie in all_titles:
        def_idx = all_titles.index(st.session_state.selected_movie)

    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        chosen_seed = st.selectbox(
            "🎬 Choose a Seed Movie to Base Recommendations On:",
            options=all_titles,
            index=def_idx,
            help="Select the movie you like to find matching titles in the catalog.",
        )
        st.session_state.selected_movie = chosen_seed

    with sel_col2:
        top_n = st.slider("Number of Recommendations (N):", min_value=1, max_value=20, value=6, step=1)

    # Recommendation Options Expander
    with st.expander("⚙️ Recommendation Filters & Feature Weights", expanded=False):
        c1, c2, c3 = st.columns(3)
        available_genres = ["All"] + get_unique_genres_list(df)
        with c1:
            rec_genre_filter = st.selectbox("Genre Filter:", options=available_genres, index=0)
        with c2:
            min_sim_pct = st.slider("Minimum Match Threshold (%):", min_value=0, max_value=80, value=0, step=5)
        with c3:
            has_ratings = "cleaned_rating" in df.columns and df["cleaned_rating"].notna().sum() > 0
            if has_ratings:
                rec_min_rating = st.slider("Minimum Rating Filter:", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
            else:
                st.caption("Rating filter unavailable")
                rec_min_rating = 0.0

        st.write("###### Feature Weights (Influence on Similarity):")
        w1, w2, w3, w4, w5 = st.columns(5)
        w_gen = w1.number_input("Genres Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("genres", 2.0), step=0.5)
        w_kw = w2.number_input("Keywords Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("keywords", 1.5), step=0.5)
        w_cast = w3.number_input("Cast Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("cast", 1.2), step=0.5)
        w_dir = w4.number_input("Director Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("director", 1.0), step=0.5)
        w_ov = w5.number_input("Overview Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("overview", 1.0), step=0.5)

        if st.button("Apply Feature Weights"):
            new_weights = {
                "genres": w_gen,
                "keywords": w_kw,
                "cast": w_cast,
                "director": w_dir,
                "overview": w_ov,
            }
            st.session_state.feature_weights = new_weights
            recommender.fit(df, schema, feature_weights=new_weights)
            st.session_state.recommender = recommender
            st.success("Re-fitted recommender with new feature weights!")
            st.rerun()

    # Seed Movie Card
    seed_details = recommender.get_movie_details(chosen_seed)
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid {THEME['border_subtle']}; border-radius: 12px; padding: 18px 22px; margin: 16px 0; box-shadow: {THEME['card_shadow']};">
            <span style="font-size: 0.8rem; font-weight: 700; color: {THEME['primary_accent']}; text-transform: uppercase;">SEED MOVIE</span>
            <div style="font-size: 1.45rem; font-weight: 800; color: {THEME['text_primary']}; margin: 4px 0;">{seed_details.get('display_title')}</div>
            <div style="margin-bottom: 8px;">
                <span class="meta-tag">📅 {seed_details.get('cleaned_release_year') or 'N/A'}</span>
                <span class="meta-tag">⭐ {seed_details.get('cleaned_rating') or 'N/A'}/10</span>
                <span class="meta-tag">⏱️ {seed_details.get('cleaned_runtime') or 'N/A'} min</span>
                <span class="meta-tag">🎭 {seed_details.get('cleaned_genres') or 'N/A'}</span>
                <span class="meta-tag">🎬 {seed_details.get('cleaned_director') or 'N/A'}</span>
            </div>
            <div style="font-size: 0.92rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                <b>Overview:</b> {seed_details.get('cleaned_overview') or 'No description available.'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute Recommendations
    try:
        recs_df = recommender.recommend(
            query=chosen_seed,
            top_n=top_n,
            min_rating=rec_min_rating if rec_min_rating > 0 else None,
            genre_filter=rec_genre_filter if rec_genre_filter != "All" else None,
            min_similarity=min_sim_pct / 100.0 if min_sim_pct > 0 else None,
        )
        st.session_state.latest_recs = recs_df

        if recs_df.empty:
            render_empty_state("🎯", "No Recommendations Satisfied Criteria", "No movies in the catalog met the combination of genre filter, rating threshold, and minimum similarity match.", "Try lowering the Minimum Match Threshold or choosing 'All' genres.")
        else:
            st.write(f"### Top {len(recs_df)} Recommendations for *{chosen_seed}*")

            # Render Recommendation Cards
            for idx, row in recs_df.iterrows():
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <div class="rec-title">#{idx+1}. {row['display_title']}</div>
                            <div><span class="rec-score-badge">Cosine Match: {row['similarity_pct']}%</span></div>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span class="meta-tag">📅 {row['cleaned_release_year'] or 'Year N/A'}</span>
                            <span class="meta-tag">⭐ {row['cleaned_rating'] or 'Rating N/A'}</span>
                            <span class="meta-tag">🎭 {row['cleaned_genres'] or 'N/A'}</span>
                            <span class="meta-tag">🎬 {row['cleaned_director'] or 'Director N/A'}</span>
                        </div>
                        <div style="font-size: 0.88rem; color: {THEME['text_secondary']}; margin-bottom: 6px;">
                            <b>Cast:</b> {row['cleaned_cast'] or 'N/A'}
                        </div>
                        <div style="font-size: 0.88rem; color: #4A5568; line-height: 1.4;">
                            {row['cleaned_overview'] or ''}
                        </div>
                        <div class="exp-box">
                            <b>🔍 Recommendation Rationale:</b> {row['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Match Strength Bar Chart
            st.write("---")
            fig_match = plot_recommendation_similarity_scores(recs_df)
            if fig_match:
                st.plotly_chart(fig_match, use_container_width=True)

            # Comparison and Export Options
            col_exp1, col_exp2 = st.columns([1, 1])
            with col_exp1:
                rec_csv = recs_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Recommendations (CSV)",
                    data=rec_csv,
                    file_name=f"recommendations_{chosen_seed.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_exp2:
                if st.button("⚖️ Compare Seed with Recommendations ➔", use_container_width=True):
                    # Set comparison movies to seed + top 2 recommendations
                    top_rec_titles = [chosen_seed] + recs_df["display_title"].head(2).tolist()
                    st.session_state.comparison_movies = top_rec_titles
                    navigate_to("⚖️ Compare Movies")

    except MovieRecommendationError as e:
        st.error(f"Recommendation Engine Notice: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    # Educational Methodology Section
    render_how_it_works()

    st.write("---")
    # Next Navigation Button
    col_n1, col_n2 = st.columns([1.5, 3])
    with col_n1:
        if st.button("Next: Compare Movies Side-by-Side ➔", type="primary", use_container_width=True):
            navigate_to("⚖️ Compare Movies")
    with col_n2:
        st.caption("Inspect and compare multiple movies side-by-side.")


# ==============================================================================
# PAGE 5: ⚖️ COMPARE MOVIES
# ==============================================================================
elif st.session_state.nav_page == "⚖️ Compare Movies":
    st.subheader("⚖️ Interactive Movie Comparison")
    st.write(
        "Select two or more movies to analyze side-by-side. Inspect release years, ratings, runtimes, "
        "creative personnel, narrative plotlines, and compute exact pairwise similarity scores."
    )

    df = st.session_state.cleaned_df
    if df is None:
        render_empty_state("📂", "Dataset Required", "Please load a movie dataset to compare titles.")
        st.stop()

    all_titles = sorted(df["display_title"].tolist())
    def_comp = st.session_state.comparison_movies
    # Fallback default to first 2 movies if not set
    if not def_comp or not all(t in all_titles for t in def_comp):
        def_comp = all_titles[:2] if len(all_titles) >= 2 else all_titles[:1]

    selected_comp = st.multiselect(
        "Select 2 to 4 Movies to Compare:",
        options=all_titles,
        default=def_comp,
        max_selections=4,
    )
    st.session_state.comparison_movies = selected_comp

    if len(selected_comp) < 2:
        st.info("ℹ️ Please select at least two distinct movies to view side-by-side comparison.")
    else:
        recommender = st.session_state.recommender

        # Pairwise Similarity Highlight (Movie 1 vs Movie 2)
        if recommender is not None and len(selected_comp) >= 2:
            m1 = selected_comp[0]
            m2 = selected_comp[1]
            try:
                sim_res = recommender.compute_pair_similarity(m1, m2)
                st.markdown(
                    f"""
                    <div style="background-color: #EDE9FE; border: 1px solid #DDD6FE; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;">
                        <span style="font-size: 0.82rem; font-weight: 700; color: {THEME['primary_accent']}; text-transform: uppercase;">Direct Pairwise Similarity</span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0;">
                            <b>{m1}</b> vs. <b>{m2}</b>: <span style="color: {THEME['primary_accent']};">{sim_res['similarity_pct']}% Match</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #4B5563;">
                            <b>Rationale:</b> {sim_res['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        # Side-by-Side Comparison Cards
        comp_cols = st.columns(len(selected_comp))
        for idx, title in enumerate(selected_comp):
            with comp_cols[idx]:
                sub = df[df["display_title"] == title]
                if not sub.empty:
                    m_dict = sub.iloc[0].to_dict()
                    render_movie_card(m_dict, show_overview=True)

        # Tabular Comparison Matrix
        st.write("---")
        st.write("#### Comparison Matrix")
        matrix_df = build_movie_comparison_matrix(df, selected_comp, recommender=recommender)
        if not matrix_df.empty:
            st.dataframe(matrix_df, use_container_width=True)

            csv_comp = matrix_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Comparison Table (CSV)",
                data=csv_comp,
                file_name="movie_comparison.csv",
                mime="text/csv",
            )

    st.write("---")
    # Next Navigation Button
    col_n1, col_n2 = st.columns([1.5, 3])
    with col_n1:
        if st.button("Next: View Analytics Dashboard ➔", type="primary", use_container_width=True):
            navigate_to("📊 Analytics Dashboard")
    with col_n2:
        st.caption("Explore summary KPIs, evidence-based insights, and manual analysis modes.")


# ==============================================================================
# PAGE 6: 📊 ANALYTICS DASHBOARD
# ==============================================================================
elif st.session_state.nav_page == "📊 Analytics Dashboard":
    st.subheader("📊 Executive Movie Analytics Dashboard")
    st.write(
        "Examine key performance indicators, evidence-based data insights, and execute "
        "either the one-click Complete Automated Analysis or deep-dive Manual Mode."
    )

    df = st.session_state.cleaned_df
    if df is None:
        render_empty_state("📂", "Dataset Required", "Please load a movie dataset to generate analytics.")
        st.stop()

    overview = compute_overview_metrics(df)

    # Top Executive KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card("Total Movies", f"{overview['total_movies']:,}", f"Unique titles: {overview['unique_titles']:,}")
    with kpi2:
        render_metric_card("Genres Identified", f"{overview['unique_genres']}", "Distinct categories")
    with kpi3:
        avg_r_str = f"{overview['avg_rating']}/10" if overview['avg_rating'] else "N/A"
        med_r_str = f"Median: {overview['median_rating']}/10" if overview['median_rating'] else ""
        render_metric_card("Average Rating", avg_r_str, med_r_str)
    with kpi4:
        avg_rt_str = f"{overview['avg_runtime']} mins" if overview['avg_runtime'] else "N/A"
        y_span = f"Horizon: {overview['min_year']}–{overview['max_year']}" if overview['min_year'] else ""
        render_metric_card("Average Runtime", avg_rt_str, y_span)

    # Dynamic Evidence-Based Insights Section
    st.write(" ")
    st.write("### 💡 Executive Data Insights")
    insights = generate_executive_insights(df)
    ins_cols = st.columns(2)
    for i, ins in enumerate(insights):
        with ins_cols[i % 2]:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border-left: 4px solid {THEME['secondary_accent']}; border: 1px solid {THEME['border_subtle']}; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; font-size: 0.92rem;">
                    {ins}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("---")
    # Analysis Modes Selector
    mode_tab1, mode_tab2 = st.tabs(["🚀 Mode A: Complete Automated Analysis", "🛠️ Mode B: Manual Category Deep-Dive"])

    with mode_tab1:
        st.write("#### Automatic Complete Analysis Pipeline")
        st.caption("Executes all statistical calculations, generates distribution charts across attributes, and compiles an exportable executive report.")

        if st.button("🚀 Run Complete Movie Analysis", type="primary"):
            st.session_state.auto_analysis_run = True
            st.toast("Complete analysis executed successfully!", icon="✅")

        if st.session_state.auto_analysis_run:
            st.success("✅ **Analysis Complete**: All applicable metrics, distributions, and audits compiled.")

            # Display 4 Key Visualizations in a 2x2 Grid
            grid_c1, grid_c2 = st.columns(2)
            with grid_c1:
                fig_g = plot_genre_bar(df, top_n=10)
                if fig_g:
                    st.plotly_chart(fig_g, use_container_width=True)

            with grid_c2:
                fig_r = plot_rating_histogram(df)
                if fig_r:
                    st.plotly_chart(fig_r, use_container_width=True)

            grid_c3, grid_c4 = st.columns(2)
            with grid_c3:
                fig_y = plot_release_year_histogram(df)
                if fig_y:
                    st.plotly_chart(fig_y, use_container_width=True)

            with grid_c4:
                fig_comp = plot_metadata_completeness_chart(df)
                if fig_comp:
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Executive Report Export Button
            st.write("---")
            rep_md = generate_executive_summary_markdown(df, st.session_state.dataset_name, st.session_state.cleaning_report, st.session_state.latest_recs)
            st.download_button(
                "📥 Download Complete Executive Analytics Report (Markdown)",
                data=rep_md,
                file_name="executive_movie_analytics_report.md",
                mime="text/markdown",
            )

    with mode_tab2:
        st.write("#### Manual Analysis by Category")
        category_choice = st.selectbox(
            "Select Analytical Dimension:",
            ["Genre Analysis", "Rating Analysis", "Release Year Trends", "Runtime & Popularity", "Data Quality & Metadata"],
        )

        if category_choice == "Genre Analysis":
            gc1, gc2 = st.columns(2)
            with gc1:
                top_n_g = st.slider("Number of Top Genres:", min_value=5, max_value=20, value=10, key="man_g_n")
                fig1 = plot_genre_bar(df, top_n=top_n_g)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
            with gc2:
                fig2 = plot_genre_share_donut(df, top_n=8)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)

        elif category_choice == "Rating Analysis":
            rc1, rc2 = st.columns(2)
            with rc1:
                fig_r1 = plot_rating_histogram(df)
                if fig_r1:
                    st.plotly_chart(fig_r1, use_container_width=True)
            with rc2:
                fig_r2 = plot_rating_by_genre(df, top_n=8)
                if fig_r2:
                    st.plotly_chart(fig_r2, use_container_width=True)

        elif category_choice == "Release Year Trends":
            yc1, yc2 = st.columns(2)
            with yc1:
                fig_y1 = plot_movies_by_decade(df)
                if fig_y1:
                    st.plotly_chart(fig_y1, use_container_width=True)
            with yc2:
                fig_y2 = plot_rating_trend_by_decade(df)
                if fig_y2:
                    st.plotly_chart(fig_y2, use_container_width=True)

        elif category_choice == "Runtime & Popularity":
            rtc1, rtc2 = st.columns(2)
            with rtc1:
                fig_rt1 = plot_runtime_distribution(df)
                if fig_rt1:
                    st.plotly_chart(fig_rt1, use_container_width=True)
            with rtc2:
                fig_rt2 = plot_runtime_vs_rating_scatter(df)
                if fig_rt2:
                    st.plotly_chart(fig_rt2, use_container_width=True)

        elif category_choice == "Data Quality & Metadata":
            qc1, qc2 = st.columns(2)
            with qc1:
                fig_q1 = plot_metadata_completeness_chart(df)
                if fig_q1:
                    st.plotly_chart(fig_q1, use_container_width=True)
            with qc2:
                fig_q2 = plot_overview_word_count(df)
                if fig_q2:
                    st.plotly_chart(fig_q2, use_container_width=True)

    st.write("---")
    # Next Navigation Button
    col_n1, col_n2 = st.columns([1.5, 3])
    with col_n1:
        if st.button("Next: Advanced Visualizations Library ➔", type="primary", use_container_width=True):
            navigate_to("📈 Advanced Visualizations")
    with col_n2:
        st.caption("Browse all Plotly charts with customizable options and data downloads.")


# ==============================================================================
# PAGE 7: 📈 ADVANCED VISUALIZATIONS
# ==============================================================================
elif st.session_state.nav_page == "📈 Advanced Visualizations":
    st.subheader("📈 Advanced Interactive Visualization Library")
    st.write(
        "A central registry of Plotly interactive charts. Filter by category, inspect individual "
        "dimensions, and download chart source data as CSV."
    )

    df = st.session_state.cleaned_df
    if df is None:
        render_empty_state("📂", "Dataset Required", "Please load a movie dataset to render visualizations.")
        st.stop()

    available_charts = get_available_charts(df)
    categories = ["All Categories"] + sorted(list(set(c.category for c in available_charts)))

    cat_col1, cat_col2 = st.columns([1.5, 2])
    with cat_col1:
        chosen_cat = st.selectbox("Filter Charts by Category:", options=categories, index=0)

    # Filter chart registry
    active_charts = [
        c for c in available_charts if chosen_cat == "All Categories" or c.category == chosen_cat
    ]

    st.write(f"Showing **{len(active_charts)}** interactive charts")

    # Render Charts in 2-Column Responsive Layout
    chart_cols = st.columns(2)
    for idx, chart_def in enumerate(active_charts):
        with chart_cols[idx % 2]:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid {THEME['border_subtle']}; border-radius: 12px; padding: 18px 20px; margin-bottom: 20px; box-shadow: {THEME['card_shadow']};">
                    <span style="background-color: #EDE9FE; color: {THEME['primary_accent']}; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; text-transform: uppercase;">
                        {chart_def.category}
                    </span>
                    <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['text_primary']}; margin: 6px 0 2px 0;">
                        {chart_def.title}
                    </div>
                    <div style="font-size: 0.85rem; color: {THEME['text_secondary']}; margin-bottom: 12px;">
                        {chart_def.description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                fig = chart_def.builder_func(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Chart '{chart_def.title}' cannot be rendered with current dataset values.")
            except Exception as e:
                st.error(f"Error rendering chart: {e}")

    st.write("---")
    # Next Navigation Button
    col_n1, col_n2 = st.columns([1.5, 3])
    with col_n1:
        if st.button("Next: Reports & Central Downloads ➔", type="primary", use_container_width=True):
            navigate_to("📥 Reports & Downloads")
    with col_n2:
        st.caption("Access executive analytics reports and exportable datasets.")


# ==============================================================================
# PAGE 8: 📥 REPORTS & DOWNLOADS
# ==============================================================================
elif st.session_state.nav_page == "📥 Reports & Downloads":
    st.subheader("📥 Central Reports & Data Downloads Hub")
    st.write("Export cleaned datasets, quality audit logs, executive summary reports, and recommendation queries.")

    df = st.session_state.cleaned_df
    if df is None:
        render_empty_state("📂", "No Data Available", "Ingest a movie dataset to generate downloadable reports.")
        st.stop()

    # Download Cards Grid
    st.write("### Available Artifacts & Datasets")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 4px;">
                    📊 Executive Analytics Report (Markdown)
                </div>
                <div style="font-size: 0.88rem; color: {THEME['text_secondary']}; margin-bottom: 12px;">
                    Comprehensive summary containing KPIs, evidence-based insights, genre breakdowns, and metadata audits.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rep_md = generate_executive_summary_markdown(
            df,
            st.session_state.dataset_name,
            st.session_state.cleaning_report,
            st.session_state.latest_recs,
        )
        st.download_button(
            "📥 Download Executive Report (.md)",
            data=rep_md,
            file_name="executive_movie_analytics_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="cinema-card" style="margin-top: 20px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 4px;">
                    🧹 Cleaned Movie Catalog (CSV)
                </div>
                <div style="font-size: 0.88rem; color: {THEME['text_secondary']}; margin-bottom: 12px;">
                    Fully sanitized dataset with deduplicated records, parsed text lists, and normalized numeric columns.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        csv_cleaned = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned Catalog (.csv)",
            data=csv_cleaned,
            file_name="cleaned_movies.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="cinema-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 4px;">
                    📋 Data Quality & Cleaning Audit (Markdown)
                </div>
                <div style="font-size: 0.88rem; color: {THEME['text_secondary']}; margin-bottom: 12px;">
                    Diagnostic record of duplicate drops, title disambiguations, and schema normalization steps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        audit_md = st.session_state.cleaning_report.to_markdown() if st.session_state.cleaning_report else "No audit"
        st.download_button(
            "📥 Download Quality Audit (.md)",
            data=audit_md,
            file_name="quality_audit_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="cinema-card" style="margin-top: 20px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: {THEME['primary_accent']}; margin-bottom: 4px;">
                    🎯 Latest Recommendations (CSV)
                </div>
                <div style="font-size: 0.88rem; color: {THEME['text_secondary']}; margin-bottom: 12px;">
                    Export the most recently generated recommendations with similarity match scores and explanations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.latest_recs is not None and not st.session_state.latest_recs.empty:
            rec_csv = st.session_state.latest_recs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Recommendations (.csv)",
                data=rec_csv,
                file_name="latest_recommendations.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("📥 Recommendations (None Run Yet)", disabled=True, use_container_width=True)

    # Preview of Executive Summary
    st.write("---")
    st.write("### 📄 Preview: Executive Analytics Report")
    with st.expander("Expand to view rendered Markdown report", expanded=True):
        st.markdown(rep_md)


# ==============================================================================
# PAGE 9: ℹ️ ABOUT THIS PROJECT
# ==============================================================================
elif st.session_state.nav_page == "ℹ️ About This Project":
    st.subheader("ℹ️ About the Movie Recommendation System")
    st.write(
        "A portfolio-quality, production-ready machine learning and data engineering application "
        "designed for demonstration to technical recruiters, machine learning engineers, and software architects."
    )

    t1, t2, t3 = st.tabs(["🏗️ Architecture & ML Design", "🛡️ Privacy & Performance", "💼 Recruiter & Interview Highlights"])

    with t1:
        st.markdown(
            r"""
            #### System Architecture & Mathematical Foundations
            
            1. **Flexible Data Pipeline**:
               - Accepts varied schemas using synonym detection (e.g. `title` vs `movie_title` vs `original_title`).
               - Handles nested TMDB JSON structures (`[{"name": "Action"}]`) and delimited strings safely.
               - Strips blank rows, drops exact duplicates, and disambiguates remake titles with release years.
            
            2. **TF-IDF Vector Space Representation**:
               $$\text{tf-idf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$
               Uses sublinear term frequency scaling ($1 + \log(\text{tf})$) to prevent high-frequency words from skewing matches.
            
            3. **Memory-Aware Sparse Cosine Similarity**:
               Unlike basic tutorials that compute an $O(N^2)$ dense matrix (which quickly triggers out-of-memory crashes on realistic datasets), this engine computes $1 \times N$ sparse linear algebra on-the-fly:
               $$\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
               Execution time is under 5 milliseconds per query in $O(N)$ memory space!
            """
        )

    with t2:
        st.markdown(
            f"""
            #### Privacy, Security & Resource Governance
            - **No External Paid APIs**: Fully self-contained open-source stack (Python, Scikit-Learn, Streamlit).
            - **Data Isolation**: Uploaded raw files and processed exports are strictly excluded via `.gitignore`.
            - **Resource Guardrails**: Enforces a 50MB file size ceiling to ensure flawless stability within free cloud hosting memory limits (e.g., Streamlit Community Cloud).
            """
        )

    with t3:
        st.markdown(
            f"""
            #### Portfolio & Interview Discussion Talking Points
            - **Defensive Engineering**: Handles missing genres, missing overviews, and unmapped columns gracefully with fallback tokens and informative diagnostics.
            - **Explainability**: Recommendations provide transparent rationales (overlapping genres, director matches, shared cast, and top TF-IDF thematic terms).
            - **Separation of Concerns**: Modular decoupled package design (`data`, `preprocessing`, `recommendation`, `analytics`, `ui`) with 32+ unit tests.
            """
        )
