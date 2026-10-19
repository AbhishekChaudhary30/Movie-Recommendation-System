"""Movie Recommendation System — Streamlit Application Entrypoint."""

import io
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
from src.analytics.metrics import (
    compute_overview_metrics,
    get_unique_genres_list,
    plot_genre_distribution,
    plot_rating_distribution,
    plot_recommendation_similarity_scores,
    plot_release_year_distribution,
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
from src.utils.exceptions import DataLoadError, MovieRecommendationError

# Page Configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | Discovery & Analytics",
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
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Editorial Header Box */
    .cinema-header {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%);
        border: 1px solid {THEME['border_subtle']};
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: {THEME['card_shadow']};
    }}
    .cinema-title {{
        color: {THEME['primary_accent']};
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }}
    .cinema-subtitle {{
        color: {THEME['text_secondary']};
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 8px;
    }}
    .cinema-desc {{
        color: {THEME['text_primary']};
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 0px;
    }}

    /* Metric Cards */
    .metric-card {{
        background-color: {THEME['bg_surface']};
        border: 1px solid {THEME['border_subtle']};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: {THEME['card_shadow']};
        text-align: left;
    }}
    .metric-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {THEME['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {THEME['primary_accent']};
    }}
    .metric-sub {{
        font-size: 0.8rem;
        color: {THEME['text_secondary']};
        margin-top: 2px;
    }}

    /* Workflow Stepper */
    .step-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 8px;
    }}
    .step-active {{
        background-color: {THEME['primary_accent']};
        color: white;
    }}
    .step-done {{
        background-color: {THEME['secondary_accent']};
        color: white;
    }}
    .step-pending {{
        background-color: #E2E8F0;
        color: {THEME['text_secondary']};
    }}

    /* Movie Recommendation Card */
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
    """Initialize persistent workflow state."""
    if "workflow_stage" not in st.session_state:
        st.session_state.workflow_stage = 1  # 1: Upload, 2: Validate, 3: Clean, 4: Recommend
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
    if "feature_weights" not in st.session_state:
        st.session_state.feature_weights = DEFAULT_FEATURE_WEIGHTS.copy()


init_session_state()


def reset_workflow():
    """Reset the application workflow state for a new dataset."""
    st.session_state.workflow_stage = 1
    st.session_state.raw_df = None
    st.session_state.dataset_name = ""
    st.session_state.user_mappings = {}
    st.session_state.schema_mapping = None
    st.session_state.validation_report = None
    st.session_state.cleaned_df = None
    st.session_state.cleaning_report = None
    st.session_state.recommender = None
    st.session_state.selected_movie = None
    st.session_state.feature_weights = DEFAULT_FEATURE_WEIGHTS.copy()
    st.rerun()


# Render Application Header
st.markdown(
    f"""
    <div class="cinema-header">
        <div class="cinema-title">🎬 {APP_TITLE}</div>
        <div class="cinema-subtitle">{APP_SUBTITLE} (v{APP_VERSION})</div>
        <div class="cinema-desc">{APP_DESCRIPTION}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Workflow Stage Progress Bar
stages_info = [
    (1, "1. Dataset Upload"),
    (2, "2. Schema & Validation"),
    (3, "3. Cleaning & Quality Audit"),
    (4, "4. Recommendation Engine Ready"),
]
curr_stage = st.session_state.workflow_stage

stepper_cols = st.columns(4)
for i, (stage_num, stage_label) in enumerate(stages_info):
    with stepper_cols[i]:
        if curr_stage > stage_num:
            st.markdown(
                f'<span class="step-badge step-done">✓ Step {stage_num}</span><br><b>{stage_label}</b>',
                unsafe_allow_html=True,
            )
        elif curr_stage == stage_num:
            st.markdown(
                f'<span class="step-badge step-active">● Active</span><br><b>{stage_label}</b>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="step-badge step-pending">○ Step {stage_num}</span><br><span style="color:#A0AEC0">{stage_label}</span>',
                unsafe_allow_html=True,
            )
st.write("---")

# Sidebar Controls
with st.sidebar:
    st.subheader("⚙️ Session & Navigation")
    st.write(f"**Current Stage:** Step {curr_stage} of 4")

    if st.session_state.raw_df is not None:
        st.success(f"📂 Dataset: **{st.session_state.dataset_name}**")
        st.caption(f"Raw shape: {st.session_state.raw_df.shape[0]} rows × {st.session_state.raw_df.shape[1]} cols")

    if st.button("🔄 Reset / Load New Dataset", use_container_width=True):
        reset_workflow()

    st.write("---")
    st.subheader("💡 Methodology Note")
    st.caption(
        "This system uses **genuine content-based filtering** powered by TF-IDF metadata vectorization "
        "and sparse Cosine Similarity. Similarity scores reflect attribute alignment (genres, cast, keywords, "
        "overview), not user rating predictions."
    )


# ==============================================================================
# STAGE 1: DATASET UPLOAD / SAMPLE SELECTION
# ==============================================================================
if curr_stage == 1:
    st.subheader("Step 1: Choose or Upload a Movie Dataset")
    st.write(
        "Select the built-in curated 50-movie cinema gem dataset or upload your own CSV file "
        "(such as TMDB 5000, MovieLens, or custom metadata)."
    )

    source_tab1, source_tab2 = st.tabs(["🌟 Curated Sample Dataset (50 Classic & Modern Movies)", "📁 Upload Custom CSV File"])

    with source_tab1:
        st.info("The sample dataset contains 50 diverse titles across genres, directors, and eras with complete attributes.")
        if st.button("🚀 Load Sample Dataset", type="primary", use_container_width=False):
            sample_path = SAMPLE_DATA_DIR / "sample_movies.csv"
            try:
                df = load_csv(sample_path)
                st.session_state.raw_df = df
                st.session_state.dataset_name = "sample_movies.csv"
                st.session_state.user_mappings = {}
                # Detect initial schema
                st.session_state.schema_mapping = detect_schema(list(df.columns))
                st.success("Loaded curated sample dataset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load sample dataset: {e}")

    with source_tab2:
        uploaded_file = st.file_uploader(
            "Upload a Movie CSV Dataset (Max 50MB)",
            type=["csv"],
            help="Compatible with standard movie datasets having title, overview, genres, cast, etc.",
        )
        if uploaded_file is not None:
            if st.button("📥 Process Uploaded File", type="primary"):
                try:
                    with st.spinner("Reading and parsing CSV file..."):
                        df = load_csv(uploaded_file)
                        st.session_state.raw_df = df
                        st.session_state.dataset_name = uploaded_file.name
                        st.session_state.user_mappings = {}
                        st.session_state.schema_mapping = detect_schema(list(df.columns))
                        st.success(f"Uploaded '{uploaded_file.name}' successfully! ({len(df)} rows, {len(df.columns)} columns)")
                        st.rerun()
                except DataLoadError as e:
                    st.error(f"Data Load Error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error while loading CSV: {e}")

    # Preview Loaded Dataset
    if st.session_state.raw_df is not None:
        st.write("---")
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.raw_df.head(8), use_container_width=True)

        # NEXT BUTTON (Stage 1 -> Stage 2)
        col_next, col_info = st.columns([1, 3])
        with col_next:
            if st.button("Next: Validate Dataset & Schema ➔", type="primary", use_container_width=True):
                st.session_state.workflow_stage = 2
                st.rerun()
        with col_info:
            st.caption("Proceed to schema detection, column mapping confirmation, and data validation.")


# ==============================================================================
# STAGE 2: SCHEMA DETECTION & VALIDATION
# ==============================================================================
elif curr_stage == 2:
    st.subheader("Step 2: Schema Mapping & Dataset Validation")
    st.write(
        "Verify how dataset columns map to canonical movie attributes. "
        "You can customize mappings if any field was ambiguously identified."
    )

    df = st.session_state.raw_df
    if df is None:
        st.warning("No dataset loaded. Please return to Step 1.")
        if st.button("⬅ Back to Upload"):
            st.session_state.workflow_stage = 1
            st.rerun()
        st.stop()

    cols = list(df.columns)
    detected_mapping = detect_schema(cols, user_overrides=st.session_state.user_mappings)

    # Column Mapping Configuration Expander
    with st.expander("🛠️ Customize / Review Column Mappings", expanded=not detected_mapping.has_title):
        st.caption("Adjust which column from your CSV represents each movie attribute.")
        map_cols = st.columns(3)

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
            col_target = map_cols[idx % 3]
            current_mapped = detected_mapping.get_original(canon_key)
            default_idx = options.index(current_mapped) if current_mapped in options else 0

            selected = col_target.selectbox(
                label,
                options=options,
                index=default_idx,
                key=f"map_{canon_key}",
            )
            if selected != "<Unmapped>":
                new_overrides[canon_key] = selected

        if st.button("Apply Mapping Adjustments"):
            st.session_state.user_mappings = new_overrides
            st.rerun()

    # Re-run detection with applied overrides
    final_mapping = detect_schema(cols, user_overrides=st.session_state.user_mappings)
    st.session_state.schema_mapping = final_mapping

    # Perform Validation
    report = validate_dataset(df, final_mapping)
    st.session_state.validation_report = report

    # Render Validation Status
    if report.is_valid:
        st.success("✅ **Dataset Validation Passed**: All essential structural requirements are satisfied.")
    else:
        st.error("❌ **Validation Failed**: Critical issues prevent recommendation generation.")
        for err in report.critical_errors:
            st.markdown(f"- ⛔ {err}")

    if report.warnings:
        with st.expander(f"⚠️ Validation Warnings & Diagnostics ({len(report.warnings)})", expanded=True):
            for w in report.warnings:
                st.markdown(f"- ⚠️ {w}")

    # Summary of Identified Attributes
    st.write("##### Feature Coverage Summary:")
    feat_cols = st.columns(5)
    key_features = [
        ("title", "Title", final_mapping.has_title),
        ("genres", "Genres", "genres" in final_mapping.available_features),
        ("overview", "Overview", "overview" in final_mapping.available_features),
        ("cast", "Cast", "cast" in final_mapping.available_features),
        ("director", "Director", "director" in final_mapping.available_features),
    ]
    for i, (k, name, present) in enumerate(key_features):
        with feat_cols[i]:
            if present:
                st.markdown(f"**{name}**: :green[Detected ✓]")
                st.caption(f"Col: `{final_mapping.get_original(k)}`")
            else:
                st.markdown(f"**{name}**: :red[Not Found]")
                st.caption("Optional" if k != "title" else "Required")

    st.write("---")
    # Navigation: Back & Next Button
    col_back, col_next, col_fill = st.columns([1, 1.5, 3])
    with col_back:
        if st.button("⬅ Back to Upload", use_container_width=True):
            st.session_state.workflow_stage = 1
            st.rerun()

    with col_next:
        if report.is_valid:
            if st.button("Next: Clean & Audit Data ➔", type="primary", use_container_width=True):
                st.session_state.workflow_stage = 3
                st.rerun()
        else:
            st.button(
                "Next: Blocked (Resolve Critical Errors)",
                disabled=True,
                help="Cannot proceed until title column is assigned and dataset is valid.",
                use_container_width=True,
            )


# ==============================================================================
# STAGE 3: AUTOMATIC CLEANING & QUALITY REPORT
# ==============================================================================
elif curr_stage == 3:
    st.subheader("Step 3: Data Cleaning & Quality Audit Report")
    st.write(
        "The automated pipeline normalizes metadata, removes exact duplicates, "
        "disambiguates duplicate titles, and parses nested structures (e.g., JSON/pipe lists)."
    )

    df = st.session_state.raw_df
    mapping = st.session_state.schema_mapping

    if df is None or mapping is None:
        st.warning("Please complete Steps 1 and 2 first.")
        st.session_state.workflow_stage = 1
        st.rerun()

    # Perform Cleaning if not already in session state
    if st.session_state.cleaned_df is None or st.session_state.cleaning_report is None:
        with st.spinner("Executing data cleaning and quality audit..."):
            cleaned_df, report = clean_dataset(df, mapping)
            st.session_state.cleaned_df = cleaned_df
            st.session_state.cleaning_report = report
    else:
        cleaned_df = st.session_state.cleaned_df
        report = st.session_state.cleaning_report

    # Top-Level Cleaning Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Cleaned Movies</div>
                <div class="metric-value">{report.cleaned_rows}</div>
                <div class="metric-sub">From {report.original_rows} raw records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Exact Duplicates Dropped</div>
                <div class="metric-value">{report.exact_duplicates_removed}</div>
                <div class="metric-sub">Redundant rows pruned</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Titles Disambiguated</div>
                <div class="metric-value">{report.potential_duplicate_titles_count}</div>
                <div class="metric-sub">Year/ID appended for uniqueness</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Rec Features Ready</div>
                <div class="metric-value">{len(report.available_recommendation_features)}</div>
                <div class="metric-sub">{', '.join(report.available_recommendation_features) or 'None'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write(" ")
    # Quality Report Details & Audit Trail
    audit_tab1, audit_tab2, audit_tab3 = st.tabs(["📋 Audit Trail", "🔍 Cleaned Preview", "📥 Export Cleaned Data"])

    with audit_tab1:
        st.markdown(report.to_markdown())

    with audit_tab2:
        preview_cols = [
            col for col in [
                "display_title", "cleaned_genres", "cleaned_release_year",
                "cleaned_rating", "cleaned_director", "cleaned_cast"
            ] if col in cleaned_df.columns
        ]
        st.dataframe(cleaned_df[preview_cols].head(10), use_container_width=True)

    with audit_tab3:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_movies.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                label="📥 Download Quality Report (Markdown)",
                data=report.to_markdown(),
                file_name="quality_audit_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    st.write("---")
    # Navigation: Back & Next Button
    col_back, col_next, col_fill = st.columns([1, 1.8, 3])
    with col_back:
        if st.button("⬅ Back to Validation", use_container_width=True):
            st.session_state.workflow_stage = 2
            st.rerun()

    with col_next:
        if st.button("Next: Initialize Recommendation Engine ➔", type="primary", use_container_width=True):
            st.session_state.workflow_stage = 4
            st.rerun()


# ==============================================================================
# STAGE 4: RECOMMENDATION ENGINE & ANALYTICS
# ==============================================================================
elif curr_stage == 4:
    st.subheader("Step 4: Interactive Recommendation Engine & Analytics")
    st.write("Discover similar movies, explore transparent similarity scoring, and examine dataset analytics.")

    cleaned_df = st.session_state.cleaned_df
    schema_mapping = st.session_state.schema_mapping

    if cleaned_df is None or schema_mapping is None:
        st.warning("Please complete Steps 1-3 first.")
        st.session_state.workflow_stage = 1
        st.rerun()

    # Train / Fit Recommender if needed
    if st.session_state.recommender is None:
        with st.spinner("Vectorizing metadata and initializing Content-Based Engine..."):
            recommender = ContentBasedRecommender(feature_weights=st.session_state.feature_weights)
            recommender.fit(cleaned_df, schema_mapping)
            st.session_state.recommender = recommender

    recommender: ContentBasedRecommender = st.session_state.recommender

    # Main Recommendation UI Layout
    rec_tab1, rec_tab2 = st.tabs(["🎯 Movie Recommendations & Explanations", "📊 Dataset Analytics"])

    with rec_tab1:
        # Selection & Controls Section
        ctrl_col1, ctrl_col2 = st.columns([1.5, 1])

        titles_list = sorted(cleaned_df["display_title"].tolist())
        default_movie_idx = 0
        if st.session_state.selected_movie in titles_list:
            default_movie_idx = titles_list.index(st.session_state.selected_movie)

        with ctrl_col1:
            selected_title = st.selectbox(
                "🎬 Select a Movie as Seed:",
                options=titles_list,
                index=default_movie_idx,
                help="Search and pick a movie to generate content-based recommendations.",
            )
            st.session_state.selected_movie = selected_title

        with ctrl_col2:
            top_n = st.slider("Number of Recommendations (N):", min_value=3, max_value=20, value=6, step=1)

        # Filters Expander
        with st.expander("🔍 Filter Controls & Feature Weights", expanded=False):
            fcol1, fcol2, fcol3 = st.columns(3)

            # Genre Filter
            available_genres = ["All"] + get_unique_genres_list(cleaned_df)
            with fcol1:
                genre_filter = st.selectbox("Genre Filter:", options=available_genres, index=0)

            # Rating Filter
            has_ratings = "cleaned_rating" in cleaned_df.columns and cleaned_df["cleaned_rating"].notna().sum() > 0
            with fcol2:
                if has_ratings:
                    min_rating = st.slider(
                        "Minimum Rating Filter:",
                        min_value=0.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.5,
                        help="Only recommend movies at or above this rating score.",
                    )
                else:
                    st.caption("Rating filter disabled (no rating column detected).")
                    min_rating = 0.0

            # Year Filter
            has_years = "cleaned_release_year" in cleaned_df.columns and cleaned_df["cleaned_release_year"].notna().sum() > 0
            with fcol3:
                if has_years:
                    valid_years = cleaned_df["cleaned_release_year"].dropna().astype(int)
                    min_y = int(valid_years.min())
                    max_y = int(valid_years.max())
                    if min_y < max_y:
                        year_range = st.slider("Release Year Range:", min_value=min_y, max_value=max_y, value=(min_y, max_y))
                    else:
                        year_range = (min_y, max_y)
                else:
                    st.caption("Year filter disabled (no release date column detected).")
                    year_range = (None, None)

            # Feature Weight Customization
            st.write("###### Feature Weights (Relative Importance):")
            wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
            w_genres = wcol1.number_input("Genres Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("genres", 2.0), step=0.5)
            w_kw = wcol2.number_input("Keywords Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("keywords", 1.5), step=0.5)
            w_cast = wcol3.number_input("Cast Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("cast", 1.2), step=0.5)
            w_dir = wcol4.number_input("Director Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("director", 1.0), step=0.5)
            w_ov = wcol5.number_input("Overview Weight", min_value=0.5, max_value=5.0, value=st.session_state.feature_weights.get("overview", 1.0), step=0.5)

            if st.button("Apply New Feature Weights"):
                new_weights = {
                    "genres": w_genres,
                    "keywords": w_kw,
                    "cast": w_cast,
                    "director": w_dir,
                    "overview": w_ov,
                }
                st.session_state.feature_weights = new_weights
                # Re-fit recommender with updated weights
                recommender.fit(cleaned_df, schema_mapping, feature_weights=new_weights)
                st.session_state.recommender = recommender
                st.success("Feature weights updated and recommender re-vectorized!")
                st.rerun()

        # Seed Movie Highlight Box
        seed_info = recommender.get_movie_details(selected_title)
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid {THEME['border_subtle']}; border-radius: 12px; padding: 18px 22px; margin: 16px 0; box-shadow: {THEME['card_shadow']};">
                <span style="font-size: 0.8rem; font-weight: 700; color: {THEME['primary_accent']}; text-transform: uppercase;">SEED MOVIE</span>
                <div style="font-size: 1.4rem; font-weight: 800; color: {THEME['text_primary']}; margin: 4px 0;">{seed_info.get('display_title')}</div>
                <div style="margin-bottom: 8px;">
                    <span class="meta-tag">📅 {seed_info.get('cleaned_release_year') or 'N/A'}</span>
                    <span class="meta-tag">⭐ {seed_info.get('cleaned_rating') or 'N/A'}/10</span>
                    <span class="meta-tag">⏱️ {seed_info.get('cleaned_runtime') or 'N/A'} min</span>
                    <span class="meta-tag">🎭 {seed_info.get('cleaned_genres') or 'N/A'}</span>
                    <span class="meta-tag">🎬 {seed_info.get('cleaned_director') or 'N/A'}</span>
                </div>
                <div style="font-size: 0.92rem; color: {THEME['text_secondary']}; line-height: 1.45;">
                    <b>Overview:</b> {seed_info.get('cleaned_overview') or 'No description available.'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Generate Recommendations
        try:
            year_min_val, year_max_val = year_range if year_range != (None, None) else (None, None)
            recs_df = recommender.recommend(
                query=selected_title,
                top_n=top_n,
                min_rating=min_rating if min_rating > 0 else None,
                genre_filter=genre_filter if genre_filter != "All" else None,
                year_min=year_min_val,
                year_max=year_max_val,
            )

            if recs_df.empty:
                st.info("ℹ️ No movies matched the chosen filter combination. Try relaxing your genre, rating, or year filters.")
            else:
                st.write(f"### Top {len(recs_df)} Recommendations for *{selected_title}*")

                # Recommendation Cards
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
                            <div style="font-size: 0.9rem; color: {THEME['text_secondary']}; margin-bottom: 6px;">
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

                # Visual Match Chart
                st.write("---")
                st.write("#### Similarity Match Strength Comparison")
                fig_match = plot_recommendation_similarity_scores(recs_df)
                if fig_match:
                    st.plotly_chart(fig_match, use_container_width=True)

                # Comparison Table
                with st.expander("📊 Detailed Comparison Table"):
                    comp_cols = [
                        "display_title", "similarity_score", "similarity_pct",
                        "cleaned_release_year", "cleaned_rating", "cleaned_genres",
                        "cleaned_director", "explanation"
                    ]
                    st.dataframe(recs_df[comp_cols], use_container_width=True)

        except MovieRecommendationError as e:
            st.error(f"Recommendation Engine Notice: {e}")
        except Exception as e:
            st.error(f"Unexpected recommendation error: {e}")

    # Dataset Analytics Tab
    with rec_tab2:
        st.write("### Dataset Analytics & Distributions")
        overview = compute_overview_metrics(cleaned_df)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Movies", overview["total_movies"])
        a2.metric("Unique Genres", overview["unique_genres"])
        a3.metric("Average Rating", f"{overview['avg_rating']}/10" if overview['avg_rating'] else "N/A")
        a4.metric("Average Runtime", f"{overview['avg_runtime']} min" if overview['avg_runtime'] else "N/A")

        st.write("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_g = plot_genre_distribution(cleaned_df)
            if fig_g:
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.info("Genre distribution chart unavailable (no genres column).")

        with chart_col2:
            fig_y = plot_release_year_distribution(cleaned_df)
            if fig_y:
                st.plotly_chart(fig_y, use_container_width=True)
            else:
                st.info("Release year distribution chart unavailable (no date column).")

        st.write("---")
        fig_r = plot_rating_distribution(cleaned_df)
        if fig_r:
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Rating distribution chart unavailable (no ratings column).")
