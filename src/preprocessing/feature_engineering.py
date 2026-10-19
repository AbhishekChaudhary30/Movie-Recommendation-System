"""Feature engineering: metadata soup construction and TF-IDF vectorization."""


import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DEFAULT_FEATURE_WEIGHTS, MAX_TFIDF_FEATURES
from src.preprocessing.text_processing import (
    clean_text,
    process_entities_string,
    process_genres_string,
)
from src.utils.exceptions import InsufficientFeaturesError
from src.utils.logging_config import get_logger

logger = get_logger("feature_engineering")


def build_movie_soup(
    df: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Construct a weighted content representation ('soup') for each movie in the dataframe.
    Higher weighted features are proportionally repeated to influence TF-IDF term frequencies.
    """
    weights = feature_weights if feature_weights is not None else DEFAULT_FEATURE_WEIGHTS
    soups: list[str] = []

    for _, row in df.iterrows():
        parts: list[str] = []

        # 1. Genres
        genres_weight = max(1, int(round(weights.get("genres", 2.0))))
        raw_genres = row.get("cleaned_genres", "")
        clean_g = process_genres_string(raw_genres)
        if clean_g:
            parts.extend([clean_g] * genres_weight)

        # 2. Keywords
        kw_weight = max(1, int(round(weights.get("keywords", 1.5))))
        raw_kw = row.get("cleaned_keywords", "")
        clean_kw = process_entities_string(raw_kw, max_items=8)
        if clean_kw:
            parts.extend([clean_kw] * kw_weight)

        # 3. Cast
        cast_weight = max(1, int(round(weights.get("cast", 1.2))))
        raw_cast = row.get("cleaned_cast", "")
        clean_cast = process_entities_string(raw_cast, max_items=5)
        if clean_cast:
            parts.extend([clean_cast] * cast_weight)

        # 4. Director
        dir_weight = max(1, int(round(weights.get("director", 1.0))))
        raw_dir = row.get("cleaned_director", "")
        clean_dir = process_entities_string(raw_dir, max_items=2)
        if clean_dir:
            parts.extend([clean_dir] * dir_weight)

        # 5. Overview
        overview_weight = max(1, int(round(weights.get("overview", 1.0))))
        raw_ov = row.get("cleaned_overview", "")
        clean_ov = clean_text(raw_ov)
        if clean_ov:
            parts.extend([clean_ov] * overview_weight)

        # Fallback to title words if all content features are empty
        soup_text = " ".join(parts).strip()
        if not soup_text:
            title_text = clean_text(row.get("raw_title", row.get("display_title", "")))
            soup_text = title_text

        soups.append(soup_text)

    return pd.Series(soups, index=df.index, name="metadata_soup")


def vectorize_features(
    soup_series: pd.Series,
    max_features: int = MAX_TFIDF_FEATURES,
) -> tuple[spmatrix, TfidfVectorizer]:
    """
    Vectorize the text soup using TF-IDF with sublinear term frequency scaling.
    Returns sparse CSR matrix and the fitted TfidfVectorizer.
    """
    # Verify non-empty content
    non_empty = soup_series[soup_series.str.strip() != ""]
    if len(non_empty) == 0:
        raise InsufficientFeaturesError(
            "Cannot vectorize dataset: All movie content features and titles are empty."
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=max_features,
        min_df=1,
    )

    tfidf_matrix = vectorizer.fit_transform(soup_series)
    logger.info(
        f"Vectorized {tfidf_matrix.shape[0]} documents into {tfidf_matrix.shape[1]} TF-IDF features."
    )
    return tfidf_matrix, vectorizer
