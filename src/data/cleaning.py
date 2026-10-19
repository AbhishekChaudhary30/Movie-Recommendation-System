"""Automatic data cleaning and quality reporting pipeline."""

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import SchemaMapping
from src.utils.logging_config import get_logger

logger = get_logger("data_cleaning")


@dataclass
class CleaningReport:
    """Detailed audit report produced by the cleaning pipeline."""
    original_rows: int
    original_cols: int
    cleaned_rows: int
    cleaned_cols: int
    exact_duplicates_removed: int
    potential_duplicate_titles_count: int
    missing_titles_removed: int
    missing_values_before: dict[str, int]
    missing_values_after: dict[str, int]
    anomalies_handled: dict[str, int]
    available_recommendation_features: list[str]
    audit_log: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the cleaning report as a clean Markdown string."""
        md = [
            "### Data Quality & Cleaning Audit Report",
            f"- **Original Shape**: {self.original_rows} rows × {self.original_cols} columns",
            f"- **Cleaned Shape**: {self.cleaned_rows} rows × {self.cleaned_cols} columns",
            f"- **Exact Duplicates Removed**: {self.exact_duplicates_removed}",
            f"- **Missing Titles Excluded**: {self.missing_titles_removed}",
            f"- **Potential Duplicate Titles Identified**: {self.potential_duplicate_titles_count}",
            f"- **Available Recommendation Features**: {', '.join(self.available_recommendation_features) or 'None'}",
            "",
            "#### Audit Trail:",
        ]
        for item in self.audit_log:
            md.append(f"- {item}")
        return "\n".join(md)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "original_rows": self.original_rows,
            "original_cols": self.original_cols,
            "cleaned_rows": self.cleaned_rows,
            "cleaned_cols": self.cleaned_cols,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "potential_duplicate_titles_count": self.potential_duplicate_titles_count,
            "missing_titles_removed": self.missing_titles_removed,
            "missing_values_before": self.missing_values_before,
            "missing_values_after": self.missing_values_after,
            "anomalies_handled": self.anomalies_handled,
            "available_recommendation_features": self.available_recommendation_features,
            "audit_log": self.audit_log,
        }


def parse_json_or_delimited(val: Any, max_items: int | None = 5) -> str:
    """
    Parse complex values such as JSON strings, stringified lists,
    pipe-delimited, or comma-delimited strings into clean, readable text.
    Examples:
      - '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]' -> 'Action, Adventure'
      - "['Christopher Nolan', 'Christian Bale']" -> 'Christopher Nolan, Christian Bale'
      - "Action|Adventure|Sci-Fi" -> 'Action, Adventure, Sci-Fi'
    """
    if pd.isna(val) or val is None:
        return ""

    if isinstance(val, (list, tuple, set)):
        items = [str(x).strip() for x in val if x]
        return ", ".join(items[:max_items]) if max_items else ", ".join(items)

    val_str = str(val).strip()
    if not val_str:
        return ""

    # Check if stringified JSON/Python literal
    if (val_str.startswith("[") and val_str.endswith("]")) or (
        val_str.startswith("{") and val_str.endswith("}")
    ):
        try:
            parsed = json.loads(val_str)
        except Exception:
            try:
                parsed = ast.literal_eval(val_str)
            except Exception:
                parsed = None

        if isinstance(parsed, list):
            names = []
            for item in parsed:
                if isinstance(item, dict):
                    # TMDB commonly stores dictionaries with 'name' key
                    name = item.get("name") or item.get("title") or item.get("character")
                    if name:
                        names.append(str(name).strip())
                elif isinstance(item, str):
                    names.append(item.strip())
            if names:
                return ", ".join(names[:max_items]) if max_items else ", ".join(names)

        elif isinstance(parsed, dict):
            name = parsed.get("name") or parsed.get("title")
            if name:
                return str(name).strip()

    # Handle pipe-delimited strings e.g. "Action|Thriller"
    if "|" in val_str:
        parts = [p.strip() for p in val_str.split("|") if p.strip()]
        return ", ".join(parts[:max_items]) if max_items else ", ".join(parts)

    return val_str


def extract_year_from_value(val: Any) -> int | None:
    """Extract 4-digit release year safely from date strings or numbers."""
    if pd.isna(val) or val is None:
        return None

    # If already numeric
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            year = int(val)
            if 1880 <= year <= 2040:
                return year
            return None
        except (ValueError, OverflowError):
            return None

    val_str = str(val).strip()
    match = re.search(r"\b(18\d\d|19\d\d|20\d\d)\b", val_str)
    if match:
        return int(match.group(1))

    # Try pd.to_datetime as fallback
    try:
        dt = pd.to_datetime(val_str, errors="coerce")
        if pd.notna(dt) and 1880 <= dt.year <= 2040:
            return dt.year
    except Exception:
        pass

    return None


def extract_numeric_value(val: Any) -> float | None:
    """Safely extract float from numeric, string with units (e.g. '120 min', '1,500'), or text."""
    if pd.isna(val) or val is None:
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    val_str = str(val).strip().replace(",", "")
    if not val_str:
        return np.nan
    try:
        return float(val_str)
    except ValueError:
        pass
    match = re.search(r"[-+]?\d*\.?\d+", val_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return np.nan


def clean_dataset(
    df: pd.DataFrame,
    schema_mapping: SchemaMapping,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Execute full data cleaning pipeline:
    1. Drop completely empty rows and columns.
    2. Remove exact duplicate rows.
    3. Exclude records with missing movie titles.
    4. Clean and parse text fields (genres, cast, keywords, overview).
    5. Disambiguate duplicate movie titles with release years.
    6. Sanitize numeric fields (rating, runtime, vote_count).
    7. Generate detailed CleaningReport.
    """
    audit: list[str] = []
    anomalies: dict[str, int] = {}

    orig_rows, orig_cols = df.shape
    missing_before = {col: int(df[col].isna().sum()) for col in df.columns}

    # Work on a copy to preserve raw data immutability
    cleaned = df.copy()

    # 1. Drop completely empty rows & columns
    cleaned.dropna(how="all", inplace=True)
    cleaned.dropna(axis=1, how="all", inplace=True)
    audit.append("Removed entirely empty rows and columns.")

    # 2. Drop exact duplicate rows
    initial_count = len(cleaned)
    cleaned.drop_duplicates(inplace=True)
    exact_dups = initial_count - len(cleaned)
    if exact_dups > 0:
        audit.append(f"Removed {exact_dups} exact duplicate rows.")

    # 3. Handle Title column
    title_orig_col = schema_mapping.get_original("title")
    missing_titles = 0
    if title_orig_col and title_orig_col in cleaned.columns:
        # Strip whitespace from titles
        cleaned[title_orig_col] = cleaned[title_orig_col].astype(str).str.strip()
        # Drop rows with null/empty/'nan' titles
        invalid_titles_mask = (
            cleaned[title_orig_col].isna()
            | (cleaned[title_orig_col] == "")
            | (cleaned[title_orig_col].str.lower() == "nan")
            | (cleaned[title_orig_col].str.lower() == "none")
            | (cleaned[title_orig_col].str.lower() == "null")
        )
        missing_titles = int(invalid_titles_mask.sum())
        if missing_titles > 0:
            cleaned = cleaned[~invalid_titles_mask].copy()
            audit.append(f"Excluded {missing_titles} records with invalid/empty titles.")
    else:
        audit.append("Warning: No title column detected to sanitize.")

    # Reset index
    cleaned.reset_index(drop=True, inplace=True)

    # 4. Standardize / Extract Release Year
    release_date_col = schema_mapping.get_original("release_date")
    if release_date_col and release_date_col in cleaned.columns:
        years = cleaned[release_date_col].apply(extract_year_from_value)
        cleaned["cleaned_release_year"] = years.astype("Int64")
        valid_years = int(years.notna().sum())
        audit.append(f"Extracted release year for {valid_years} records from '{release_date_col}'.")
    else:
        cleaned["cleaned_release_year"] = pd.Series([None] * len(cleaned), dtype="Int64")

    # 5. Clean Recommendation Text Features
    for feature_key in ["genres", "keywords", "cast", "director", "overview"]:
        orig_col = schema_mapping.get_original(feature_key)
        clean_col = f"cleaned_{feature_key}"
        if orig_col and orig_col in cleaned.columns:
            if feature_key in ["genres", "keywords", "cast", "director"]:
                cleaned[clean_col] = cleaned[orig_col].apply(
                    lambda v: parse_json_or_delimited(v, max_items=8 if feature_key != "director" else 2)
                )
            else:  # overview
                cleaned[clean_col] = (
                    cleaned[orig_col]
                    .fillna("")
                    .astype(str)
                    .apply(lambda s: re.sub(r"\s+", " ", s.strip()))
                )
            audit.append(f"Parsed and normalized feature '{feature_key}' from '{orig_col}'.")
        else:
            cleaned[clean_col] = ""

    # 6. Sanitize Numeric Fields (rating, runtime, vote_count)
    rating_col = schema_mapping.get_original("rating")
    if rating_col and rating_col in cleaned.columns:
        ratings = cleaned[rating_col].apply(extract_numeric_value)
        # Check negative ratings
        neg_ratings = int((ratings < 0).sum())
        if neg_ratings > 0:
            ratings = ratings.apply(lambda r: np.nan if r < 0 else r)
            anomalies["negative_ratings"] = neg_ratings
            audit.append(f"Sanitized {neg_ratings} negative rating values.")
        cleaned["cleaned_rating"] = ratings.round(2)
    else:
        cleaned["cleaned_rating"] = np.nan

    runtime_col = schema_mapping.get_original("runtime")
    if runtime_col and runtime_col in cleaned.columns:
        runtimes = cleaned[runtime_col].apply(extract_numeric_value)
        neg_runtimes = int((runtimes < 0).sum())
        if neg_runtimes > 0:
            runtimes = runtimes.apply(lambda r: np.nan if r < 0 else r)
            anomalies["negative_runtimes"] = neg_runtimes
            audit.append(f"Sanitized {neg_runtimes} negative runtime values.")
        cleaned["cleaned_runtime"] = runtimes
    else:
        cleaned["cleaned_runtime"] = np.nan

    vote_count_col = schema_mapping.get_original("vote_count")
    if vote_count_col and vote_count_col in cleaned.columns:
        votes = cleaned[vote_count_col].apply(extract_numeric_value)
        neg_votes = int((votes < 0).sum())
        if neg_votes > 0:
            votes = votes.apply(lambda v: np.nan if v < 0 else v)
            anomalies["negative_vote_counts"] = neg_votes
            audit.append(f"Sanitized {neg_votes} negative vote counts.")
        cleaned["cleaned_vote_count"] = votes
    else:
        cleaned["cleaned_vote_count"] = np.nan

    # 7. Disambiguate Potential Duplicate Movie Titles
    # Movies with same title (e.g. remake or same name in different years)
    if title_orig_col and title_orig_col in cleaned.columns:
        cleaned["raw_title"] = cleaned[title_orig_col].astype(str)
        title_counts = cleaned["raw_title"].value_counts()
        potential_dups = title_counts[title_counts > 1]
        dup_titles_count = int(potential_dups.sum())

        display_titles = []
        seen_title_indices: dict[str, int] = {}

        for _, row in cleaned.iterrows():
            title = row["raw_title"]
            if title_counts[title] > 1:
                year = row["cleaned_release_year"]
                if pd.notna(year):
                    display_title = f"{title} ({int(year)})"
                else:
                    count = seen_title_indices.get(title, 0) + 1
                    seen_title_indices[title] = count
                    display_title = f"{title} (#{count})"
            else:
                display_title = title
            display_titles.append(display_title)

        cleaned["display_title"] = display_titles
        if dup_titles_count > 0:
            audit.append(
                f"Identified {dup_titles_count} records with duplicate titles. Disambiguated in 'display_title'."
            )
    else:
        cleaned["display_title"] = [f"Movie {i+1}" for i in range(len(cleaned))]
        cleaned["raw_title"] = cleaned["display_title"]
        dup_titles_count = 0

    cleaned_rows, cleaned_cols = cleaned.shape
    missing_after = {col: int(cleaned[col].isna().sum()) for col in cleaned.columns}

    report = CleaningReport(
        original_rows=orig_rows,
        original_cols=orig_cols,
        cleaned_rows=cleaned_rows,
        cleaned_cols=cleaned_cols,
        exact_duplicates_removed=exact_dups,
        potential_duplicate_titles_count=dup_titles_count,
        missing_titles_removed=missing_titles,
        missing_values_before=missing_before,
        missing_values_after=missing_after,
        anomalies_handled=anomalies,
        available_recommendation_features=schema_mapping.available_features,
        audit_log=audit,
    )

    logger.info(f"Cleaning complete. Output shape: {cleaned.shape}. Exact dups removed: {exact_dups}")
    return cleaned, report
