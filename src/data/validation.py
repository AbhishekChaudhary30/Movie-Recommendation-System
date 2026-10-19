"""Validation layer for uploaded movie datasets and detected schemas."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.schema import SchemaMapping
from src.utils.logging_config import get_logger

logger = get_logger("validation")

@dataclass
class ValidationReport:
    """Comprehensive validation result with diagnostic checks and warnings."""
    is_valid: bool
    row_count: int
    column_count: int
    schema_mapping: SchemaMapping
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    column_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    suspicious_values: dict[str, int] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serializable logs or exports."""
        return {
            "is_valid": self.is_valid,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "critical_errors": self.critical_errors,
            "warnings": self.warnings,
            "mapped_columns": self.schema_mapping.canonical_to_original,
            "available_features": self.schema_mapping.available_features,
            "suspicious_values": self.suspicious_values,
        }


def validate_dataset(df: pd.DataFrame, schema_mapping: SchemaMapping) -> ValidationReport:
    """
    Validate dataset structure, schema mapping, and column integrity.
    Identifies missing mandatory fields, feature coverage, and suspicious data.
    """
    critical_errors: list[str] = []
    warnings: list[str] = []
    suspicious_values: dict[str, int] = {}
    col_stats: dict[str, dict[str, Any]] = {}

    row_count, col_count = df.shape

    # 1. Structural Checks
    if row_count == 0:
        critical_errors.append("Dataset has zero data rows.")
    if col_count == 0:
        critical_errors.append("Dataset has zero columns.")

    # 2. Check for duplicate column names in DataFrame
    col_names = [str(c) for c in df.columns]
    dup_cols = [c for c in set(col_names) if col_names.count(c) > 1]
    if dup_cols:
        warnings.append(f"Duplicate column headers detected: {dup_cols}. Deduplication recommended.")

    # 3. Mandatory Title Column
    if not schema_mapping.has_title:
        critical_errors.append(
            "Movie identification failed: No column mapped to 'title'. "
            "A movie title column is required to index and recommend movies. "
            "Please assign a title column in the column mapping section."
        )
    else:
        title_col = schema_mapping.get_original("title")
        if title_col and title_col in df.columns:
            null_titles = df[title_col].isna().sum()
            if null_titles == row_count:
                critical_errors.append(
                    f"The mapped title column '{title_col}' is entirely empty (100% missing)."
                )
            elif null_titles > 0:
                pct = (null_titles / row_count) * 100
                warnings.append(
                    f"Mapped title column '{title_col}' contains {null_titles} missing titles ({pct:.1f}%). "
                    "These rows will be excluded during cleaning."
                )

    # 4. Recommendation Features Check
    if not schema_mapping.available_features:
        warnings.append(
            "No recommendation features (genres, overview, keywords, cast, director) detected. "
            "Content-based similarity cannot be computed without at least one descriptive feature."
        )

    # 5. Column-by-Column Statistics and Suspicious Value Detection
    for col in df.columns:
        col_str = str(col)
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / row_count) * 100, 2) if row_count > 0 else 0.0
        unique_count = int(df[col].nunique(dropna=True))

        col_stats[col_str] = {
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
        }

        # Check mapped numeric columns for anomalies
        canon_name = schema_mapping.get_canonical(col_str)
        if canon_name in ["runtime", "vote_count", "rating"]:
            # Coerce to numeric for check
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            neg_count = int((numeric_vals < 0).sum())
            if neg_count > 0:
                suspicious_values[f"negative_{col_str}"] = neg_count
                warnings.append(
                    f"Found {neg_count} negative value(s) in '{col_str}'. "
                    "These will be flagged or sanitized during cleaning."
                )

            if canon_name == "rating":
                high_count = int((numeric_vals > 100).sum())
                if high_count > 0:
                    warnings.append(
                        f"Found {high_count} rating values > 100 in '{col_str}'. Verify rating scale."
                    )

    is_valid = len(critical_errors) == 0

    logger.info(
        f"Validation completed. Valid: {is_valid}, Critical Errors: {len(critical_errors)}, "
        f"Warnings: {len(warnings)}"
    )

    return ValidationReport(
        is_valid=is_valid,
        row_count=row_count,
        column_count=col_count,
        schema_mapping=schema_mapping,
        critical_errors=critical_errors,
        warnings=warnings,
        column_stats=col_stats,
        suspicious_values=suspicious_values,
    )
