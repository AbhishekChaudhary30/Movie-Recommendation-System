"""Data engineering module: loading, schema mapping, validation, and cleaning."""

from src.data.cleaning import CleaningReport, clean_dataset
from src.data.loader import check_file_size, load_csv
from src.data.posters import (
    get_poster_source,
    normalize_title_for_lookup,
    render_poster_card_element,
    render_poster_thumbnail_element,
)
from src.data.schema import SchemaMapping, detect_schema, normalize_column_name
from src.data.validation import ValidationReport, validate_dataset

__all__ = [
    "CleaningReport",
    "SchemaMapping",
    "ValidationReport",
    "check_file_size",
    "clean_dataset",
    "detect_schema",
    "get_poster_source",
    "load_csv",
    "normalize_column_name",
    "normalize_title_for_lookup",
    "render_poster_card_element",
    "render_poster_thumbnail_element",
    "validate_dataset",
]

