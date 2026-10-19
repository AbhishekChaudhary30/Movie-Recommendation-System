"""Data engineering module: loading, schema mapping, validation, and cleaning."""

from src.data.loader import load_csv, check_file_size
from src.data.schema import SchemaMapping, detect_schema, normalize_column_name
from src.data.validation import ValidationReport, validate_dataset
from src.data.cleaning import CleaningReport, clean_dataset

__all__ = [
    "load_csv",
    "check_file_size",
    "SchemaMapping",
    "detect_schema",
    "normalize_column_name",
    "ValidationReport",
    "validate_dataset",
    "CleaningReport",
    "clean_dataset",
]
