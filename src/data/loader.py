"""Dataset loader with robust encoding detection, size validation, and error handling."""

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
from src.config import MAX_FILE_SIZE_MB
from src.utils.exceptions import DataLoadError
from src.utils.logging_config import get_logger

logger = get_logger("data_loader")

# Standard null value representations in common movie datasets
NULL_VALUES = [
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "NULL",
    "None",
    "none",
    "NA",
    "N/A",
    "na",
    "n/a",
    "?",
    "-",
    r"\N",
    "<NA>",
]


def check_file_size(file_input: Union[str, Path, Any], max_size_mb: int = MAX_FILE_SIZE_MB) -> int:
    """
    Check file size against max_size_mb. Returns size in bytes.
    Raises DataLoadError if file is empty or exceeds limit.
    """
    size_bytes = 0
    if isinstance(file_input, (int, float)):
        size_bytes = int(file_input)
    elif isinstance(file_input, (str, Path)):
        p = Path(file_input)
        if not p.exists():
            raise DataLoadError(f"File not found: {p}")
        size_bytes = p.stat().st_size
    elif hasattr(file_input, "size"):
        size_bytes = file_input.size
    elif hasattr(file_input, "getbuffer"):
        size_bytes = len(file_input.getbuffer())
    elif hasattr(file_input, "seek") and hasattr(file_input, "tell"):
        file_input.seek(0, 2)
        size_bytes = file_input.tell()
        file_input.seek(0)
    else:
        # Cannot determine size ahead of time, will check after read
        return 0

    if size_bytes == 0:
        raise DataLoadError("The provided file is empty (0 bytes).")

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        raise DataLoadError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb} MB."
        )

    return size_bytes


def load_csv(
    file_input: Union[str, Path, Any],
    max_size_mb: int = MAX_FILE_SIZE_MB,
    na_values: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load CSV dataset into a pandas DataFrame.
    Tries multiple encodings ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252').
    Validates file integrity and ensures columns and rows exist.
    """
    check_file_size(file_input, max_size_mb=max_size_mb)

    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    na_vals = na_values if na_values is not None else NULL_VALUES

    df: Optional[pd.DataFrame] = None
    last_err: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            # If stream, seek back to 0 before reading
            if hasattr(file_input, "seek"):
                file_input.seek(0)

            df = pd.read_csv(
                file_input,
                encoding=enc,
                na_values=na_vals,
                keep_default_na=True,
                low_memory=False,
            )
            logger.info(f"Successfully loaded CSV using encoding '{enc}'. Shape: {df.shape}")
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_err = exc
            logger.debug(f"Encoding '{enc}' failed: {exc}. Trying next...")
            continue
        except Exception as exc:
            last_err = exc
            break

    if df is None:
        raise DataLoadError(f"Failed to read CSV with supported encodings. Detail: {last_err}")

    # Integrity Checks
    if df.empty and len(df.columns) == 0:
        raise DataLoadError("The CSV file contains no columns and no data.")

    if len(df.columns) == 0:
        raise DataLoadError("The CSV file contains no usable columns.")

    if len(df) == 0:
        raise DataLoadError("The CSV file has headers but contains 0 data rows.")

    # Check for duplicate column names in the raw file
    col_counts = pd.Series(df.columns).value_counts()
    dups = col_counts[col_counts > 1].index.tolist()
    if dups:
        logger.warning(f"Detected duplicate column names in uploaded dataset: {dups}")

    return df
