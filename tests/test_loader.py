"""Unit tests for dataset loading and file validation."""

from io import StringIO
import pytest
import pandas as pd
from src.data.loader import check_file_size, load_csv
from src.utils.exceptions import DataLoadError


def test_load_valid_csv(tmp_path):
    """Test loading a normal valid CSV file."""
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text("title,genres\nInception,Action\nInterstellar,Sci-Fi\n", encoding="utf-8")

    df = load_csv(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["title", "genres"]


def test_load_empty_file_raises_error(tmp_path):
    """Test that an empty file raises DataLoadError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadError, match="empty"):
        load_csv(empty_file)


def test_load_no_data_rows(tmp_path):
    """Test that a file with header only but 0 data rows raises DataLoadError."""
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("title,genres\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="0 data rows"):
        load_csv(header_only)


def test_load_csv_latin1_encoding(tmp_path):
    """Test that files with latin-1 characters are read without error."""
    latin1_file = tmp_path / "latin1.csv"
    # 'Amélie' with accented character in latin-1
    content = "title,genres\nAmélie,Romance\n"
    latin1_file.write_bytes(content.encode("latin-1"))

    df = load_csv(latin1_file)
    assert len(df) == 1
    assert "Amélie" in df["title"].iloc[0]


def test_load_csv_from_stringio():
    """Test loading from an in-memory StringIO buffer (simulating Streamlit UploadedFile)."""
    buffer = StringIO("movie_title,score\nAvatar,7.6\nTitanic,7.9\n")
    df = load_csv(buffer)
    assert len(df) == 2
    assert "movie_title" in df.columns


def test_check_file_size_limit(tmp_path):
    """Test that file exceeding max_size_mb raises DataLoadError."""
    large_file = tmp_path / "oversize.csv"
    # Write 1024 bytes and test with max_size_mb = 0 (or very small threshold)
    large_file.write_text("a" * 2000, encoding="utf-8")
    
    with pytest.raises(DataLoadError, match="exceeds maximum"):
        # 1 byte limit to trigger size error
        check_file_size(large_file, max_size_mb=0)
