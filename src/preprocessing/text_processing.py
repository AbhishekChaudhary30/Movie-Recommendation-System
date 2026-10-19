"""Text preprocessing routines for recommendation metadata."""

import re

import pandas as pd


def clean_text(text: str | None) -> str:
    """Normalize general text: lowercase, remove special characters, collapse whitespace."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text).lower()
    # Replace non-alphanumeric (except standard spaces) with space
    text_str = re.sub(r"[^\w\s]", " ", text_str)
    # Collapse multiple spaces
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str


def entity_to_token(name: str) -> str:
    """
    Transform names/entities into joined tokens to avoid ambiguity.
    Example: 'Christopher Nolan' -> 'christopher_nolan'
    """
    if not name or pd.isna(name):
        return ""
    clean = re.sub(r"[^\w\s]", "", str(name).lower())
    clean = re.sub(r"\s+", "_", clean.strip())
    return clean


def process_entities_string(val_str: str | None, max_items: int = 5) -> str:
    """
    Tokenize comma-separated entity lists.
    Example: 'Leonardo DiCaprio, Joseph Gordon-Levitt'
          -> 'leonardo_dicaprio joseph_gordonlevitt'
    """
    if not val_str or pd.isna(val_str):
        return ""
    tokens: list[str] = []
    for item in str(val_str).split(","):
        token = entity_to_token(item)
        if token:
            tokens.append(token)
    return " ".join(tokens[:max_items])


def process_genres_string(genres_str: str | None) -> str:
    """
    Normalize genres string into clean tokens.
    Example: 'Science Fiction, Action' -> 'science_fiction action'
    """
    if not genres_str or pd.isna(genres_str):
        return ""
    tokens: list[str] = []
    for g in str(genres_str).split(","):
        g_clean = entity_to_token(g)
        if g_clean:
            tokens.append(g_clean)
    return " ".join(tokens)
