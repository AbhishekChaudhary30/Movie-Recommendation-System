"""Schema detection and mapping layer for movie datasets."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import re
from src.config import SCHEMA_SYNONYMS

@dataclass
class SchemaMapping:
    """Represents mapped dataset columns to canonical application schema."""
    canonical_to_original: Dict[str, str] = field(default_factory=dict)
    original_to_canonical: Dict[str, str] = field(default_factory=dict)
    unmapped_columns: List[str] = field(default_factory=list)
    available_features: List[str] = field(default_factory=list)
    has_title: bool = False

    def get_original(self, canonical_name: str) -> Optional[str]:
        """Return the original column name mapped to a canonical key."""
        return self.canonical_to_original.get(canonical_name)

    def get_canonical(self, original_name: str) -> Optional[str]:
        """Return the canonical key mapped to an original column name."""
        return self.original_to_canonical.get(original_name)


def normalize_column_name(col_name: str) -> str:
    """Normalize column name for robust matching: lowercase, strip, single underscores."""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    col_clean = col_name.strip().lower()
    col_clean = re.sub(r"[\s\-\.]+", "_", col_clean)
    col_clean = re.sub(r"[^\w_]", "", col_clean)
    col_clean = col_clean.strip("_")
    return col_clean


def detect_schema(
    columns: List[str],
    user_overrides: Optional[Dict[str, str]] = None
) -> SchemaMapping:
    """
    Detect canonical schema from raw column names.
    Supports user overrides where user_overrides maps {canonical_name: original_column}.
    """
    canonical_to_orig: Dict[str, str] = {}
    orig_to_canonical: Dict[str, str] = {}
    normalized_to_orig: Dict[str, str] = {
        normalize_column_name(col): col for col in columns
    }
    
    assigned_origs: Set[str] = set()

    # Apply user overrides first if provided and valid
    if user_overrides:
        for canonical, orig in user_overrides.items():
            if orig in columns and canonical in SCHEMA_SYNONYMS:
                canonical_to_orig[canonical] = orig
                orig_to_canonical[orig] = canonical
                assigned_origs.add(orig)

    # Detect remaining canonical fields using synonym matching
    for canonical, synonyms in SCHEMA_SYNONYMS.items():
        if canonical in canonical_to_orig:
            continue

        matched_orig = None
        for syn in synonyms:
            norm_syn = normalize_column_name(syn)
            if norm_syn in normalized_to_orig:
                candidate = normalized_to_orig[norm_syn]
                if candidate not in assigned_origs:
                    matched_orig = candidate
                    break

        if matched_orig:
            canonical_to_orig[canonical] = matched_orig
            orig_to_canonical[matched_orig] = canonical
            assigned_origs.add(matched_orig)

    # Identify unmapped columns
    unmapped = [col for col in columns if col not in assigned_origs]

    # Check for title column
    has_title = "title" in canonical_to_orig

    # Identify available recommendation features
    rec_candidate_keys = ["genres", "overview", "keywords", "cast", "director"]
    available_features = [
        k for k in rec_candidate_keys if k in canonical_to_orig
    ]

    return SchemaMapping(
        canonical_to_original=canonical_to_orig,
        original_to_canonical=orig_to_canonical,
        unmapped_columns=unmapped,
        available_features=available_features,
        has_title=has_title,
    )
