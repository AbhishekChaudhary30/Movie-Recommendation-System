"""Unit tests for similarity calculations."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.recommendation.similarity import (
    compute_full_similarity_matrix,
    compute_query_similarity,
)


def test_compute_query_similarity_bounds_and_values():
    """Test query similarity returns scores between 0 and 1, with self-similarity ~ 1.0."""
    corpus = [
        "action superhero space alien",
        "action superhero save world",
        "romantic comedy love wedding",
    ]
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(corpus)

    # Query index 0
    scores = compute_query_similarity(0, tfidf)

    assert len(scores) == 3
    assert np.isclose(scores[0], 1.0)  # Self similarity
    assert scores[1] > scores[2]       # Superhero is more similar to superhero than romantic comedy
    assert all(0.0 <= s <= 1.0001 for s in scores)


def test_compute_query_similarity_out_of_bounds():
    """Test out-of-bounds query index raises IndexError."""
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(["movie one", "movie two"])

    with pytest.raises(IndexError):
        compute_query_similarity(5, tfidf)

    with pytest.raises(IndexError):
        compute_query_similarity(-1, tfidf)


def test_compute_full_similarity_matrix():
    """Test pairwise similarity matrix calculation and size guard."""
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(["apple banana", "banana cherry", "date elderberry"])

    sim_matrix = compute_full_similarity_matrix(tfidf, max_rows=10)
    assert sim_matrix.shape == (3, 3)
    assert np.allclose(np.diag(sim_matrix), 1.0)

    # Exceeding max_rows guard
    with pytest.raises(ValueError, match="exceeding safe dense matrix threshold"):
        compute_full_similarity_matrix(tfidf, max_rows=2)
