"""Similarity calculation routines with memory-aware sparse linear algebra."""

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import linear_kernel

from src.utils.logging_config import get_logger

logger = get_logger("similarity")


def compute_query_similarity(query_idx: int, tfidf_matrix: spmatrix) -> np.ndarray:
    """
    Compute cosine similarity between a single query movie and all movies in the dataset.
    Because TfidfVectorizer produces L2-normalized rows, linear_kernel computes exact
    cosine similarity in O(N) memory, avoiding dense O(N^2) memory footprint.
    """
    n_docs = tfidf_matrix.shape[0]
    if query_idx < 0 or query_idx >= n_docs:
        raise IndexError(f"Query index {query_idx} is out of bounds for matrix with {n_docs} items.")

    query_vec = tfidf_matrix[query_idx]
    # linear_kernel returns shape (1, n_docs)
    sim_scores = linear_kernel(query_vec, tfidf_matrix).flatten()
    return sim_scores


def compute_full_similarity_matrix(tfidf_matrix: spmatrix, max_rows: int = 3000) -> np.ndarray:
    """
    Compute full pairwise similarity matrix for small datasets or visualization.
    Refuses execution if row count exceeds max_rows to prevent memory exhaustion.
    """
    n_docs = tfidf_matrix.shape[0]
    if n_docs > max_rows:
        raise ValueError(
            f"Dataset has {n_docs} rows, exceeding safe dense matrix threshold of {max_rows}. "
            "Use query-based on-demand similarity instead."
        )
    return linear_kernel(tfidf_matrix, tfidf_matrix)
