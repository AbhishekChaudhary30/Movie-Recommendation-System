"""Content-based recommendation engine with feature weighting and explainability."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.config import DEFAULT_FEATURE_WEIGHTS, MAX_RECOMMENDATIONS_LIMIT
from src.data.schema import SchemaMapping
from src.preprocessing.feature_engineering import build_movie_soup, vectorize_features
from src.recommendation.similarity import compute_query_similarity
from src.utils.exceptions import InsufficientFeaturesError, RecommendationEngineError
from src.utils.logging_config import get_logger

logger = get_logger("recommender")


@dataclass
class RecommendationItem:
    """Structure for a single recommended movie with explanation."""
    display_title: str
    similarity_score: float
    similarity_pct: float
    release_year: Optional[int]
    rating: Optional[float]
    genres: str
    cast: str
    director: str
    overview: str
    shared_genres: List[str]
    shared_cast: List[str]
    shared_director: Optional[str]
    shared_keywords: List[str]
    shared_themes: List[str]
    explanation: str


class ContentBasedRecommender:
    """
    Content-Based Movie Recommender using TF-IDF representation and Cosine Similarity.
    Features:
      - Multi-feature weighted metadata soup
      - Memory-aware on-demand similarity scoring
      - Configurable metadata filters (genre, year, rating)
      - Self-exclusion guarantee
      - In-depth recommendation explanations
    """

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        self.feature_weights = feature_weights or DEFAULT_FEATURE_WEIGHTS.copy()
        self.df: Optional[pd.DataFrame] = None
        self.schema_mapping: Optional[SchemaMapping] = None
        self.tfidf_matrix: Optional[spmatrix] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.title_to_idx: Dict[str, int] = {}
        self.feature_names: np.ndarray = np.array([])

    def fit(
        self,
        df: pd.DataFrame,
        schema_mapping: SchemaMapping,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> "ContentBasedRecommender":
        """
        Fit the recommender by building metadata soups and vectorizing features.
        """
        if df.empty:
            raise RecommendationEngineError("Cannot fit recommender on empty DataFrame.")

        if feature_weights:
            self.feature_weights = feature_weights.copy()

        self.df = df.copy().reset_index(drop=True)
        self.schema_mapping = schema_mapping

        # Build Title-to-Index Lookup
        self.title_to_idx = {}
        for idx, row in self.df.iterrows():
            disp_title = row.get("display_title", "")
            raw_title = row.get("raw_title", "")
            if disp_title:
                self.title_to_idx[disp_title.lower()] = idx
            if raw_title and raw_title.lower() not in self.title_to_idx:
                self.title_to_idx[raw_title.lower()] = idx

        # Build Content Soup
        logger.info("Building metadata soup for movies...")
        soup_series = build_movie_soup(self.df, self.feature_weights)
        self.df["metadata_soup"] = soup_series

        # Vectorize Features
        logger.info("Vectorizing metadata soups with TF-IDF...")
        self.tfidf_matrix, self.vectorizer = vectorize_features(soup_series)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        return self

    def find_movie_index(self, query: Union[str, int]) -> int:
        """Find the internal row index for a given title or index."""
        if self.df is None or self.df.empty:
            raise RecommendationEngineError("Recommender has not been fitted yet.")

        if isinstance(query, (int, np.integer)):
            if 0 <= query < len(self.df):
                return int(query)
            raise RecommendationEngineError(f"Movie index {query} out of bounds (0 to {len(self.df)-1}).")

        query_clean = str(query).strip().lower()
        if query_clean in self.title_to_idx:
            return self.title_to_idx[query_clean]

        # Try partial matching if exact match not found
        for title_key, idx in self.title_to_idx.items():
            if query_clean in title_key:
                return idx

        raise RecommendationEngineError(
            f"Movie '{query}' not found in the dataset. Please select an available movie."
        )

    def explain_similarity(self, query_idx: int, target_idx: int) -> Dict[str, Any]:
        """
        Extract detailed comparison explaining why target_idx was recommended for query_idx.
        """
        query_row = self.df.iloc[query_idx]
        target_row = self.df.iloc[target_idx]

        def split_items(val: Any) -> Set[str]:
            if pd.isna(val) or not val:
                return set()
            return {x.strip().lower() for x in str(val).split(",") if x.strip()}

        # 1. Matching Genres
        q_genres = split_items(query_row.get("cleaned_genres", ""))
        t_genres = split_items(target_row.get("cleaned_genres", ""))
        shared_genres = sorted(list(q_genres.intersection(t_genres)))

        # 2. Matching Cast
        q_cast = split_items(query_row.get("cleaned_cast", ""))
        t_cast = split_items(target_row.get("cleaned_cast", ""))
        shared_cast = sorted(list(q_cast.intersection(t_cast)))

        # 3. Matching Director
        q_dir = split_items(query_row.get("cleaned_director", ""))
        t_dir = split_items(target_row.get("cleaned_director", ""))
        shared_dir_set = q_dir.intersection(t_dir)
        shared_director = sorted(list(shared_dir_set))[0].title() if shared_dir_set else None

        # 4. Matching Keywords
        q_kw = split_items(query_row.get("cleaned_keywords", ""))
        t_kw = split_items(target_row.get("cleaned_keywords", ""))
        shared_keywords = sorted(list(q_kw.intersection(t_kw)))

        # 5. Overlapping TF-IDF Theme Words
        shared_themes: List[str] = []
        if self.tfidf_matrix is not None and len(self.feature_names) > 0:
            q_vec = self.tfidf_matrix[query_idx].toarray().flatten()
            t_vec = self.tfidf_matrix[target_idx].toarray().flatten()
            overlap_scores = q_vec * t_vec
            top_word_indices = np.argsort(overlap_scores)[::-1]
            for w_idx in top_word_indices[:6]:
                if overlap_scores[w_idx] > 0:
                    shared_themes.append(str(self.feature_names[w_idx]))

        # Synthesize Human-Readable Explanation
        reasons = []
        if shared_genres:
            reasons.append(f"Genres: {', '.join([g.title() for g in shared_genres])}")
        if shared_director:
            reasons.append(f"Director: {shared_director}")
        if shared_cast:
            reasons.append(f"Cast: {', '.join([c.title() for c in shared_cast[:3]])}")
        if shared_keywords:
            reasons.append(f"Keywords: {', '.join([k.title() for k in shared_keywords[:4]])}")
        elif shared_themes:
            reasons.append(f"Themes: {', '.join(shared_themes[:4])}")

        explanation = "; ".join(reasons) if reasons else "General thematic & narrative similarity"

        return {
            "shared_genres": [g.title() for g in shared_genres],
            "shared_cast": [c.title() for c in shared_cast],
            "shared_director": shared_director,
            "shared_keywords": [k.title() for k in shared_keywords],
            "shared_themes": shared_themes,
            "explanation": explanation,
        }

    def recommend(
        self,
        query: Union[str, int],
        top_n: int = 10,
        min_rating: Optional[float] = None,
        genre_filter: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate top N content-based recommendations for a selected movie.
        Excludes the selected movie itself and respects user filters.
        """
        if self.df is None or self.tfidf_matrix is None:
            raise RecommendationEngineError("Recommender is not trained or dataset is missing.")

        query_idx = self.find_movie_index(query)
        top_n = min(max(1, top_n), MAX_RECOMMENDATIONS_LIMIT)

        # Compute Cosine Similarities on-the-fly (O(N) memory)
        sim_scores = compute_query_similarity(query_idx, self.tfidf_matrix)

        # Check if query movie has any non-zero features
        query_norm = np.linalg.norm(self.tfidf_matrix[query_idx].toarray())
        is_weak_query = query_norm < 1e-6

        # Build Candidate Records
        candidates: List[Dict[str, Any]] = []
        for idx, score in enumerate(sim_scores):
            # 1. Strictly exclude the selected movie itself
            if idx == query_idx:
                continue

            # Minimum similarity threshold filter
            if min_similarity is not None and min_similarity > 0:
                # min_similarity can be given as 0.0-1.0 or 0-100%
                threshold = min_similarity if min_similarity <= 1.0 else (min_similarity / 100.0)
                if float(score) < threshold:
                    continue

            row = self.df.iloc[idx]

            # 2. Filter: Genre
            if genre_filter and genre_filter.strip().lower() != "all":
                row_genres = str(row.get("cleaned_genres", "")).lower()
                if genre_filter.strip().lower() not in row_genres:
                    continue

            # 3. Filter: Minimum Rating
            if min_rating is not None and min_rating > 0:
                row_rating = row.get("cleaned_rating")
                if pd.isna(row_rating) or float(row_rating) < float(min_rating):
                    continue

            # 4. Filter: Year Range
            row_year = row.get("cleaned_release_year")
            if pd.notna(row_year):
                try:
                    y = int(row_year)
                    if year_min is not None and y < year_min:
                        continue
                    if year_max is not None and y > year_max:
                        continue
                except (ValueError, TypeError):
                    pass

            candidates.append({
                "idx": idx,
                "display_title": row.get("display_title", "Unknown"),
                "similarity_score": round(float(score), 4),
                "similarity_pct": round(float(score) * 100, 1),
                "cleaned_release_year": row.get("cleaned_release_year"),
                "cleaned_rating": row.get("cleaned_rating"),
                "cleaned_vote_count": row.get("cleaned_vote_count", 0),
                "cleaned_genres": row.get("cleaned_genres", ""),
                "cleaned_cast": row.get("cleaned_cast", ""),
                "cleaned_director": row.get("cleaned_director", ""),
                "cleaned_overview": row.get("cleaned_overview", ""),
            })

        if not candidates:
            logger.info("No candidates satisfied the recommendation filters.")
            return pd.DataFrame()

        # Sort candidates deterministically:
        # 1. Similarity score (descending)
        # 2. Rating (descending, if available)
        # 3. Vote count (descending, if available)
        # 4. Title (alphabetical)
        candidates.sort(
            key=lambda c: (
                c["similarity_score"],
                c["cleaned_rating"] if pd.notna(c["cleaned_rating"]) else 0,
                c["cleaned_vote_count"] if pd.notna(c["cleaned_vote_count"]) else 0,
                c["display_title"],
            ),
            reverse=True,
        )

        top_candidates = candidates[:top_n]

        # Generate Explanations for Top Recommendations
        results = []
        for cand in top_candidates:
            target_idx = cand["idx"]
            exp_info = self.explain_similarity(query_idx, target_idx)
            cand.update(exp_info)
            cand["is_weak_query"] = is_weak_query
            results.append(cand)

        rec_df = pd.DataFrame(results)
        return rec_df

    def get_movie_details(self, query: Union[str, int]) -> Dict[str, Any]:
        """Retrieve full details of a specific movie."""
        idx = self.find_movie_index(query)
        row = self.df.iloc[idx].to_dict()
        row["index"] = idx
        return row

    def compute_pair_similarity(
        self,
        movie1: Union[str, int],
        movie2: Union[str, int],
    ) -> Dict[str, Any]:
        """
        Compute direct pairwise similarity between any two movies in the dataset.
        Returns similarity score, percentage match, and detailed attribute rationale.
        """
        if self.df is None or self.tfidf_matrix is None:
            raise RecommendationEngineError("Recommender is not trained or dataset is missing.")

        idx1 = self.find_movie_index(movie1)
        idx2 = self.find_movie_index(movie2)

        if idx1 == idx2:
            score = 1.0
        else:
            vec1 = self.tfidf_matrix[idx1]
            vec2 = self.tfidf_matrix[idx2]
            score = float(linear_kernel(vec1, vec2)[0, 0])

        score = max(0.0, min(1.0, score))
        explanation_info = self.explain_similarity(idx1, idx2)

        return {
            "movie1_title": self.df.iloc[idx1].get("display_title", str(movie1)),
            "movie2_title": self.df.iloc[idx2].get("display_title", str(movie2)),
            "similarity_score": round(score, 4),
            "similarity_pct": round(score * 100, 1),
            "shared_genres": explanation_info["shared_genres"],
            "shared_cast": explanation_info["shared_cast"],
            "shared_director": explanation_info["shared_director"],
            "shared_keywords": explanation_info["shared_keywords"],
            "shared_themes": explanation_info["shared_themes"],
            "explanation": explanation_info["explanation"],
        }
