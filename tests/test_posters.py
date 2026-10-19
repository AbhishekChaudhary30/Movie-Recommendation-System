from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import BASE_DIR, SAMPLE_DATA_DIR
from src.data.posters import (
    _POSTER_REGISTRY,
    get_poster_source,
    normalize_title_for_lookup,
    render_poster_card_element,
    render_poster_thumbnail_element,
)


def test_all_sample_movies_resolve_genuine_posters():
    """Verify that all 50 movies in sample_movies.csv resolve to genuine verified posters."""
    sample_file = SAMPLE_DATA_DIR / "sample_movies.csv"
    assert sample_file.exists(), "Sample dataset file must exist"

    df = pd.read_csv(sample_file)
    assert len(df) == 50, "Sample dataset must contain 50 movies"

    resolved_sources = {}
    for _, row in df.iterrows():
        movie = row.to_dict()
        title = movie["title"]
        source = get_poster_source(movie)
        assert source is not None, f"Movie '{title}' must resolve a verified poster"
        assert (
            source.startswith(("data:image/", "https://", "http://"))
            or Path(source).exists()
        ), f"Invalid source format for '{title}'"
        resolved_sources[title] = source

    # Ensure no two different movies are mistakenly assigned the exact same poster
    unique_sources = set(resolved_sources.values())
    assert len(unique_sources) == 50, "Every movie must have a unique genuine poster (no accidental duplicate assignment)"


def test_all_50_local_posters_are_high_resolution():
    """Verify that all 50 catalog movies have genuine 500x750 HD portrait poster assets."""
    posters_dir = BASE_DIR / "assets" / "posters"
    assert posters_dir.exists() and posters_dir.is_dir(), "Posters directory must exist"

    poster_files = list(posters_dir.glob("*.jpg"))
    assert len(poster_files) == 50, f"Expected 50 HD poster image files, found {len(poster_files)}"

    for p in poster_files:
        with Image.open(p) as img:
            w, h = img.size
            assert w >= 500, f"Poster {p.name} width {w}px is less than 500px HD requirement"
            assert h >= 700, f"Poster {p.name} height {h}px is less than 700px HD requirement"
            aspect = w / h
            assert 0.60 <= aspect <= 0.72, f"Poster {p.name} aspect ratio {aspect:.2f} deviates from portrait standard"


def test_poster_filename_title_matching():
    """Verify that resolved poster files accurately correspond to movie identities."""
    checks = [
        ("Inception", 2010, "inception_2010"),
        ("Interstellar", 2014, "interstellar_2014"),
        ("The Dark Knight", 2008, "the_dark_knight_2008"),
        ("Pulp Fiction", 1994, "pulp_fiction_1994"),
        ("The Matrix", 1999, "the_matrix_1999"),
        ("The Godfather", 1972, "the_godfather_1972"),
        ("Parasite", 2019, "parasite_2019"),
        ("Spirited Away", 2001, "spirited_away_2001"),
    ]
    for title, year, expected_slug in checks:
        norm_t = normalize_title_for_lookup(title)
        entry = _POSTER_REGISTRY.get(f"{norm_t}::{year}") or _POSTER_REGISTRY.get(norm_t)
        assert entry is not None, f"Registry entry for '{title}' must exist"
        assert expected_slug in entry["filename"].lower(), f"Filename for '{title}' must contain '{expected_slug}'"


def test_direct_poster_url_priority():
    """Verify that direct poster_url or poster_path in movie dict takes precedence."""
    movie = {
        "title": "Inception",
        "release_year": 2010,
        "poster_url": "https://custom-cdn.example.com/custom_inception.jpg",
    }
    src = get_poster_source(movie)
    assert src == "https://custom-cdn.example.com/custom_inception.jpg"


def test_tmdb_relative_path_resolution():
    """Verify that a TMDB relative poster path (/path.jpg) is correctly formatted."""
    movie = {
        "title": "Arbitrary Movie",
        "poster_path": "/test_poster_12345.jpg",
    }
    src = get_poster_source(movie)
    assert src == "https://image.tmdb.org/t/p/w500/test_poster_12345.jpg"


def test_unknown_movie_fallback():
    """Verify that movies without posters return None and render neutral fallback HTML."""
    movie = {
        "title": "Completely Unindexed Indie Film 2099",
        "release_year": 2099,
    }
    src = get_poster_source(movie)
    assert src is None

    html = render_poster_card_element(movie, height=260)
    assert "Poster Unavailable" in html
    assert "Completely Unindexed Indie Film 2099" in html
    assert "height: 260px" in html


def test_render_poster_card_element_has_onerror_safety():
    """Verify that rendered poster HTML includes an onerror fallback handler."""
    movie = {"title": "Inception", "release_year": 2010}
    html = render_poster_card_element(movie, height=270)
    assert "img" in html
    assert "onerror=" in html
    assert "Poster Unavailable" in html
    assert "height: 270px" in html


def test_render_poster_thumbnail_element():
    """Verify compact thumbnail rendering for recommendation cards."""
    movie = {"title": "Interstellar", "release_year": 2014}
    html = render_poster_thumbnail_element(movie, width=100, height=150)
    assert "width: 100px" in html
    assert "height: 150px" in html
    assert "img" in html

    unknown = {"title": "Unknown Mystery Film"}
    html_unk = render_poster_thumbnail_element(unknown, width=100, height=150)
    assert "No Poster" in html_unk


def test_custom_uploaded_dataset_without_posters():
    """Verify that arbitrary uploaded datasets without poster columns do not crash."""
    custom_df = pd.DataFrame([
        {"title": "Custom Movie A", "genres": "Action"},
        {"title": "Custom Movie B", "genres": "Drama"},
    ])
    for _, row in custom_df.iterrows():
        src = get_poster_source(row.to_dict())
        assert src is None
        html = render_poster_card_element(row.to_dict())
        assert "Poster Unavailable" in html
