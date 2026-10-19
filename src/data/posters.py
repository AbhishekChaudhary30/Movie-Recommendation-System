"""Movie poster resolution, verified registry mapping, and reliable card rendering."""

import base64
import functools
import html
import json
import re
from pathlib import Path
from typing import Any

from src.config import BASE_DIR, THEME

# Paths
POSTER_REGISTRY_PATH = Path(__file__).resolve().parent / "poster_registry.json"
ASSETS_POSTERS_DIR = BASE_DIR / "assets" / "posters"

# Load Verified Registry
_POSTER_REGISTRY: dict[str, dict[str, Any]] = {}
try:
    if POSTER_REGISTRY_PATH.exists():
        with open(POSTER_REGISTRY_PATH, "r", encoding="utf-8") as _f:
            _POSTER_REGISTRY = json.load(_f)
except Exception:
    _POSTER_REGISTRY = {}


def normalize_title_for_lookup(title: Any) -> str:
    """Normalize a movie title for deterministic lookup in the poster registry."""
    if not title:
        return ""
    t = str(title).strip()
    # Strip disambiguation suffixes like '(2010)' or '(#1)'
    t = re.sub(r"\s*\(\d{4}\)$", "", t)
    t = re.sub(r"\s*\(#\d+\)$", "", t)
    # Lowercase and normalize whitespace and special chars
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@functools.lru_cache(maxsize=256)
def _load_image_as_data_uri(file_path_str: str) -> str | None:
    """Read a local image file and encode it as a Base64 data URI (cached in memory)."""
    try:
        p = Path(file_path_str)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists() and p.is_file() and p.stat().st_size > 1000:
            ext = p.suffix.lower().lstrip(".")
            mime = "image/png" if ext == "png" else "image/jpeg"
            with open(p, "rb") as f_img:
                encoded = base64.b64encode(f_img.read()).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
    except Exception:
        pass
    return None


def get_poster_source(movie: dict[str, Any]) -> str | None:
    """Resolve the best verified poster source (Data URI or URL) for a movie.

    Resolution Priority:
    1. Direct URL or local path provided in the movie dictionary (e.g. uploaded CSV with poster_path).
    2. Verified local cached poster image in assets/posters/.
    3. Verified public poster URL from the poster registry.
    4. None (clean fallback indicator).
    """
    if not isinstance(movie, dict):
        return None

    # 1. Direct poster column in movie dict (e.g. from user uploaded CSV)
    for col_key in ["poster_path", "poster_url", "poster", "cleaned_poster_path"]:
        val = movie.get(col_key)
        if val and not (isinstance(val, float) and str(val) == "nan"):
            val_str = str(val).strip()
            if val_str.startswith(("http://", "https://", "data:image/")):
                return val_str
            if val_str.startswith("/") and any(val_str.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                # TMDB relative path format
                return f"https://image.tmdb.org/t/p/w500{val_str}"
            # Check if it references an existing local file
            local_p = Path(val_str)
            if not local_p.is_absolute():
                local_p = BASE_DIR / local_p
            if local_p.exists() and local_p.is_file():
                return str(local_p)

    # 2. Check the Verified Poster Registry
    raw_title = movie.get("raw_title") or movie.get("display_title") or movie.get("title") or movie.get("name", "")
    norm_title = normalize_title_for_lookup(raw_title)
    year = movie.get("cleaned_release_year") or movie.get("release_year") or movie.get("year")

    entry = None
    if norm_title:
        # Try exact title + year
        if year and str(year).isdigit():
            lookup_key = f"{norm_title}::{int(year)}"
            entry = _POSTER_REGISTRY.get(lookup_key)

        # Fallback to normalized title alone
        if not entry:
            entry = _POSTER_REGISTRY.get(norm_title)

    if entry:
        local_rel = entry.get("local_path")
        if local_rel:
            p = BASE_DIR / local_rel
            if p.exists() and p.is_file():
                return str(p)

        # If local file is missing, return the verified external URL
        url = entry.get("url")
        if url and str(url).startswith(("http://", "https://")):
            return str(url)

    return None


def render_poster_card_element(movie: dict[str, Any], height: int = 260) -> str:
    """Render HTML for a movie poster inside a 3x3 catalog or comparison card.

    Maintains a 2:3 portrait aspect ratio with object-fit: cover, and includes
    a clean, neutral fallback if the poster is unavailable or fails to load.
    """
    title = str(movie.get("display_title") or movie.get("raw_title") or movie.get("title") or movie.get("name") or "Unknown Title")
    safe_title = html.escape(title)
    poster_src = get_poster_source(movie)

    if poster_src:
        safe_src = html.escape(poster_src)
        return f"""
        <div style="width: 100%; height: {height}px; border-radius: 8px; overflow: hidden; margin-bottom: 12px; background-color: {THEME['surface_secondary']}; border: 1px solid {THEME['border_subtle']}; position: relative;">
            <img src="{safe_src}"
                 alt="{safe_title} official poster"
                 style="width: 100%; height: 100%; object-fit: cover; display: block;"
                 loading="lazy"
                 onerror="this.onerror=null; this.style.display='none'; this.nextElementSibling.style.display='flex';" />
            <div style="display: none; width: 100%; height: 100%; background-color: {THEME['bg_main']}; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 14px; box-sizing: border-box;">
                <div style="font-size: 2.2rem; margin-bottom: 6px; opacity: 0.65;">🎬</div>
                <div style="font-size: 0.86rem; font-weight: 700; color: {THEME['text_primary']}; line-height: 1.25; margin-bottom: 4px; max-width: 90%;">{safe_title}</div>
                <div style="font-size: 0.74rem; color: {THEME['text_secondary']}; font-weight: 600;">Poster Unavailable</div>
            </div>
        </div>
        """

    # Neutral Fallback Card
    return f"""
    <div style="width: 100%; height: {height}px; border-radius: 8px; background-color: {THEME['bg_main']}; border: 1px dashed {THEME['border_subtle']}; display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 12px; text-align: center; padding: 14px; box-sizing: border-box;">
        <div style="font-size: 2.2rem; margin-bottom: 6px; opacity: 0.65;">🎬</div>
        <div style="font-size: 0.86rem; font-weight: 700; color: {THEME['text_primary']}; line-height: 1.25; margin-bottom: 4px; max-width: 90%;">{safe_title}</div>
        <div style="font-size: 0.74rem; color: {THEME['text_secondary']}; font-weight: 600;">Poster Unavailable</div>
    </div>
    """


def render_poster_thumbnail_element(movie: dict[str, Any], width: int = 110, height: int = 165) -> str:
    """Render HTML for a compact portrait poster thumbnail in recommendation lists."""
    title = str(movie.get("display_title") or movie.get("raw_title") or movie.get("title") or movie.get("name") or "Unknown Title")
    safe_title = html.escape(title)
    poster_src = get_poster_source(movie)

    if poster_src:
        safe_src = html.escape(poster_src)
        return f"""
        <div style="width: {width}px; min-width: {width}px; height: {height}px; border-radius: 6px; overflow: hidden; background-color: {THEME['surface_secondary']}; border: 1px solid {THEME['border_subtle']}; position: relative; flex-shrink: 0;">
            <img src="{safe_src}"
                 alt="{safe_title} poster"
                 style="width: 100%; height: 100%; object-fit: cover; display: block;"
                 loading="lazy"
                 onerror="this.onerror=null; this.style.display='none'; this.nextElementSibling.style.display='flex';" />
            <div style="display: none; width: 100%; height: 100%; background-color: {THEME['bg_main']}; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 6px; box-sizing: border-box;">
                <div style="font-size: 1.4rem; opacity: 0.65;">🎬</div>
                <div style="font-size: 0.65rem; color: {THEME['text_secondary']}; font-weight: 600; margin-top: 2px;">No Poster</div>
            </div>
        </div>
        """

    # Neutral Fallback Thumbnail
    return f"""
    <div style="width: {width}px; min-width: {width}px; height: {height}px; border-radius: 6px; background-color: {THEME['bg_main']}; border: 1px dashed {THEME['border_subtle']}; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 6px; box-sizing: border-box; flex-shrink: 0;">
        <div style="font-size: 1.4rem; opacity: 0.65;">🎬</div>
        <div style="font-size: 0.65rem; color: {THEME['text_secondary']}; font-weight: 600; margin-top: 2px;">No Poster</div>
    </div>
    """
