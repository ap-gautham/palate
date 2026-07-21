"""Join this project's own film catalog to the gsimonx37/letterboxd Kaggle
metadata dump (genres, themes, studios, cast, crew, countries, languages) by
normalized (title, year), producing per-film facet sets and a small numeric
tail. This gives the trained models real movie content beyond the single
first-listed genre previously used, and lets a member's own seen history
establish per-facet taste ("this member over-rates A24 films / Tarantino /
1990s films") via the affinity features built downstream in `features.py`
(this module only supplies the facet *sets*, not the affinity numbers).

Design: each of the eight facets below gets a user-affinity feature pair
(mean deviation + confidence on seen films sharing that facet) in
`features.py`. Only the two genuinely low-cardinality facets, genre and
decade, also get a target-side multi-hot descriptor (a global base-rate
signal); the high-cardinality facets (theme/language/country/studio/
director/actor) rely on the affinity pair alone, to avoid an unmanageable
one-hot width and unreliable rare-category statistics.

This module is duplicated near-verbatim in
`rotten_tomatoes/movie_features.py` so the two projects stay fully
code-isolated; only the catalog join keys and the own-dataset genre fallback
differ.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from .config import PROCESSED, RAW, SEED

GSIMONX37_DIR = RAW / "gsimonx37"
FACETS_CACHE = PROCESSED / "movie_facets.pkl"

FACETS = ["genre", "decade", "theme", "language", "country", "studio", "director", "actor"]
MULTIHOT_FACETS = ["genre", "decade"]   # low-cardinality: also get a target multi-hot
# Fixed vocab widths so FEATURE_COLS (features.py) is a static list that never
# depends on running the gsimonx37 join first -- unused ids simply stay all-zero.
GENRE_VOCAB_K = 30
DECADE_VOCAB_K = 20
TOP_ACTORS_PER_FILM = 3


def _norm_title(title: object) -> str:
    """Lowercase, drop a leading article, strip punctuation, collapse spaces."""
    if not isinstance(title, str):
        return ""
    t = title.lower().strip()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Gsimonx37:
    """Per-gsimonx37-id facet lists and numerics, plus the (title, year) index
    used to join to a project's own catalog."""
    by_title_year: dict            # (norm_title, year) -> gs_id (first match)
    by_title: dict                 # norm_title -> gs_id (only when unambiguous)
    genre: dict                    # gs_id -> list[str]
    theme: dict
    language: dict
    country: dict
    studio: dict
    director: dict
    actor: dict
    runtime: dict                  # gs_id -> float minutes
    rating: dict                   # gs_id -> float (gsimonx37's own site-average rating)
    n_themes: dict
    n_languages: dict
    n_countries: dict


def load_gsimonx37() -> Gsimonx37:
    """Load and index the raw gsimonx37 CSVs (data/letterboxd/raw/gsimonx37/).
    Excludes the (unused, undownloaded) posters file."""
    movies = pd.read_csv(GSIMONX37_DIR / "movies.csv",
                         usecols=["id", "name", "date", "minute", "rating"])
    genres = pd.read_csv(GSIMONX37_DIR / "genres.csv")
    themes = pd.read_csv(GSIMONX37_DIR / "themes.csv")
    studios = pd.read_csv(GSIMONX37_DIR / "studios.csv")
    actors = pd.read_csv(GSIMONX37_DIR / "actors.csv", usecols=["id", "name"])
    crew = pd.read_csv(GSIMONX37_DIR / "crew.csv", usecols=["id", "role", "name"])
    countries = pd.read_csv(GSIMONX37_DIR / "countries.csv")
    languages = pd.read_csv(GSIMONX37_DIR / "languages.csv")

    def group_list(df, key, col):
        return df.groupby(key)[col].apply(list).to_dict()

    genre_by_id = group_list(genres, "id", "genre")
    theme_by_id = group_list(themes, "id", "theme")
    studio_by_id = group_list(studios, "id", "studio")
    director_by_id = group_list(crew[crew["role"] == "Director"], "id", "name")
    actor_by_id = (actors.groupby("id")["name"]
                   .apply(lambda s: list(s)[:TOP_ACTORS_PER_FILM]).to_dict())
    country_by_id = group_list(countries, "id", "country")
    language_by_id = group_list(languages[languages["type"] == "Language"], "id", "language")

    movies["norm_title"] = movies["name"].map(_norm_title)
    by_title_year: dict = {}
    title_counts: dict = {}
    for row in movies.itertuples(index=False):
        key = (row.norm_title, int(row.date) if pd.notna(row.date) else None)
        by_title_year.setdefault(key, row.id)
        title_counts[row.norm_title] = title_counts.get(row.norm_title, 0) + 1
    by_title = {t: by_title_year[(t, y)] for (t, y) in by_title_year
                if title_counts.get(t, 0) == 1}

    runtime = movies.set_index("id")["minute"].astype(float).to_dict()
    rating = movies.set_index("id")["rating"].astype(float).to_dict()
    n_themes = {k: len(v) for k, v in theme_by_id.items()}
    n_languages = {k: len(v) for k, v in language_by_id.items()}
    n_countries = {k: len(v) for k, v in country_by_id.items()}

    return Gsimonx37(by_title_year, by_title, genre_by_id, theme_by_id,
                     language_by_id, country_by_id, studio_by_id, director_by_id,
                     actor_by_id, runtime, rating, n_themes, n_languages, n_countries)


def _top_k_vocab(facet_lists: list, k: int) -> dict:
    """Top-k values by frequency get ids 0..k-1; everything else maps to a
    fixed "__other__" id = k, so the vocab (and therefore the multi-hot width)
    is always exactly k+1 regardless of how many distinct values actually
    occur -- ids beyond the real count simply never fire, which keeps
    FEATURE_COLS a static list independent of the data snapshot."""
    counts: dict = {}
    for values in facet_lists:
        for v in values:
            counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts, key=lambda v: -counts[v])[:k]
    vocab = {v: i for i, v in enumerate(ranked)}
    vocab["__other__"] = k
    return vocab


@dataclass
class MovieFacets:
    """Per-movie facet id sets (for affinity overlap) and multi-hot ids (for
    genre/decade), plus the numeric tail. Indexed by this project's own
    `movie_id`. Facets absent from the source or unmatched to gsimonx37 are
    empty sets / NaN numerics -- never missing keys."""
    facet_sets: dict                 # movie_id -> {facet: frozenset[str]}
    genre_multihot: dict              # movie_id -> list[int] ids (own genre vocab)
    decade_multihot: dict             # movie_id -> [int] single-element decade id
    genre_vocab: dict
    decade_vocab: dict
    runtime_log: dict                # movie_id -> float
    gs_rating: dict                  # movie_id -> float
    n_themes: dict
    n_languages: dict
    n_countries: dict
    match_rate: float


def build_movie_facets(movies: pd.DataFrame, own_genre: dict | None = None) -> MovieFacets:
    """``movies`` is this project's own movie table with ``movie_id``,
    ``title``, ``year`` columns. ``own_genre``, if given, maps movie_id -> the
    project's own first-listed genre string, used as a fallback whenever the
    gsimonx37 join misses (keeps every film's genre facet non-empty when
    possible)."""
    gs = load_gsimonx37()
    matched = 0
    facet_sets, genre_mh, decade_mh = {}, {}, {}
    runtime_log, gs_rating = {}, {}
    n_themes, n_languages, n_countries = {}, {}, {}
    genre_lists, decade_lists = [], []

    resolved = {}
    for row in movies.itertuples(index=False):
        title = _norm_title(row.title)
        year = int(row.year) if pd.notna(getattr(row, "year", None)) else None
        gs_id = gs.by_title_year.get((title, year))
        if gs_id is None:
            gs_id = gs.by_title.get(title)
        resolved[row.movie_id] = gs_id
        if gs_id is not None:
            matched += 1

    for row in movies.itertuples(index=False):
        mid = row.movie_id
        gs_id = resolved[mid]
        genre = list(gs.genre.get(gs_id, [])) if gs_id is not None else []
        if not genre and own_genre is not None:
            fallback = own_genre.get(mid)
            if isinstance(fallback, list):
                genre = fallback
            elif fallback:
                genre = [fallback]
        year = int(row.year) if pd.notna(getattr(row, "year", None)) else None
        decade = [str((year // 10) * 10)] if year is not None else ["__unknown__"]
        facet_sets[mid] = {
            "genre": frozenset(genre),
            "decade": frozenset(decade),
            "theme": frozenset(gs.theme.get(gs_id, [])) if gs_id is not None else frozenset(),
            "language": frozenset(gs.language.get(gs_id, [])) if gs_id is not None else frozenset(),
            "country": frozenset(gs.country.get(gs_id, [])) if gs_id is not None else frozenset(),
            "studio": frozenset(gs.studio.get(gs_id, [])) if gs_id is not None else frozenset(),
            "director": frozenset(gs.director.get(gs_id, [])) if gs_id is not None else frozenset(),
            "actor": frozenset(gs.actor.get(gs_id, [])) if gs_id is not None else frozenset(),
        }
        genre_lists.append(genre)
        decade_lists.append(decade)
        runtime_log[mid] = float(np.log1p(gs.runtime.get(gs_id, np.nan))) if gs_id is not None else np.nan
        gs_rating[mid] = gs.rating.get(gs_id, np.nan) if gs_id is not None else np.nan
        n_themes[mid] = gs.n_themes.get(gs_id, 0) if gs_id is not None else 0
        n_languages[mid] = gs.n_languages.get(gs_id, 0) if gs_id is not None else 0
        n_countries[mid] = gs.n_countries.get(gs_id, 0) if gs_id is not None else 0

    genre_vocab = _top_k_vocab(genre_lists, k=GENRE_VOCAB_K)
    decade_vocab = _top_k_vocab(decade_lists, k=DECADE_VOCAB_K)
    for mid, genres in zip(movies["movie_id"], genre_lists):
        genre_mh[mid] = sorted({genre_vocab.get(g, genre_vocab["__other__"]) for g in genres})
    for mid, decades in zip(movies["movie_id"], decade_lists):
        decade_mh[mid] = sorted({decade_vocab.get(d, decade_vocab["__other__"]) for d in decades})

    match_rate = matched / max(len(movies), 1)
    return MovieFacets(facet_sets, genre_mh, decade_mh, genre_vocab, decade_vocab,
                       runtime_log, gs_rating, n_themes, n_languages, n_countries, match_rate)


def load_or_build_movie_facets(movies: pd.DataFrame, own_genre: dict | None = None,
                               force: bool = False) -> MovieFacets:
    """Cache `build_movie_facets` to disk -- the gsimonx37 join is a ~100s
    scan over ~285k rows and every training/analysis script needs the same
    result, so build it once and reuse it. Delete the cache file (or pass
    `force=True`) after the catalog or gsimonx37 data changes."""
    if not force and FACETS_CACHE.exists():
        with open(FACETS_CACHE, "rb") as f:
            return pickle.load(f)
    facets = build_movie_facets(movies, own_genre=own_genre)
    FACETS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(FACETS_CACHE, "wb") as f:
        pickle.dump(facets, f)
    return facets


# ---- "movies like this one" suggestions (K-means, app catalog only) --------
# Offline, deterministic precomputation for the app's suggestion dropdown ("to
# improve this prediction, rate one of these similar films"): cluster the
# app's ~1,000-film catalog on content facets with K-means, then for each
# film return its k_neighbors nearest neighbours within its own cluster.
# Mirrors rotten_tomatoes/movie_features.py's top_similar exactly.
_TOP_STUDIO_K = 30
_TOP_DIRECTOR_K = 30
_TOP_ACTOR_K = 30
_TOP_THEME_K = 30
_TOP_LANGUAGE_K = 15
_TOP_COUNTRY_K = 15
_SIMILARITY_FACETS = {
    "studio": _TOP_STUDIO_K, "director": _TOP_DIRECTOR_K, "actor": _TOP_ACTOR_K,
    "theme": _TOP_THEME_K, "language": _TOP_LANGUAGE_K, "country": _TOP_COUNTRY_K,
}


def _facet_vocab(movies_json: list[dict], facet: str, k: int) -> dict:
    """Catalog-scoped top-k vocabulary (by frequency) for one facet -- this is
    intentionally a *different* vocab from the trained models' genre/decade
    one (movie_features.GENRE_VOCAB_K etc): it only needs to distinguish
    similarity within this app catalog, not align with a model's columns."""
    counts: dict = {}
    for m in movies_json:
        for v in m["facets"].get(facet, []):
            counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts, key=lambda v: -counts[v])[:k]
    return {v: i for i, v in enumerate(ranked)}


def _multihot(values: list, vocab: dict, width: int) -> np.ndarray:
    row = np.zeros(width, dtype=np.float64)
    for v in values:
        i = vocab.get(v)
        if i is not None:
            row[i] = 1.0
    return row


def top_similar(movies_json: list[dict], consensus: dict, k_neighbors: int = 20,
                seed: int = SEED) -> list[list[int]]:
    """Content-based "movies like this one" neighbours, position-aligned to
    ``movies_json``. Builds one standardized feature vector per film --
    genre/decade multi-hot (the trained models' fixed vocab, from
    ``genreMh``/``decadeMh``), studio/director/actor/theme/language/country
    multi-hots (a catalog-scoped top-k vocabulary per facet, so cast and crew
    are explicitly part of the similarity, not just genre), and standardized
    numerics (runtime, the gsimonx37 site rating, facet counts, release year,
    and ``consensus`` -- the one signal not already in ``movies_json``:
    Tomatometer-mapped score for Rotten Tomatoes, mean member rating for
    Letterboxd, keyed by movie id).

    Clusters with K-means (``k = max(2, n // 25)`` clusters, ~25 films each)
    and returns each film's ``k_neighbors`` nearest neighbours (Euclidean, on
    the standardized vector) within its own cluster; a cluster smaller than
    ``k_neighbors`` is topped up from the globally nearest remaining films so
    every movie always gets a full list. Deterministic for a fixed ``seed``.
    """
    n = len(movies_json)
    vocabs = {f: _facet_vocab(movies_json, f, k) for f, k in _SIMILARITY_FACETS.items()}
    genre_width = GENRE_VOCAB_K + 1
    decade_width = DECADE_VOCAB_K + 1
    genre_ids = {i: i for i in range(genre_width)}
    decade_ids = {i: i for i in range(decade_width)}

    rows = []
    for m in movies_json:
        genre_row = _multihot(m["genreMh"], genre_ids, genre_width)
        decade_row = _multihot(m["decadeMh"], decade_ids, decade_width)
        facet_rows = [_multihot(m["facets"].get(f, []), vocabs[f], k)
                     for f, k in _SIMILARITY_FACETS.items()]
        numeric = np.array([
            m.get("runtimeLog") or 0.0,
            m.get("gsRating") or 0.0,
            m.get("nThemesLog") or 0.0,
            m.get("nLanguagesLog") or 0.0,
            m.get("nCountriesLog") or 0.0,
            float(m.get("year") or 0),
            float(consensus.get(m["id"]) or 0.0),
        ], dtype=np.float64)
        rows.append(np.concatenate([genre_row, decade_row, *facet_rows, numeric]))

    data = np.vstack(rows)
    mu = data.mean(axis=0)
    sd = data.std(axis=0)
    sd[sd < 1e-9] = 1.0
    data = (data - mu) / sd

    k = max(2, min(n - 1, n // 25)) if n > 2 else 1
    _, labels = kmeans2(data, k, seed=seed, minit="++")

    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    result = []
    for i in range(n):
        def dist(j, i=i):
            diff = data[i] - data[j]
            return float(diff @ diff)
        cluster_members = sorted((j for j in clusters[int(labels[i])] if j != i), key=dist)
        chosen = cluster_members[:k_neighbors]
        if len(chosen) < k_neighbors:
            chosen_set = set(chosen)
            rest = sorted((j for j in range(n) if j != i and j not in chosen_set), key=dist)
            chosen += rest[:k_neighbors - len(chosen)]
        result.append(chosen)
    return result
