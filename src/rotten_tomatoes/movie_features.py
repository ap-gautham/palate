"""Join this project's own film catalog to the gsimonx37/letterboxd Kaggle
metadata dump (genres, themes, cast, crew, languages) by normalized (title,
year), producing per-film facet sets, a canonical genre vocabulary, a theme
embedding similarity matrix, and a small numeric tail. This gives the trained
models real movie content beyond the single first-listed genre previously
used, and lets a member's own seen history establish per-facet taste ("this
member over-rates Tarantino / Denis Villeneuve / horror films") via the
affinity features built downstream in `features.py` (this module only
supplies the facet *sets* and the theme similarity matrix, not the affinity
numbers themselves).

Five facets: genre, theme, language, director, actor. Decade, country and
studio were dropped -- decade duplicates the `year` numeric already in the
row, and country/studio proved low-signal for the columns they cost (an
earlier iteration carried all eight; see report.pdf's feature-engineering
section for the full comparison). Genre is low-cardinality and canonical (the
19 gsimonx37 genre names, fixed, not frequency-ranked) so it gets a per-genre
affinity block built directly off this vocab in `features.py`; theme is
higher-cardinality (109 values) but each one is a descriptive English phrase,
so it is embedded with a sentence-transformer instead of one-hot, letting
"Gothic and eerie haunting horror" and "Terrifying, haunted, and supernatural
horror" be recognized as close even though they share no facet string;
language/director/actor rely on affinity/overlap features computed straight
from the facet sets (no multi-hot).

This module is duplicated near-verbatim from `letterboxd/movie_features.py`
so the two projects stay fully code-isolated; only the catalog join keys and
the own-dataset genre fallback differ. The gsimonx37 CSVs themselves are one
shared external download and physically live under
`data/letterboxd/raw/gsimonx37/` (that's just where they were placed, not a
Letterboxd-specific asset) -- both projects read the same raw files
independently.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import DATA_PROCESSED as PROCESSED, ROOT, SEED

GSIMONX37_DIR = ROOT / "data" / "letterboxd" / "raw" / "gsimonx37"
FACETS_CACHE = PROCESSED / "movie_facets.pkl"
THEME_SIM_CACHE = PROCESSED / "theme_similarity.npz"

FACETS = ["genre", "theme", "language", "director", "actor"]
# The 19 gsimonx37 genre names -- fixed and canonical (not frequency-ranked
# like the old top-k vocab), because the whole vocabulary is known and small.
# "__other__" catches strings from a project's own-genre fallback that don't
# match any of the 19 (see build_movie_facets's own_genre argument).
CANONICAL_GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime",
                    "Documentary", "Drama", "Family", "Fantasy", "History",
                    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
                    "TV Movie", "Thriller", "War", "Western"]
GENRE_VOCAB_K = len(CANONICAL_GENRES)   # 19 named + 1 "__other__" = 20 slots
TOP_ACTORS_PER_FILM = 20
THEME_EMBED_MODEL = "all-MiniLM-L6-v2"


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
    director: dict
    actor: dict
    runtime: dict                  # gs_id -> float minutes
    rating: dict                   # gs_id -> float (gsimonx37's own site-average rating)
    n_themes: dict
    n_languages: dict


def load_gsimonx37() -> Gsimonx37:
    """Load and index the raw gsimonx37 CSVs (data/letterboxd/raw/gsimonx37/).
    Excludes studios/countries (dropped facets) and the (unused, undownloaded)
    posters file."""
    movies = pd.read_csv(GSIMONX37_DIR / "movies.csv",
                         usecols=["id", "name", "date", "minute", "rating"])
    genres = pd.read_csv(GSIMONX37_DIR / "genres.csv")
    themes = pd.read_csv(GSIMONX37_DIR / "themes.csv")
    actors = pd.read_csv(GSIMONX37_DIR / "actors.csv", usecols=["id", "name"])
    crew = pd.read_csv(GSIMONX37_DIR / "crew.csv", usecols=["id", "role", "name"])
    languages = pd.read_csv(GSIMONX37_DIR / "languages.csv")

    def group_list(df, key, col):
        # .agg(list) dispatches through SeriesGroupBy.aggregate rather than
        # the generic (much slower, per-group-callable) .apply path -- same
        # result, ~5x faster with ~800k groups (the dominant cost of a cold
        # cache; the join itself is cheap by comparison).
        return df.groupby(key)[col].agg(list).to_dict()

    genre_by_id = group_list(genres, "id", "genre")
    theme_by_id = group_list(themes, "id", "theme")
    director_by_id = group_list(crew[crew["role"] == "Director"], "id", "name")
    actor_by_id = {k: v[:TOP_ACTORS_PER_FILM] for k, v in group_list(actors, "id", "name").items()}
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

    return Gsimonx37(by_title_year, by_title, genre_by_id, theme_by_id,
                     language_by_id, director_by_id, actor_by_id, runtime, rating,
                     n_themes, n_languages)


def _canonical_genre_vocab() -> dict:
    """The full known gsimonx37 genre vocabulary (19 names, sorted) plus one
    "__other__" slot -- fixed and data-independent, unlike a frequency-ranked
    top-k vocab, because the whole genre vocabulary is small and known."""
    vocab = {g: i for i, g in enumerate(sorted(CANONICAL_GENRES))}
    vocab["__other__"] = len(vocab)
    return vocab


def _top_k_vocab(facet_lists: list, k: int) -> dict:
    """Top-k values by frequency get ids 0..k-1; everything else maps to a
    fixed "__other__" id = k, so the vocab (and therefore the multi-hot width)
    is always exactly k+1 regardless of how many distinct values actually
    occur -- ids beyond the real count simply never fire, which keeps
    FEATURE_COLS a static list independent of the data snapshot. Used for
    the app-catalog similarity vocabs, not genre (which has its own fixed
    canonical vocab, see `_canonical_genre_vocab`)."""
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
    """Per-movie facet id sets (for affinity overlap) and genre multi-hot ids,
    plus the numeric tail. Indexed by this project's own `movie_id`. Facets
    absent from the source or unmatched to gsimonx37 are empty sets / NaN
    numerics -- never missing keys."""
    facet_sets: dict                 # movie_id -> {facet: frozenset[str]}
    genre_multihot: dict              # movie_id -> list[int] ids (canonical genre vocab)
    genre_vocab: dict
    runtime_log: dict                # movie_id -> float
    gs_rating: dict                  # movie_id -> float
    n_themes: dict
    n_languages: dict
    match_rate: float


def build_movie_facets(movies: pd.DataFrame, own_genre: dict | None = None) -> MovieFacets:
    """``movies`` is this project's own movie table with ``movie_id``,
    ``title``, ``year`` columns. ``own_genre``, if given, maps movie_id -> the
    project's own first-listed genre string, used as a fallback whenever the
    gsimonx37 join misses (keeps every film's genre facet non-empty when
    possible)."""
    gs = load_gsimonx37()
    matched = 0
    facet_sets, genre_mh = {}, {}
    runtime_log, gs_rating = {}, {}
    n_themes, n_languages = {}, {}
    genre_lists = []

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
        facet_sets[mid] = {
            "genre": frozenset(genre),
            "theme": frozenset(gs.theme.get(gs_id, [])) if gs_id is not None else frozenset(),
            "language": frozenset(gs.language.get(gs_id, [])) if gs_id is not None else frozenset(),
            "director": frozenset(gs.director.get(gs_id, [])) if gs_id is not None else frozenset(),
            "actor": frozenset(gs.actor.get(gs_id, [])) if gs_id is not None else frozenset(),
        }
        genre_lists.append(genre)
        runtime_log[mid] = float(np.log1p(gs.runtime.get(gs_id, np.nan))) if gs_id is not None else np.nan
        gs_rating[mid] = gs.rating.get(gs_id, np.nan) if gs_id is not None else np.nan
        n_themes[mid] = gs.n_themes.get(gs_id, 0) if gs_id is not None else 0
        n_languages[mid] = gs.n_languages.get(gs_id, 0) if gs_id is not None else 0

    genre_vocab = _canonical_genre_vocab()
    for mid, genres in zip(movies["movie_id"], genre_lists):
        genre_mh[mid] = sorted({genre_vocab.get(g, genre_vocab["__other__"]) for g in genres})

    match_rate = matched / max(len(movies), 1)
    return MovieFacets(facet_sets, genre_mh, genre_vocab,
                       runtime_log, gs_rating, n_themes, n_languages, match_rate)


def prepare_rt_catalog(movies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """RT's own movie table has `movie_id`, `title`, `genre` (a comma-separated
    multi-genre string) and a mostly-missing `releaseDateTheaters` -- not the
    generic `movie_id`/`title`/`year` shape `build_movie_facets` expects. This
    derives a `year` column (from `releaseDateTheaters` where present; NaN
    elsewhere, which falls back to title-only join) and the full own-genre
    list per movie (used as a fallback when gsimonx37 has no match, richer
    than the old first-genre-only parse)."""
    out = movies.drop_duplicates("movie_id").copy()
    year = pd.to_datetime(out["releaseDateTheaters"], errors="coerce").dt.year
    out["year"] = year
    own_genre = (out.set_index("movie_id")["genre"].fillna("")
                .apply(lambda s: [g.strip() for g in s.split(",") if g.strip()]).to_dict())
    return out, own_genre


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


# ---- theme embeddings -------------------------------------------------------
# The 109 gsimonx37 theme phrases are few enough to embed directly: encode
# each with a sentence-transformer, L2-normalize, and cache the resulting
# vocab + cosine similarity matrix. features.py's theme block uses this
# matrix to recognize thematically-close films that share no facet string
# (e.g. "Gothic and eerie haunting horror" vs. "Terrifying, haunted, and
# supernatural horror", cosine ~0.79 -- see report.pdf for the full writeup).
def load_all_themes() -> list[str]:
    """The full known gsimonx37 theme vocabulary (109 distinct phrases),
    independent of which films are in any project's catalog -- so the theme
    embedding vocab is stable even if the catalog changes."""
    themes = pd.read_csv(GSIMONX37_DIR / "themes.csv")
    return sorted(themes["theme"].unique())


@dataclass
class ThemeSimilarity:
    vocab: dict            # theme string -> id
    matrix: np.ndarray      # [n_themes, n_themes] float32 cosine similarity


def build_theme_similarity(theme_strings: list[str],
                           model_name: str = THEME_EMBED_MODEL) -> ThemeSimilarity:
    """Encode every distinct theme phrase once and return the id vocab plus
    the dense cosine similarity matrix. Runs in seconds on CPU for ~109
    themes; not called per-movie or per-episode."""
    from sentence_transformers import SentenceTransformer

    vocab = {t: i for i, t in enumerate(sorted(theme_strings))}
    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(vocab), normalize_embeddings=True, show_progress_bar=False)
    matrix = (embeddings @ embeddings.T).astype(np.float32)
    return ThemeSimilarity(vocab, matrix)


def load_or_build_theme_similarity(theme_strings: list[str], force: bool = False) -> ThemeSimilarity:
    """Cache `build_theme_similarity` to disk (a tiny .npz -- ~109x109 floats
    plus the vocab list). Delete the cache or pass `force=True` if the theme
    vocabulary changes (it hasn't since the gsimonx37 dump was last updated)."""
    if not force and THEME_SIM_CACHE.exists():
        with np.load(THEME_SIM_CACHE, allow_pickle=False) as data:
            vocab = {t: i for i, t in enumerate(data["themes"])}
            return ThemeSimilarity(vocab, data["matrix"])
    sim = build_theme_similarity(theme_strings)
    THEME_SIM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    themes_sorted = sorted(sim.vocab, key=lambda t: sim.vocab[t])
    np.savez(THEME_SIM_CACHE, themes=np.array(themes_sorted), matrix=sim.matrix)
    return sim


# ---- "movies like this one" suggestions (K-means, app catalog only) --------
# Offline, deterministic precomputation for the app's suggestion dropdown ("to
# improve this prediction, rate one of these similar films"): cluster the
# app's ~1,000-film catalog on content facets with K-means, then for each
# film return its k_neighbors nearest neighbours within its own cluster.
_TOP_DIRECTOR_K = 30
_TOP_ACTOR_K = 30
_TOP_THEME_K = 30
_TOP_LANGUAGE_K = 15
_SIMILARITY_FACETS = {
    "director": _TOP_DIRECTOR_K, "actor": _TOP_ACTOR_K,
    "theme": _TOP_THEME_K, "language": _TOP_LANGUAGE_K,
}


def _facet_vocab(movies_json: list[dict], facet: str, k: int) -> dict:
    """Catalog-scoped top-k vocabulary (by frequency) for one facet -- this is
    intentionally a *different* vocab from the trained models' genre one
    (movie_features.GENRE_VOCAB_K etc): it only needs to distinguish
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
    genre multi-hot (the trained models' fixed canonical vocab, from
    ``genreMh``), director/actor/theme/language multi-hots (a catalog-scoped
    top-k vocabulary per facet, so cast and crew are explicitly part of the
    similarity, not just genre), and standardized numerics (runtime, the
    gsimonx37 site rating, facet counts, release year, and ``consensus`` --
    the one signal not already in ``movies_json``: Tomatometer-mapped score
    for Rotten Tomatoes, mean member rating for Letterboxd, keyed by movie
    id). Decade/country/studio dropped along with the feature contract (see
    module docstring); ``year`` already carries decade-scale information.

    Clusters with K-means (``k = max(2, n // 25)`` clusters, ~25 films each)
    and returns each film's ``k_neighbors`` nearest neighbours (Euclidean, on
    the standardized vector) within its own cluster; a cluster smaller than
    ``k_neighbors`` is topped up from the globally nearest remaining films so
    every movie always gets a full list. Deterministic for a fixed ``seed``.
    """
    n = len(movies_json)
    vocabs = {f: _facet_vocab(movies_json, f, k) for f, k in _SIMILARITY_FACETS.items()}
    genre_width = GENRE_VOCAB_K + 1
    genre_ids = {i: i for i in range(genre_width)}

    rows = []
    for m in movies_json:
        genre_row = _multihot(m["genreMh"], genre_ids, genre_width)
        facet_rows = [_multihot(m["facets"].get(f, []), vocabs[f], k)
                     for f, k in _SIMILARITY_FACETS.items()]
        numeric = np.array([
            m.get("runtimeLog") or 0.0,
            m.get("gsRating") or 0.0,
            m.get("nThemesLog") or 0.0,
            m.get("nLanguagesLog") or 0.0,
            float(m.get("year") or 0),
            float(consensus.get(m["id"]) or 0.0),
        ], dtype=np.float64)
        rows.append(np.concatenate([genre_row, *facet_rows, numeric]))

    # scikit-learn does both steps: StandardScaler centres/scales each column
    # (constant columns are left alone, as the old manual guard did), and
    # KMeans clusters with k-means++ init.
    data = StandardScaler().fit_transform(np.vstack(rows))

    k = max(2, min(n - 1, n // 25)) if n > 2 else 1
    labels = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=seed).fit_predict(data)

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
