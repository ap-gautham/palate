/** Per-facet affinity sets (raw string values, gsimonx37 join), compared by
 * string-overlap in features.ts -- mirrors the Python frozenset intersection
 * in rotten_tomatoes/features.py's affinity blocks. Decade/country/studio were
 * dropped from the feature contract (decade duplicates `year`; country/studio
 * were low-signal) -- see report.pdf's feature-engineering section. */
export interface MovieFacets {
  genre: string[];
  theme: string[];
  language: string[];
  director: string[];
  actor: string[];
}

export interface Movie {
  id: string;
  title: string;
  year: number | null;
  tomatoMeter: number | null;
  tomatometerScore: number | null;
  nScores: number;
  /** Rich movie-facet payload (genre/theme/language/director/actor) -- see
   * `MovieFacets` in this file and `_SIMILARITY_FACETS`/`FACETS` in
   * movie_features.py. */
  facets: MovieFacets;
  /** Fixed canonical-vocab genre multi-hot ids (0..19 = 19 named genres +
   * "__other__") -- same training vocab as movie_features.GENRE_VOCAB_K. */
  genreMh: number[];
  /** Ids into `Catalog.themeMatrix`/`themeVocab` (theme_similarity.json) --
   * used by features.ts's theme block instead of re-deriving from strings. */
  themeIds: number[];
  runtimeLog: number | null;
  gsRating: number | null;
  nThemesLog: number;
  nLanguagesLog: number;
}

export interface Critic {
  id: string;
  publicationName: string;
  scoreCount: number;
  scoreSum: number;
  /** All-time sample std of this critic's scores, or null if undefined
   * (a single review, or a perfectly constant critic) -- the z-score track
   * skips peer contributions with no sigma. */
  scoreSigma: number | null;
}

export interface MovieRows {
  criticIdx: Uint16Array;
  score: Float32Array;
}

export interface Catalog {
  movies: Movie[];
  critics: Critic[];
  byMovie: MovieRows[];
  /** "Movies like this one" -- 20 nearest-neighbour catalog positions per
   * movie (K-means on content facets), position-aligned to `movies`. Used by
   * the predict-row suggestion dropdown. */
  similar: number[][];
  /** Theme embedding cosine similarity, flattened row-major
   * [themeVocab.length x themeVocab.length] (theme_similarity.json) -- see
   * movie_features.build_theme_similarity. */
  themeMatrix: Float32Array;
  themeVocab: string[];
  /** sigma_u fallback for the affinity blocks' z-scores (features.ts) when a
   * user's own seen-set std is ~0 -- see k_shrink.json / features.py's
   * `_facet_tail`. */
  globalStd: number;
}

export interface XgbTree {
  left: Int32Array;
  right: Int32Array;
  splitIdx: Int32Array;
  splitCond: Float64Array;
  defaultLeft: Int8Array;
  leafValue: Float64Array;
}

export interface XgbModel {
  baseScore: number;
  featureColumns: string[];
  trees: XgbTree[];
}

export interface NNLayer {
  name: string;
  shape: number[];
  offset: number;
  size: number;
}

export interface NNMeta {
  numericCols: string[];
  logCols: string[];
  muImpute: number[];
  mu: number[];
  sd: number[];
  width: number;
  depth: number;
  ensembleSize: number;
  layers: NNLayer[];
  totalParams: number;
}

export interface NNModel {
  meta: NNMeta;
  members: Float32Array[];
  layerOffset: Map<string, { offset: number; size: number; shape: number[] }>;
}

export interface CriticMatch {
  overlap: number;
  sim: number;
  magSim: number;
  /** Magnitude multiplier in z-space (peer standardized by all-time sigma).
   * Undefined when the critic's sigma is unavailable (see Critic.scoreSigma). */
  magSimZ?: number;
}

export interface Models {
  xgb: XgbModel;
  nn: NNModel;
  kShrink: number;
  /** z-score-track models: same architecture, trained on (raw - mu_user) /
   * sigma_user targets; predictions are converted back with zscore.ts. */
  xgbZ?: XgbModel;
  nnZ?: NNModel;
}
