// Letterboxd types. XgbModel/NNModel and their pieces are dataset-agnostic
// (feature-vector driven), so they're reused directly from the shared types
// module rather than redefined here.
import type { XgbModel, NNModel } from "../types";
export type { XgbModel, NNModel, XgbTree, NNLayer, NNMeta } from "../types";

/** Per-facet affinity sets (raw string values, gsimonx37 join), compared by
 * string-overlap in features.ts. Identical shape to the RT MovieFacets.
 * Decade/country/studio were dropped from the feature contract. */
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
  nScores: number;
  facets: MovieFacets;
  /** Fixed canonical-vocab genre multi-hot ids (0..19). */
  genreMh: number[];
  /** Ids into `Catalog.themeMatrix`/`themeVocab` (theme_similarity.json). */
  themeIds: number[];
  runtimeLog: number | null;
  gsRating: number | null;
  nThemesLog: number;
  nLanguagesLog: number;
}

export interface Member {
  id: string;
  ratingSum: number;
  ratingCount: number;
  /** All-time sample std of this member's ratings, or null if undefined --
   * the z-score track skips peer contributions with no sigma. */
  ratingSigma: number | null;
}

export interface MovieRows {
  memberIdx: Uint16Array;
  score: Float32Array;
}

export interface Catalog {
  movies: Movie[];
  members: Member[];
  byMovie: MovieRows[];
  /** "Movies like this one" -- 20 nearest-neighbour catalog positions per
   * movie (K-means on content facets), position-aligned to `movies`. Used by
   * the predict-row suggestion dropdown. */
  similar: number[][];
  /** Theme embedding cosine similarity, flattened row-major
   * [themeVocab.length x themeVocab.length] (theme_similarity.json). */
  themeMatrix: Float32Array;
  themeVocab: string[];
  /** sigma_u fallback for the affinity blocks' z-scores when a user's own
   * seen-set std is ~0 -- see meta.json / features.py's `_facet_tail`. */
  globalStd: number;
}

export interface MemberMatch {
  overlap: number;
  sim: number;
  magSim: number;
  /** Magnitude multiplier in z-space (peer standardized by all-time sigma).
   * Undefined when the member's sigma is unavailable. */
  magSimZ?: number;
}

export interface Models {
  xgb: XgbModel;
  nn: NNModel;
  kShrink: number;
  ratingMin: number;
  ratingMax: number;
  /** z-score-track models: same architecture, trained on (raw - mu_user) /
   * sigma_user targets; predictions are converted back with zscore.ts. */
  xgbZ?: XgbModel;
  nnZ?: NNModel;
}
