// Letterboxd types. XgbModel/NNModel and their pieces are dataset-agnostic
// (feature-vector driven), so they're reused directly from the shared types
// module rather than redefined here.
import type { XgbModel, NNModel } from "../types";
export type { XgbModel, NNModel, XgbTree, NNLayer, NNMeta } from "../types";

/** Per-facet affinity sets (raw string values, gsimonx37 join), compared by
 * string-overlap in features.ts. Identical shape to the RT MovieFacets. */
export interface MovieFacets {
  genre: string[];
  decade: string[];
  theme: string[];
  language: string[];
  country: string[];
  studio: string[];
  director: string[];
  actor: string[];
}

export interface Movie {
  id: string;
  title: string;
  year: number | null;
  genreId: number;
  nScores: number;
  facets: MovieFacets;
  /** Fixed-vocab multi-hot ids (genre: 0..30, decade: 0..20). */
  genreMh: number[];
  decadeMh: number[];
  runtimeLog: number | null;
  gsRating: number | null;
  nThemesLog: number;
  nLanguagesLog: number;
  nCountriesLog: number;
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
  /** "Movies like this one" -- 10 nearest-neighbour catalog positions per
   * movie (K-means on content facets), position-aligned to `movies`. Used by
   * the predict-row suggestion dropdown. */
  similar: number[][];
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
