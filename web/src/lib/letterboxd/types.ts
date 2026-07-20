// Letterboxd types. XgbModel/NNModel and their pieces are dataset-agnostic
// (feature-vector driven), so they're reused directly from the shared types
// module rather than redefined here.
import type { XgbModel, NNModel } from "../types";
export type { XgbModel, NNModel, XgbTree, NNLayer, NNMeta } from "../types";

export interface Movie {
  id: string;
  title: string;
  year: number | null;
  genreId: number;
  nScores: number;
}

export interface Member {
  id: string;
  ratingSum: number;
  ratingCount: number;
}

export interface MovieRows {
  memberIdx: Uint16Array;
  score: Float32Array;
}

export interface Catalog {
  movies: Movie[];
  members: Member[];
  byMovie: MovieRows[];
}

export interface MemberMatch {
  overlap: number;
  sim: number;
  magSim: number;
}

export interface Models {
  xgb: XgbModel;
  nn: NNModel;
  kShrink: number;
  ratingMin: number;
  ratingMax: number;
}
