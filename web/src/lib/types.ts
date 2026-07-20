export interface Movie {
  id: string;
  title: string;
  year: number | null;
  tomatoMeter: number | null;
  genreId: number;
  tomatometerScore: number | null;
  nScores: number;
}

export interface Critic {
  id: string;
  publicationName: string;
  scoreCount: number;
  scoreSum: number;
}

export interface MovieRows {
  criticIdx: Uint16Array;
  score: Float32Array;
}

export interface Catalog {
  movies: Movie[];
  critics: Critic[];
  byMovie: MovieRows[];
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
  genreCol: string;
  muImpute: number[];
  mu: number[];
  sd: number[];
  nGenres: number;
  embDim: number;
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
}

export interface Models {
  xgb: XgbModel;
  nn: NNModel;
  kShrink: number;
}
