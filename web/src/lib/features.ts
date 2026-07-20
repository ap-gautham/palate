import type { Catalog, CriticMatch } from "./types";

const DECILES = 10;

/** Mirrors design2_xgboost/features.py:decile_features -- summarize
 * similarity x deviation in ten similarity-ranked deciles (numpy array_split
 * semantics: the first `n % k` chunks get one extra element). */
// Ties broken by original (stable) order; numpy's default quicksort argsort
// is unstable and can break ties differently, which very occasionally moves
// a critic across a decile boundary and perturbs XGBoost/NN output by up to
// ~0.02 on the 0-5 scale (~0.4%) vs the Python app -- confirmed negligible
// against the offline Python predict.py path in web/scripts/validate.ts.
function decileFeatures(sims: Float64Array, deviations: Float64Array): number[] {
  const n = sims.length;
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => sims[b] - sims[a]);
  const products = order.map((i) => sims[i] * deviations[i]);

  const out = new Array(3 * DECILES).fill(0);
  const base = Math.floor(n / DECILES);
  const rem = n % DECILES;
  let start = 0;
  for (let c = 0; c < DECILES; c++) {
    const size = base + (c < rem ? 1 : 0);
    const chunk = products.slice(start, start + size);
    if (chunk.length) {
      const mean = chunk.reduce((s, v) => s + v, 0) / chunk.length;
      const variance = chunk.reduce((s, v) => s + (v - mean) ** 2, 0) / chunk.length;
      out[c] = mean;
      out[DECILES + c] = chunk.length;
      out[2 * DECILES + c] = Math.sqrt(variance);
    }
    start += size;
  }
  return out;
}

export interface MatchStats {
  meanOverlap: number;
  maxOverlap: number;
}

/** mean_overlap / max_overlap depend only on the user's match table, not the
 * target movie -- computed once and reused across all target films. */
export function matchStats(matches: Map<number, CriticMatch>): MatchStats {
  const overlaps = Array.from(matches.values()).map((m) => m.overlap).filter((n) => n > 0);
  const meanOverlap = overlaps.length ? overlaps.reduce((s, v) => s + v, 0) / overlaps.length : 0;
  const maxOverlap = overlaps.length ? Math.max(...overlaps) : 0;
  return { meanOverlap, maxOverlap };
}

/** One model-ready feature row (raw, unstandardized), keyed by the same
 * column names both trained models expect. Mirrors
 * design2_xgboost/features.py:app_features (single-movie body). */
export function buildFeatureRow(
  catalog: Catalog,
  matches: Map<number, CriticMatch>,
  stats: MatchStats,
  targetMovieIdx: number,
  userCount: number,
  userMean: number
): Record<string, number> {
  const rows = catalog.byMovie[targetMovieIdx];
  const critics = catalog.critics;
  const n = rows.criticIdx.length;

  const sims = new Float64Array(n);
  const deviations = new Float64Array(n);
  let sumValues = 0;
  for (let i = 0; i < n; i++) sumValues += rows.score[i];
  const groupMean = n ? sumValues / n : 0;

  for (let i = 0; i < n; i++) {
    const cIdx = rows.criticIdx[i];
    const value = rows.score[i];
    const critic = critics[cIdx];
    const peerCount = critic.scoreCount - 1;
    const peerMean = peerCount > 0 ? (critic.scoreSum - value) / peerCount : groupMean;
    deviations[i] = value - peerMean;
    sims[i] = matches.get(cIdx)?.sim ?? 0;
  }

  let dispersion = 0;
  if (n > 1) {
    const mean = groupMean;
    const sq = rows.score.reduce((s, v) => s + (v - mean) ** 2, 0);
    dispersion = Math.sqrt(sq / (n - 1));
  }

  const dec = decileFeatures(sims, deviations);
  const row: Record<string, number> = {};
  for (let i = 0; i < DECILES; i++) {
    row[`d${i}_mean`] = dec[i];
    row[`d${i}_cnt`] = dec[DECILES + i];
    row[`d${i}_std`] = dec[2 * DECILES + i];
  }
  const movie = catalog.movies[targetMovieIdx];
  row.n_observed = userCount;
  row.mean_overlap = stats.meanOverlap;
  row.max_overlap = stats.maxOverlap;
  row.tomatometer = movie.tomatometerScore ?? NaN;
  row.n_reviewers = n;
  row.dispersion = dispersion;
  row.genre_id = movie.genreId;
  row.user_mean = userMean;
  return row;
}
