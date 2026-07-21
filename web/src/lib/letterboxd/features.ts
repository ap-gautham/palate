import type { Catalog, MemberMatch } from "./types";

const DECILES = 10;

/** Mirrors letterboxd/features.py:decile_features -- identical to the
 * Rotten Tomatoes decile summary (see the RT features.ts comment on the
 * numpy argsort tie-break note, which applies here too). */
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
export function matchStats(matches: Map<number, MemberMatch>): MatchStats {
  const overlaps = Array.from(matches.values()).map((m) => m.overlap).filter((n) => n > 0);
  const meanOverlap = overlaps.length ? overlaps.reduce((s, v) => s + v, 0) / overlaps.length : 0;
  const maxOverlap = overlaps.length ? Math.max(...overlaps) : 0;
  return { meanOverlap, maxOverlap };
}

/** One model-ready feature row (raw, unstandardized). Mirrors
 * letterboxd/features.py:app_features -- the RT contract without a
 * Tomatometer feature. */
export function buildFeatureRow(
  catalog: Catalog,
  matches: Map<number, MemberMatch>,
  stats: MatchStats,
  targetMovieIdx: number,
  userCount: number,
  userMean: number
): Record<string, number> {
  const rows = catalog.byMovie[targetMovieIdx];
  const members = catalog.members;
  const n = rows.memberIdx.length;

  const sims = new Float64Array(n);
  const deviations = new Float64Array(n);
  let sumValues = 0;
  for (let i = 0; i < n; i++) sumValues += rows.score[i];
  const groupMean = n ? sumValues / n : 0;

  for (let i = 0; i < n; i++) {
    const mIdx = rows.memberIdx[i];
    const value = rows.score[i];
    const member = members[mIdx];
    const peerCount = member.ratingCount - 1;
    const peerMean = peerCount > 0 ? (member.ratingSum - value) / peerCount : groupMean;
    deviations[i] = value - peerMean;
    sims[i] = matches.get(mIdx)?.sim ?? 0;
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
  row.n_reviewers = n;
  row.dispersion = dispersion;
  row.genre_id = movie.genreId;
  row.user_mean = userMean;
  return row;
}

/** z-space feature row: peers standardized by their all-time
 * ratingSum/ratingCount/ratingSigma (members without a defined sigma are
 * excluded entirely, mirroring the offline z-Split). Mirrors the RT
 * buildFeatureRowZ leave-one-out simplification: a member's own all-time
 * z-values sum to ~0, so their peer z-mean is `-z / (ratingCount - 1)`.
 * Returns null if no peer has a usable sigma. */
export function buildFeatureRowZ(
  catalog: Catalog,
  matches: Map<number, MemberMatch>,
  stats: MatchStats,
  targetMovieIdx: number,
  userCount: number
): Record<string, number> | null {
  const rows = catalog.byMovie[targetMovieIdx];
  const members = catalog.members;
  const total = rows.memberIdx.length;

  const sims: number[] = [];
  const deviations: number[] = [];
  const zValues: number[] = [];
  for (let i = 0; i < total; i++) {
    const mIdx = rows.memberIdx[i];
    const member = members[mIdx];
    const sigma = member.ratingSigma;
    if (sigma == null || sigma <= 1e-9 || member.ratingCount <= 1) continue;
    const mu = member.ratingSum / member.ratingCount;
    const z = (rows.score[i] - mu) / sigma;
    const peerCount = member.ratingCount - 1;
    const peerMeanZ = -z / peerCount; // member's own all-time z-sum is ~0
    zValues.push(z);
    deviations.push(z - peerMeanZ);
    sims.push(matches.get(mIdx)?.sim ?? 0);
  }
  const n = zValues.length;
  if (n === 0) return null;

  let dispersion = 0;
  if (n > 1) {
    const mean = zValues.reduce((s, v) => s + v, 0) / n;
    const sq = zValues.reduce((s, v) => s + (v - mean) ** 2, 0);
    dispersion = Math.sqrt(sq / (n - 1));
  }

  const dec = decileFeatures(Float64Array.from(sims), Float64Array.from(deviations));
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
  row.n_reviewers = n;
  row.dispersion = dispersion;
  row.genre_id = movie.genreId;
  row.user_mean = 0; // ~0 by construction -- the level we removed
  return row;
}
