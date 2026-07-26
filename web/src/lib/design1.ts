import type { Catalog, CriticMatch } from "./types";
import type { UserStats } from "./zscore";
import { convertBack } from "./zscore";

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Movie-mean-centered, magnitude-scaled analytic prediction. Mirrors
 * predict_analytic.py:predict. */
export function predictAnalytic(
  catalog: Catalog,
  matches: Map<number, CriticMatch>,
  targetMovieIdx: number
): { prediction: number; movieMean: number } {
  const rows = catalog.byMovie[targetMovieIdx];
  const n = rows.criticIdx.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += rows.score[i];
  const movieMean = n ? sum / n : 0;

  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const cIdx = rows.criticIdx[i];
    const score = rows.score[i];
    const m = matches.get(cIdx);
    const sim = m?.sim ?? 0;
    const magSim = m?.magSim ?? 1;
    num += (Math.abs(sim) * movieMean + sim * (score - movieMean)) * magSim;
    den += Math.abs(sim);
  }
  const prediction = den > 0 ? num / den : movieMean;
  return { prediction: clamp(prediction, 0, 5), movieMean };
}

/** z-space analytic prediction, converted back to the raw scale using the
 * user's own seen-set mean/std. Peers without a defined all-time sigma are
 * excluded (mirrors buildFeatureRowZ / the offline z-Split). Returns null if
 * no peer has a usable sigma. */
export function predictAnalyticZ(
  catalog: Catalog,
  matches: Map<number, CriticMatch>,
  targetMovieIdx: number,
  user: UserStats,
  range: [number, number]
): { predictionRaw: number; movieMeanZ: number } | null {
  const rows = catalog.byMovie[targetMovieIdx];
  const critics = catalog.critics;
  const total = rows.criticIdx.length;

  const zValues: number[] = [];
  const cIdxs: number[] = [];
  for (let i = 0; i < total; i++) {
    const cIdx = rows.criticIdx[i];
    const critic = critics[cIdx];
    const sigma = critic.scoreSigma;
    if (sigma == null || sigma <= 1e-9 || critic.scoreCount <= 1) continue;
    const mu = critic.scoreSum / critic.scoreCount;
    zValues.push((rows.score[i] - mu) / sigma);
    cIdxs.push(cIdx);
  }
  const n = zValues.length;
  if (n === 0) return null;
  const movieMeanZ = zValues.reduce((s, v) => s + v, 0) / n;

  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const cIdx = cIdxs[i];
    const zScore = zValues[i];
    const m = matches.get(cIdx);
    const sim = m?.sim ?? 0;
    const magSimZ = m?.magSimZ ?? 1;
    num += (Math.abs(sim) * movieMeanZ + sim * (zScore - movieMeanZ)) * magSimZ;
    den += Math.abs(sim);
  }
  const predictionZ = den > 0 ? num / den : movieMeanZ;
  return { predictionRaw: convertBack(predictionZ, user, range), movieMeanZ };
}

const TOPK_ABS = 10;

/** A new variation of Design 1 (the full-neighbourhood formula above is
 * unchanged): restrict the neighbourhood to the `k` critics with the largest
 * |sim| -- both strongly aligned and strongly anti-aligned, not just
 * positively-aligned -- then the identical movie-mean-centered,
 * magnitude-scaled formula. Mirrors
 * analytic.py:predict_movie_topk_abs. */
export function predictAnalyticTop10(
  catalog: Catalog,
  matches: Map<number, CriticMatch>,
  targetMovieIdx: number
): { prediction: number; movieMean: number } {
  const rows = catalog.byMovie[targetMovieIdx];
  const n = rows.criticIdx.length;
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => Math.abs(matches.get(rows.criticIdx[b])?.sim ?? 0)
                     - Math.abs(matches.get(rows.criticIdx[a])?.sim ?? 0));
  const top = order.slice(0, TOPK_ABS);
  let sum = 0;
  for (const i of top) sum += rows.score[i];
  const movieMean = top.length ? sum / top.length : 0;

  let num = 0;
  let den = 0;
  for (const i of top) {
    const cIdx = rows.criticIdx[i];
    const score = rows.score[i];
    const m = matches.get(cIdx);
    const sim = m?.sim ?? 0;
    const magSim = m?.magSim ?? 1;
    num += (Math.abs(sim) * movieMean + sim * (score - movieMean)) * magSim;
    den += Math.abs(sim);
  }
  const prediction = den > 0 ? num / den : movieMean;
  return { prediction: clamp(prediction, 0, 5), movieMean };
}

/** z-space top-|sim| variant, converted back to the raw scale. UNCLIPPED
 * until after the mu + sigma * z conversion, exactly like `predictAnalyticZ`. */
export function predictAnalyticTop10Z(
  catalog: Catalog,
  matches: Map<number, CriticMatch>,
  targetMovieIdx: number,
  user: UserStats,
  range: [number, number]
): { predictionRaw: number; movieMeanZ: number } | null {
  const rows = catalog.byMovie[targetMovieIdx];
  const critics = catalog.critics;
  const total = rows.criticIdx.length;

  const zValues: number[] = [];
  const cIdxs: number[] = [];
  for (let i = 0; i < total; i++) {
    const cIdx = rows.criticIdx[i];
    const critic = critics[cIdx];
    const sigma = critic.scoreSigma;
    if (sigma == null || sigma <= 1e-9 || critic.scoreCount <= 1) continue;
    const mu = critic.scoreSum / critic.scoreCount;
    zValues.push((rows.score[i] - mu) / sigma);
    cIdxs.push(cIdx);
  }
  const n = zValues.length;
  if (n === 0) return null;

  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => Math.abs(matches.get(cIdxs[b])?.sim ?? 0) - Math.abs(matches.get(cIdxs[a])?.sim ?? 0));
  const top = order.slice(0, TOPK_ABS);
  let sum = 0;
  for (const i of top) sum += zValues[i];
  const movieMeanZ = top.length ? sum / top.length : 0;

  let num = 0;
  let den = 0;
  for (const i of top) {
    const cIdx = cIdxs[i];
    const zScore = zValues[i];
    const m = matches.get(cIdx);
    const sim = m?.sim ?? 0;
    const magSimZ = m?.magSimZ ?? 1;
    num += (Math.abs(sim) * movieMeanZ + sim * (zScore - movieMeanZ)) * magSimZ;
    den += Math.abs(sim);
  }
  const predictionZ = den > 0 ? num / den : movieMeanZ;
  return { predictionRaw: convertBack(predictionZ, user, range), movieMeanZ };
}
