import type { Catalog, Models } from "./types";
import { computeMatches } from "./matches";
import { matchStats, buildFeatureRow, buildFeatureRowZ } from "./features";
import { predictAnalytic, predictAnalyticZ, predictAnalyticTop10, predictAnalyticTop10Z } from "./design1";
import { predictXgb } from "./xgboost";
import { predictNeuralNet } from "./neuralnet";
import { userStats, convertBack } from "./zscore";

const RAW_RANGE: [number, number] = [0, 5];
const Z_RANGE: [number, number] = [-Infinity, Infinity];

export interface Prediction {
  movieIdx: number;
  analytic: number;
  analyticTop10: number;
  movieMean: number;
  xgboost: number;
  neuralNet: number;
  tomatometer: number | null;
  /** z-score track: each rater standardized to their own scale, converted
   * back to the raw scale. Null when it can't be computed (the user's seen
   * ratings have ~zero variance, or -- per model -- no usable z peers/model). */
  analyticZ: number | null;
  analyticTop10Z: number | null;
  xgboostZ: number | null;
  neuralNetZ: number | null;
}

export function predictAll(
  catalog: Catalog,
  models: Models,
  seenRatings: Map<number, number>,
  targetMovieIdxs: number[]
): Prediction[] {
  const matches = computeMatches(catalog, seenRatings, models.kShrink);
  const stats = matchStats(matches);
  const userCount = seenRatings.size;
  let userSum = 0;
  for (const v of seenRatings.values()) userSum += v;
  const userMean = userCount ? userSum / userCount : 0;

  const user = userStats(seenRatings);
  const canZ = user.sigma > 1e-9;
  const matchesZ = canZ ? computeMatches(catalog, seenRatings, models.kShrink, user) : null;

  return targetMovieIdxs.map((movieIdx) => {
    const { prediction: analytic, movieMean } = predictAnalytic(catalog, matches, movieIdx);
    const { prediction: analyticTop10 } = predictAnalyticTop10(catalog, matches, movieIdx);
    const row = buildFeatureRow(catalog, matches, stats, movieIdx, userCount, userMean, seenRatings);
    const xgboost = predictXgb(models.xgb, row);
    const neuralNet = predictNeuralNet(models.nn, row);

    let analyticZ: number | null = null;
    let analyticTop10Z: number | null = null;
    let xgboostZ: number | null = null;
    let neuralNetZ: number | null = null;
    if (canZ && matchesZ) {
      const az = predictAnalyticZ(catalog, matchesZ, movieIdx, user, RAW_RANGE);
      analyticZ = az?.predictionRaw ?? null;
      const azTop10 = predictAnalyticTop10Z(catalog, matchesZ, movieIdx, user, RAW_RANGE);
      analyticTop10Z = azTop10?.predictionRaw ?? null;
      const rowZ = buildFeatureRowZ(catalog, matchesZ, stats, movieIdx, userCount, seenRatings, user);
      if (rowZ) {
        if (models.xgbZ) xgboostZ = convertBack(predictXgb(models.xgbZ, rowZ, Z_RANGE), user, RAW_RANGE);
        if (models.nnZ) neuralNetZ = convertBack(predictNeuralNet(models.nnZ, rowZ, Z_RANGE), user, RAW_RANGE);
      }
    }

    return {
      movieIdx, analytic, analyticTop10, movieMean, xgboost, neuralNet,
      tomatometer: catalog.movies[movieIdx].tomatoMeter,
      analyticZ, analyticTop10Z, xgboostZ, neuralNetZ,
    };
  });
}

export interface TopMatch {
  criticId: string;
  publicationName: string;
  sim: number;
  magSim: number;
  overlap: number;
}

export function topCriticMatches(
  catalog: Catalog,
  seenRatings: Map<number, number>,
  kShrink: number,
  minOverlap = 2,
  limit = 12
): TopMatch[] {
  const matches = computeMatches(catalog, seenRatings, kShrink);
  const rows: TopMatch[] = [];
  for (const [cIdx, m] of matches) {
    if (m.overlap >= minOverlap && m.sim !== 0) {
      const c = catalog.critics[cIdx];
      rows.push({ criticId: c.id, publicationName: c.publicationName, sim: m.sim, magSim: m.magSim, overlap: m.overlap });
    }
  }
  rows.sort((a, b) => b.sim - a.sim);
  return rows.slice(0, limit);
}

export function mse(pred: (number | null | undefined)[], truth: (number | null | undefined)[]): number | null {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i];
    const t = truth[i];
    if (p == null || t == null || Number.isNaN(p) || Number.isNaN(t)) continue;
    sum += (p - t) ** 2;
    n++;
  }
  return n ? sum / n : null;
}
