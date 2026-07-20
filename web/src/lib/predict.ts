import type { Catalog, Models } from "./types";
import { computeMatches } from "./matches";
import { matchStats, buildFeatureRow } from "./features";
import { predictAnalytic } from "./design1";
import { predictXgb } from "./xgboost";
import { predictNeuralNet } from "./neuralnet";

export interface Prediction {
  movieIdx: number;
  analytic: number;
  movieMean: number;
  xgboost: number;
  neuralNet: number;
  tomatometer: number | null;
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

  return targetMovieIdxs.map((movieIdx) => {
    const { prediction: analytic, movieMean } = predictAnalytic(catalog, matches, movieIdx);
    const row = buildFeatureRow(catalog, matches, stats, movieIdx, userCount, userMean);
    const xgboost = predictXgb(models.xgb, row);
    const neuralNet = predictNeuralNet(models.nn, row);
    return {
      movieIdx, analytic, movieMean, xgboost, neuralNet,
      tomatometer: catalog.movies[movieIdx].tomatoMeter,
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
