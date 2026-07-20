import type { Catalog, CriticMatch } from "./types";

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Movie-mean-centered, magnitude-scaled analytic prediction. Mirrors
 * design1_analytic/predict.py:predict. */
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
