import type { Catalog, MemberMatch } from "./types";

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Movie-mean-centered, magnitude-scaled analytic prediction, clipped to the
 * Letterboxd 1-10 scale. Mirrors letterboxd/analyze.py's Design 1 formula
 * (the same math as Rotten Tomatoes' design1.ts). */
export function predictAnalytic(
  catalog: Catalog,
  matches: Map<number, MemberMatch>,
  targetMovieIdx: number,
  ratingMin: number,
  ratingMax: number
): { prediction: number; movieMean: number } {
  const rows = catalog.byMovie[targetMovieIdx];
  const n = rows.memberIdx.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += rows.score[i];
  const movieMean = n ? sum / n : 0;

  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const mIdx = rows.memberIdx[i];
    const score = rows.score[i];
    const m = matches.get(mIdx);
    const sim = m?.sim ?? 0;
    const magSim = m?.magSim ?? 1;
    num += (Math.abs(sim) * movieMean + sim * (score - movieMean)) * magSim;
    den += Math.abs(sim);
  }
  const prediction = den > 0 ? num / den : movieMean;
  return { prediction: clamp(prediction, ratingMin, ratingMax), movieMean };
}
