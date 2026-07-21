import type { Catalog, CriticMatch } from "./types";
import type { UserStats } from "./zscore";

const MIN_OVERLAP = 2;

interface Accumulator {
  n: number;
  sumUser: number;
  sumCritic: number;
  sumUser2: number;
  sumCritic2: number;
  sumUserCritic: number;
}

/** Shrunk Pearson alignment + magnitude scale per critic, from the user's
 * seen-film ratings. Mirrors design1_analytic/predict.py:critic_matches.
 *
 * Pearson correlation is invariant to each side's own affine (mean/scale)
 * transform, so `sim` is identical whether computed in raw or z units --
 * only the magnitude multiplier differs. When `userZStats` is given (the
 * user's own seen-set mean/std), this also derives `magSimZ` -- the z-space
 * magnitude against each critic's all-time z (from `scoreSum`/`scoreCount`/
 * `scoreSigma`) -- from the SAME accumulated sums, without a second pass. */
export function computeMatches(
  catalog: Catalog,
  userRatings: Map<number, number>,
  kShrink: number,
  userZStats?: UserStats
): Map<number, CriticMatch> {
  const acc = new Map<number, Accumulator>();
  for (const [movieIdx, userScore] of userRatings) {
    const rows = catalog.byMovie[movieIdx];
    for (let i = 0; i < rows.criticIdx.length; i++) {
      const cIdx = rows.criticIdx[i];
      const cScore = rows.score[i];
      let a = acc.get(cIdx);
      if (!a) {
        a = { n: 0, sumUser: 0, sumCritic: 0, sumUser2: 0, sumCritic2: 0, sumUserCritic: 0 };
        acc.set(cIdx, a);
      }
      a.n++;
      a.sumUser += userScore;
      a.sumCritic += cScore;
      a.sumUser2 += userScore * userScore;
      a.sumCritic2 += cScore * cScore;
      a.sumUserCritic += userScore * cScore;
    }
  }

  const out = new Map<number, CriticMatch>();
  for (const [cIdx, a] of acc) {
    const n = a.n;
    let pearson = 0;
    let magSim = 1;
    let eligible = false;
    if (n >= MIN_OVERLAP) {
      const meanU = a.sumUser / n;
      const meanC = a.sumCritic / n;
      const varU = a.sumUser2 / n - meanU * meanU;
      const varC = a.sumCritic2 / n - meanC * meanC;
      if (varU > 1e-9 && varC > 1e-9) {
        eligible = true;
        const cov = a.sumUserCritic / n - meanU * meanC;
        pearson = cov / Math.sqrt(varU * varC);
        if (a.sumCritic2 > 1e-12) {
          magSim = a.sumUserCritic / a.sumCritic2;
        }
      }
    }
    const sim = pearson * (Math.min(n, kShrink) / kShrink);
    const match: CriticMatch = { overlap: n, sim, magSim };

    if (userZStats && userZStats.sigma > 1e-9 && eligible) {
      const critic = catalog.critics[cIdx];
      const criticSigma = critic.scoreSigma;
      if (criticSigma != null && criticSigma > 1e-9) {
        const criticMu = critic.scoreSum / critic.scoreCount;
        const numer = a.sumUserCritic - criticMu * a.sumUser - userZStats.mu * a.sumCritic
          + n * userZStats.mu * criticMu;
        const denom = a.sumCritic2 - 2 * criticMu * a.sumCritic + n * criticMu * criticMu;
        if (denom > 1e-12) {
          match.magSimZ = (criticSigma * numer) / denom;
        }
      }
    }
    out.set(cIdx, match);
  }
  return out;
}
