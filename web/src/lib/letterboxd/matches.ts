import type { Catalog, MemberMatch } from "./types";
import type { UserStats } from "../zscore";

const MIN_OVERLAP = 2;

interface Accumulator {
  n: number;
  sumUser: number;
  sumMember: number;
  sumUser2: number;
  sumMember2: number;
  sumUserMember: number;
}

/** Shrunk Pearson alignment + magnitude scale per member, from the user's
 * seen-film ratings. Mirrors letterboxd/features.py:app_similarity /
 * :similarity (the same formula as Rotten Tomatoes' matches.ts).
 *
 * Pearson correlation is invariant to each side's own affine (mean/scale)
 * transform, so `sim` is identical whether computed in raw or z units --
 * only the magnitude multiplier differs. When `userZStats` is given (the
 * user's own seen-set mean/std), this also derives `magSimZ` -- the z-space
 * magnitude against each member's all-time z (from `ratingSum`/`ratingCount`/
 * `ratingSigma`) -- from the SAME accumulated sums, without a second pass. */
export function computeMatches(
  catalog: Catalog,
  userRatings: Map<number, number>,
  kShrink: number,
  userZStats?: UserStats
): Map<number, MemberMatch> {
  const acc = new Map<number, Accumulator>();
  for (const [movieIdx, userScore] of userRatings) {
    const rows = catalog.byMovie[movieIdx];
    for (let i = 0; i < rows.memberIdx.length; i++) {
      const mIdx = rows.memberIdx[i];
      const mScore = rows.score[i];
      let a = acc.get(mIdx);
      if (!a) {
        a = { n: 0, sumUser: 0, sumMember: 0, sumUser2: 0, sumMember2: 0, sumUserMember: 0 };
        acc.set(mIdx, a);
      }
      a.n++;
      a.sumUser += userScore;
      a.sumMember += mScore;
      a.sumUser2 += userScore * userScore;
      a.sumMember2 += mScore * mScore;
      a.sumUserMember += userScore * mScore;
    }
  }

  const out = new Map<number, MemberMatch>();
  for (const [mIdx, a] of acc) {
    const n = a.n;
    let pearson = 0;
    let magSim = 1;
    let eligible = false;
    if (n >= MIN_OVERLAP) {
      const meanU = a.sumUser / n;
      const meanM = a.sumMember / n;
      const varU = a.sumUser2 / n - meanU * meanU;
      const varM = a.sumMember2 / n - meanM * meanM;
      if (varU > 1e-9 && varM > 1e-9) {
        eligible = true;
        const cov = a.sumUserMember / n - meanU * meanM;
        pearson = cov / Math.sqrt(varU * varM);
        if (a.sumMember2 > 1e-12) {
          magSim = a.sumUserMember / a.sumMember2;
        }
      }
    }
    const sim = pearson * (Math.min(n, kShrink) / kShrink);
    const match: MemberMatch = { overlap: n, sim, magSim };

    if (userZStats && userZStats.sigma > 1e-9 && eligible) {
      const member = catalog.members[mIdx];
      const memberSigma = member.ratingSigma;
      if (memberSigma != null && memberSigma > 1e-9) {
        const memberMu = member.ratingSum / member.ratingCount;
        const numer = a.sumUserMember - memberMu * a.sumUser - userZStats.mu * a.sumMember
          + n * userZStats.mu * memberMu;
        const denom = a.sumMember2 - 2 * memberMu * a.sumMember + n * memberMu * memberMu;
        if (denom > 1e-12) {
          match.magSimZ = (memberSigma * numer) / denom;
        }
      }
    }
    out.set(mIdx, match);
  }
  return out;
}
