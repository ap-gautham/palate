import type { Catalog, MemberMatch } from "./types";

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
 * :similarity (the same formula as Rotten Tomatoes' matches.ts). */
export function computeMatches(
  catalog: Catalog,
  userRatings: Map<number, number>,
  kShrink: number
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
    if (n >= MIN_OVERLAP) {
      const meanU = a.sumUser / n;
      const meanM = a.sumMember / n;
      const varU = a.sumUser2 / n - meanU * meanU;
      const varM = a.sumMember2 / n - meanM * meanM;
      if (varU > 1e-9 && varM > 1e-9) {
        const cov = a.sumUserMember / n - meanU * meanM;
        pearson = cov / Math.sqrt(varU * varM);
        if (a.sumMember2 > 1e-12) {
          magSim = a.sumUserMember / a.sumMember2;
        }
      }
    }
    const sim = pearson * (Math.min(n, kShrink) / kShrink);
    out.set(mIdx, { overlap: n, sim, magSim });
  }
  return out;
}
