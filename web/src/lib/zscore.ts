// Shared z-score-track math, reused by both the Rotten Tomatoes and
// Letterboxd inference libraries. The user (a real visitor) is standardized
// by the mean/std of THEIR OWN rated "seen" films -- never a rater's all-time
// stats, since that's what makes this faithful to the offline evaluation
// (see rotten_tomatoes/pseudo_users.py's build_split docstring). Peers
// (critics/members) are standardized by their all-time mean/std, shipped in
// the catalog export as `scoreSigma`/`ratingSigma` alongside the existing
// count/sum fields.

export interface UserStats {
  mu: number;
  sigma: number;
}

/** Mean/std of the user's own ratings (population std, ddof=0). Returns
 * sigma=0 when the ratings have no variance -- callers should skip the
 * z-track prediction in that case (can't standardize), exactly like the
 * offline z-track episode generators. */
export function userStats(ratings: Map<number, number>): UserStats {
  const values = Array.from(ratings.values());
  const n = values.length;
  const mu = n ? values.reduce((s, v) => s + v, 0) / n : 0;
  const variance = n ? values.reduce((s, v) => s + (v - mu) ** 2, 0) / n : 0;
  return { mu, sigma: Math.sqrt(variance) };
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Converts a z-space prediction back to the raw scale using the user's own
 * seen-set mean/std, clipped to the dataset's rating range. */
export function convertBack(zPred: number, user: UserStats, range: [number, number]): number {
  return clamp(user.mu + user.sigma * zPred, range[0], range[1]);
}
