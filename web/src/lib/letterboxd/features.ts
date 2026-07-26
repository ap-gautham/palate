import type { Catalog, MemberMatch } from "./types";

const DECILES = 10;

// Mirrors rotten_tomatoes/movie_features.py's GENRE_VOCAB_K (19 canonical
// genres + "__other__" = 20 slots); identical constant used by
// letterboxd/movie_features.py.
const GENRE_WIDTH = 20;

function zOrNan(sum: number, cnt: number, muU: number, sigmaU: number): number {
  return cnt > 0 ? (sum / cnt - muU) / sigmaU : NaN;
}

/** Per-genre z-score + log1p(count-including-target). Mirrors
 * features.py's `_genre_block` exactly: the count INCLUDES the target film
 * itself (0 -> target isn't this genre, 1 -> target is this genre and the
 * user has seen none, n+1 -> n seen), so it doubles as the old multi-hot
 * with no separate mh_genre_* block needed. */
function genreBlock(
  targetGenreIds: number[],
  seenGenreIds: number[][],
  seenValues: number[],
  muU: number,
  sigmaU: number
): Record<string, number> {
  const targetSet = new Set(targetGenreIds);
  const row: Record<string, number> = {};
  for (let g = 0; g < GENRE_WIDTH; g++) {
    let sum = 0;
    let cnt = 0;
    for (let i = 0; i < seenGenreIds.length; i++) {
      if (seenGenreIds[i].includes(g)) {
        sum += seenValues[i];
        cnt++;
      }
    }
    const cntIncl = cnt + (targetSet.has(g) ? 1 : 0);
    row[`user_genre_${g}_z`] = zOrNan(sum, cnt, muU, sigmaU);
    row[`user_genre_${g}_cnt`] = Math.log1p(cntIncl);
  }
  return row;
}

/** Similarity-weighted average RAW rating over (target theme, seen film)
 * pairs: w(t,f) = max_{u in seen film's themes} cos(t, u), read from the
 * embedding similarity matrix. Collapses to
 * Σ w(t,f)·rating(f) / Σ w(t,f) -- see features.py's `_theme_block`. */
function themeBlock(
  catalog: Catalog,
  targetThemeIds: number[],
  seenThemeIds: number[][],
  seenValues: number[]
): Record<string, number> {
  if (targetThemeIds.length === 0) {
    return { user_theme_avg: NaN, user_theme_mass_log: 0, user_theme_simcnt_hi: 0 };
  }
  const matrix = catalog.themeMatrix;
  const width = catalog.themeVocab.length;
  let totalNum = 0;
  let totalDen = 0;
  let hiCount = 0;
  for (let i = 0; i < seenThemeIds.length; i++) {
    const tf = seenThemeIds[i];
    if (tf.length === 0) continue;
    let wSum = 0;
    let wMax = 0;
    for (const t of targetThemeIds) {
      let best = 0;
      for (const u of tf) {
        const v = matrix[t * width + u];
        if (v > best) best = v;
      }
      best = Math.min(Math.max(best, 0), 1);
      wSum += best;
      if (best > wMax) wMax = best;
    }
    if (wSum <= 0) continue;
    totalNum += wSum * seenValues[i];
    totalDen += wSum;
    if (wMax > 0.8) hiCount++;
  }
  return {
    user_theme_avg: totalDen > 0 ? totalNum / totalDen : NaN,
    user_theme_mass_log: Math.log1p(totalDen),
    user_theme_simcnt_hi: Math.log1p(hiCount),
  };
}

interface ActorEntry {
  avg: number;
  cnt: number;
}

/** Three top-5 rankings of the target's cast against the user's seen
 * history: by mean rating, by viewing count, and seen films ranked by
 * shared-cast count. Mirrors features.py's `_actor_block`. */
function actorBlock(
  targetCast: string[],
  seenCastList: string[][],
  seenValues: number[]
): Record<string, number> {
  const targetSet = new Set(targetCast);
  const ratingsByActor = new Map<string, number[]>();
  for (let i = 0; i < seenCastList.length; i++) {
    const cast = seenCastList[i];
    if (cast.length === 0) continue;
    for (const a of cast) {
      if (!targetSet.has(a)) continue;
      let arr = ratingsByActor.get(a);
      if (!arr) { arr = []; ratingsByActor.set(a, arr); }
      arr.push(seenValues[i]);
    }
  }
  const entries: ActorEntry[] = [];
  for (const rs of ratingsByActor.values()) {
    const avg = rs.reduce((s, v) => s + v, 0) / rs.length;
    entries.push({ avg, cnt: rs.length });
  }
  const byRating = [...entries].sort((a, b) => (b.avg - a.avg) || (b.cnt - a.cnt)).slice(0, 5);
  const byCount = [...entries].sort((a, b) => (b.cnt - a.cnt) || (b.avg - a.avg)).slice(0, 5);

  const row: Record<string, number> = {};
  for (let i = 0; i < 5; i++) {
    const zCol = `user_actor_byrating${i + 1}_z`;
    const cntCol = `user_actor_byrating${i + 1}_cnt`;
    if (i < byRating.length) {
      row[zCol] = NaN;
      row[cntCol] = Math.log1p(byRating[i].cnt);
      (row as any)[`__avg_byrating${i + 1}`] = byRating[i].avg;
    } else {
      row[zCol] = NaN;
      row[cntCol] = 0;
    }
  }
  for (let i = 0; i < 5; i++) {
    const zCol = `user_actor_bycount${i + 1}_z`;
    const cntCol = `user_actor_bycount${i + 1}_cnt`;
    if (i < byCount.length) {
      row[zCol] = NaN;
      row[cntCol] = Math.log1p(byCount[i].cnt);
      (row as any)[`__avg_bycount${i + 1}`] = byCount[i].avg;
    } else {
      row[zCol] = NaN;
      row[cntCol] = 0;
    }
  }

  const overlaps: { n: number; rating: number }[] = [];
  for (let i = 0; i < seenCastList.length; i++) {
    const cast = seenCastList[i];
    if (cast.length === 0) continue;
    let n = 0;
    for (const a of cast) if (targetSet.has(a)) n++;
    if (n > 0) overlaps.push({ n, rating: seenValues[i] });
  }
  overlaps.sort((a, b) => b.n - a.n);
  for (let i = 0; i < 5; i++) {
    const nCol = `user_cast_overlap${i + 1}_n`;
    const rCol = `user_cast_overlap${i + 1}_rating`;
    if (i < overlaps.length) {
      row[nCol] = Math.log1p(overlaps[i].n);
      row[rCol] = overlaps[i].rating;
    } else {
      row[nCol] = 0;
      row[rCol] = NaN;
    }
  }
  return row;
}

/** Fills in the actor z-scores that `actorBlock` deferred (it doesn't know
 * muU/sigmaU until the caller computes them) and strips the temporary
 * `__avg_*` carriers. */
function finishActorZ(row: Record<string, number>, muU: number, sigmaU: number): void {
  for (const kind of ["byrating", "bycount"]) {
    for (let i = 1; i <= 5; i++) {
      const carrier = `__avg_${kind}${i}`;
      if (carrier in row) {
        row[`user_actor_${kind}${i}_z`] = (row[carrier] - muU) / sigmaU;
        delete row[carrier];
      }
    }
  }
}

/** One z-score + log1p(count) over seen films sharing any director with the
 * target. Mirrors features.py's `_director_block`. */
function directorBlock(
  targetDirectors: string[],
  seenDirectorList: string[][],
  seenValues: number[],
  muU: number,
  sigmaU: number
): Record<string, number> {
  const targetSet = new Set(targetDirectors);
  let sum = 0;
  let cnt = 0;
  for (let i = 0; i < seenDirectorList.length; i++) {
    const ds = seenDirectorList[i];
    if (ds.length === 0) continue;
    if (ds.some((d) => targetSet.has(d))) {
      sum += seenValues[i];
      cnt++;
    }
  }
  return {
    user_director_z: zOrNan(sum, cnt, muU, sigmaU),
    user_director_cnt: Math.log1p(cnt),
  };
}

/** Core affinity-tail computation for one target movie, given the seen set's
 * RAW values (raw track: actual ratings; z track: the episode-standardized
 * z-values, already ~mean 0 std 1). Mirrors features.py's `_facet_tail`
 * (identical logic to the RT features.ts version). */
function affinityTail(
  catalog: Catalog,
  seenIdxs: number[],
  seenValues: number[],
  targetMovieIdx: number,
  globalStd: number
): Record<string, number> {
  const target = catalog.movies[targetMovieIdx];
  const seenMovies = seenIdxs.map((i) => catalog.movies[i]);

  let muU = 0;
  if (seenValues.length) muU = seenValues.reduce((s, v) => s + v, 0) / seenValues.length;
  let sigmaU = 0;
  if (seenValues.length) {
    const variance = seenValues.reduce((s, v) => s + (v - muU) ** 2, 0) / seenValues.length;
    sigmaU = Math.sqrt(variance);
  }
  if (sigmaU < 1e-9) sigmaU = globalStd > 1e-9 ? globalStd : 1.0;

  const seenGenreIds = seenMovies.map((m) => m.genreMh);
  const seenThemeIds = seenMovies.map((m) => m.themeIds);
  const seenActorList = seenMovies.map((m) => m.facets.actor);
  const seenDirectorList = seenMovies.map((m) => m.facets.director);

  const row: Record<string, number> = {};
  Object.assign(row, genreBlock(target.genreMh, seenGenreIds, seenValues, muU, sigmaU));
  Object.assign(row, themeBlock(catalog, target.themeIds, seenThemeIds, seenValues));
  const actorRow = actorBlock(target.facets.actor, seenActorList, seenValues);
  finishActorZ(actorRow, muU, sigmaU);
  Object.assign(row, actorRow);
  Object.assign(row, directorBlock(target.facets.director, seenDirectorList, seenValues, muU, sigmaU));
  row.runtime_log = target.runtimeLog ?? 0;
  row.gs_rating = target.gsRating ?? 0;
  row.n_themes_log = target.nThemesLog;
  row.n_languages_log = target.nLanguagesLog;
  return row;
}

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
  userMean: number,
  seenRatings: Map<number, number>
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
  row.n_observed = userCount;
  row.mean_overlap = stats.meanOverlap;
  row.max_overlap = stats.maxOverlap;
  row.n_reviewers = n;
  row.dispersion = dispersion;
  row.user_mean = userMean;
  const seenIdxs = Array.from(seenRatings.keys());
  const seenValues = seenIdxs.map((i) => seenRatings.get(i)!);
  Object.assign(row, affinityTail(catalog, seenIdxs, seenValues, targetMovieIdx, catalog.globalStd));
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
  userCount: number,
  seenRatings: Map<number, number>,
  user: { mu: number; sigma: number }
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
  row.n_observed = userCount;
  row.mean_overlap = stats.meanOverlap;
  row.max_overlap = stats.maxOverlap;
  row.n_reviewers = n;
  row.dispersion = dispersion;
  row.user_mean = 0; // ~0 by construction -- the level we removed
  const seenIdxs = Array.from(seenRatings.keys());
  // seenRatings holds RAW ratings; standardize to this episode's z-units --
  // already mean 0 std 1 by construction, so it doubles as the "raw value on
  // this track's scale" the affinity tail wants (see features.py comment).
  const seenZValues = seenIdxs.map((i) => (seenRatings.get(i)! - user.mu) / user.sigma);
  Object.assign(row, affinityTail(catalog, seenIdxs, seenZValues, targetMovieIdx, catalog.globalStd));
  return row;
}
