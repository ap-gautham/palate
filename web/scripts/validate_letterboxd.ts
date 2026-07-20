// Standalone (non-Vite) sanity check for the Letterboxd port: builds the same
// Catalog/Models structs as src/lib/letterboxd/data.ts but reads from disk
// with fs instead of fetch, then runs a synthetic profile through predictAll
// and prints JSON for comparison against the Python letterboxd.analyze path.
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Catalog, Movie, Member, MovieRows, XgbModel, NNMeta, NNModel, Models } from "../src/lib/letterboxd/types";
import { predictAll, topMemberMatches } from "../src/lib/letterboxd/predict";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const DATA = join(SCRIPT_DIR, "..", "public", "data", "letterboxd");
const readJson = <T,>(name: string): T => JSON.parse(readFileSync(join(DATA, name), "utf-8"));
const readBuf = (name: string): ArrayBuffer => {
  const b = readFileSync(join(DATA, name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

function loadCatalog(): Catalog {
  const movies = readJson<Movie[]>("movies.json");
  const members = readJson<Member[]>("members.json");
  const memberIdx = new Uint16Array(readBuf("ratings_member_idx.bin"));
  const movieIdx = new Uint16Array(readBuf("ratings_movie_idx.bin"));
  const score = new Float32Array(readBuf("ratings_score.bin"));

  const counts = new Int32Array(movies.length);
  for (let i = 0; i < movieIdx.length; i++) counts[movieIdx[i]]++;
  const byMovie: MovieRows[] = movies.map((_, i) => ({
    memberIdx: new Uint16Array(counts[i]),
    score: new Float32Array(counts[i]),
  }));
  const cursor = new Int32Array(movies.length);
  for (let i = 0; i < movieIdx.length; i++) {
    const m = movieIdx[i];
    const c = cursor[m]++;
    byMovie[m].memberIdx[c] = memberIdx[i];
    byMovie[m].score[c] = score[i];
  }
  return { movies, members, byMovie };
}

function loadModels(): Models {
  const xgbRaw = readJson<any>("xgb_model.json");
  const nnMeta = readJson<NNMeta>("nn_meta.json");
  const meta = readJson<{ kShrink: number; ratingMin: number; ratingMax: number }>("meta.json");

  const xgb: XgbModel = {
    baseScore: xgbRaw.baseScore,
    featureColumns: xgbRaw.featureColumns,
    trees: xgbRaw.trees.map((t: any) => ({
      left: Int32Array.from(t.left),
      right: Int32Array.from(t.right),
      splitIdx: Int32Array.from(t.splitIdx),
      splitCond: Float64Array.from(t.splitCond),
      defaultLeft: Int8Array.from(t.defaultLeft),
      leafValue: Float64Array.from(t.leafValue),
    })),
  };
  const layerOffset = new Map(nnMeta.layers.map((l) => [l.name, l]));
  const members = Array.from({ length: nnMeta.ensembleSize }, (_, m) => new Float32Array(readBuf(`nn_weights_member${m}.bin`)));
  const nn: NNModel = { meta: nnMeta, members, layerOffset };
  return { xgb, nn, kShrink: meta.kShrink, ratingMin: meta.ratingMin, ratingMax: meta.ratingMax };
}

const catalog = loadCatalog();
const models = loadModels();

const args = process.argv.slice(2);
const seedArg = args.find((a) => a.startsWith("--seed="));
const seed = seedArg ? Number(seedArg.split("=")[1]) : 0;

function pseudoRandomProfile(n: number, seedN: number): Map<number, number> {
  let state = seedN || 42;
  const rnd = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  const seen = new Map<number, number>();
  const nMovies = catalog.movies.length;
  while (seen.size < n) {
    const idx = Math.floor(rnd() * nMovies);
    if (!seen.has(idx)) seen.set(idx, 1 + Math.floor(rnd() * 10));
  }
  return seen;
}

const seen = pseudoRandomProfile(15, seed);

const seenIds = Array.from(seen.entries()).map(([idx, rating]) => ({
  movie_id: catalog.movies[idx].id, rating,
}));

const targetIdxs = Array.from({ length: catalog.movies.length }, (_, i) => i)
  .filter((i) => !seen.has(i))
  .slice(0, 25);

const predictions = predictAll(catalog, models, seen, targetIdxs);
const matches = topMemberMatches(catalog, seen, models.kShrink);

console.log(JSON.stringify({
  seen: seenIds,
  target_ids: targetIdxs.map((i) => catalog.movies[i].id),
  predictions: predictions.map((p) => ({
    movie_id: catalog.movies[p.movieIdx].id,
    analytic: p.analytic, movie_mean: p.movieMean,
    xgboost: p.xgboost, neural_net: p.neuralNet,
  })),
  top_matches: matches.slice(0, 5).map((m) => ({ member_id: m.memberId, sim: m.sim, mag_sim: m.magSim, overlap: m.overlap })),
}, null, 2));
