// Standalone (non-Vite) sanity check: builds the same Catalog/Models structs
// as src/lib/data.ts but reads from disk with fs instead of fetch, then runs
// a synthetic profile through predictAll and prints JSON for comparison
// against the Python predict.py implementations.
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Catalog, Movie, Critic, MovieRows, XgbModel, NNMeta, NNModel, Models } from "../src/lib/types";
import { predictAll, topCriticMatches } from "../src/lib/predict";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const DATA = join(SCRIPT_DIR, "..", "public", "data");
const readJson = <T,>(name: string): T => JSON.parse(readFileSync(join(DATA, name), "utf-8"));
const readBuf = (name: string): ArrayBuffer => {
  const b = readFileSync(join(DATA, name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

function loadCatalog(): Catalog {
  const movies = readJson<Movie[]>("movies.json");
  const critics = readJson<Critic[]>("critics.json");
  const criticIdx = new Uint16Array(readBuf("ratings_critic_idx.bin"));
  const movieIdx = new Uint16Array(readBuf("ratings_movie_idx.bin"));
  const score = new Float32Array(readBuf("ratings_score.bin"));

  const counts = new Int32Array(movies.length);
  for (let i = 0; i < movieIdx.length; i++) counts[movieIdx[i]]++;
  const byMovie: MovieRows[] = movies.map((_, i) => ({
    criticIdx: new Uint16Array(counts[i]),
    score: new Float32Array(counts[i]),
  }));
  const cursor = new Int32Array(movies.length);
  for (let i = 0; i < movieIdx.length; i++) {
    const m = movieIdx[i];
    const c = cursor[m]++;
    byMovie[m].criticIdx[c] = criticIdx[i];
    byMovie[m].score[c] = score[i];
  }
  return { movies, critics, byMovie };
}

function loadModels(): Models {
  const xgbRaw = readJson<any>("xgb_model.json");
  const nnMeta = readJson<NNMeta>("nn_meta.json");
  const kShrink = readJson<{ kShrink: number }>("k_shrink.json").kShrink;

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
  return { xgb, nn, kShrink };
}

const catalog = loadCatalog();
const models = loadModels();

const args = process.argv.slice(2);
const seedArg = args.find((a) => a.startsWith("--seed="));
const seed = seedArg ? Number(seedArg.split("=")[1]) : 0;

function pseudoRandomProfile(n: number, seedN: number): Map<number, number> {
  // simple LCG for determinism without extra deps
  let state = seedN || 42;
  const rnd = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  const seen = new Map<number, number>();
  const nMovies = catalog.movies.length;
  while (seen.size < n) {
    const idx = Math.floor(rnd() * nMovies);
    if (!seen.has(idx)) seen.set(idx, 1 + Math.floor(rnd() * 5));
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
const matches = topCriticMatches(catalog, seen, models.kShrink);

console.log(JSON.stringify({
  seen: seenIds,
  target_ids: targetIdxs.map((i) => catalog.movies[i].id),
  predictions: predictions.map((p) => ({
    movie_id: catalog.movies[p.movieIdx].id,
    analytic: p.analytic, movie_mean: p.movieMean,
    xgboost: p.xgboost, neural_net: p.neuralNet,
  })),
  top_matches: matches.slice(0, 5).map((m) => ({ critic_id: m.criticId, sim: m.sim, mag_sim: m.magSim, overlap: m.overlap })),
}, null, 2));
