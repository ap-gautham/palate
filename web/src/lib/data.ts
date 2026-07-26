import type { Catalog, Movie, Critic, MovieRows, XgbModel, NNMeta, NNModel, Models } from "./types";

const BASE = import.meta.env.BASE_URL + "data/rotten_tomatoes/";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`failed to fetch ${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

async function fetchBuffer(path: string): Promise<ArrayBuffer> {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`failed to fetch ${path}: ${res.status}`);
  return res.arrayBuffer();
}

export async function loadCatalog(): Promise<Catalog> {
  const [movies, critics, criticIdxBuf, movieIdxBuf, scoreBuf, similar, themeSim, kShrinkMeta] = await Promise.all([
    fetchJson<Movie[]>("movies.json"),
    fetchJson<Critic[]>("critics.json"),
    fetchBuffer("ratings_critic_idx.bin"),
    fetchBuffer("ratings_movie_idx.bin"),
    fetchBuffer("ratings_score.bin"),
    fetchJson<number[][]>("similar.json").catch(() => []),
    fetchJson<{ themes: string[]; matrix: number[][] }>("theme_similarity.json"),
    fetchJson<{ kShrink: number; globalStd: number }>("k_shrink.json").catch(() => ({ kShrink: 8, globalStd: 1 })),
  ]);
  const criticIdx = new Uint16Array(criticIdxBuf);
  const movieIdx = new Uint16Array(movieIdxBuf);
  const score = new Float32Array(scoreBuf);

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
  const similarByMovie = movies.map((_, i) => similar[i] ?? []);
  const themeWidth = themeSim.themes.length;
  const themeMatrix = new Float32Array(themeWidth * themeWidth);
  for (let i = 0; i < themeWidth; i++) themeMatrix.set(themeSim.matrix[i], i * themeWidth);

  return {
    movies, critics, byMovie, similar: similarByMovie,
    themeMatrix, themeVocab: themeSim.themes, globalStd: kShrinkMeta.globalStd,
  };
}

async function loadXgb(xgbFile: string): Promise<XgbModel> {
  const xgbRaw = await fetchJson<any>(xgbFile);
  return {
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
}

async function loadNn(metaFile: string, weightsPrefix: string): Promise<NNModel> {
  const nnMeta = await fetchJson<NNMeta>(metaFile);
  const layerOffset = new Map(nnMeta.layers.map((l) => [l.name, l]));
  const members = await Promise.all(
    Array.from({ length: nnMeta.ensembleSize }, (_, m) =>
      fetchBuffer(`${weightsPrefix}${m}.bin`).then((buf) => new Float32Array(buf))
    )
  );
  return { meta: nnMeta, members, layerOffset };
}

export async function loadModels(): Promise<Models> {
  const [xgb, nn, kStar] = await Promise.all([
    loadXgb("xgb_model.json"),
    loadNn("nn_meta.json", "nn_weights_member"),
    fetchJson<{ kShrink: number }>("k_shrink.json").catch(() => ({ kShrink: 8 })),
  ]);

  const [xgbZ, nnZ] = await Promise.all([
    loadXgb("xgb_z_model.json").catch(() => undefined),
    loadNn("nn_z_meta.json", "nn_z_weights_member").catch(() => undefined),
  ]);

  return { xgb, nn, kShrink: kStar.kShrink, xgbZ, nnZ };
}
