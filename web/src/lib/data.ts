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
  const [movies, critics, criticIdxBuf, movieIdxBuf, scoreBuf] = await Promise.all([
    fetchJson<Movie[]>("movies.json"),
    fetchJson<Critic[]>("critics.json"),
    fetchBuffer("ratings_critic_idx.bin"),
    fetchBuffer("ratings_movie_idx.bin"),
    fetchBuffer("ratings_score.bin"),
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

  return { movies, critics, byMovie };
}

export async function loadModels(): Promise<Models> {
  const [xgbRaw, nnMeta, kStar] = await Promise.all([
    fetchJson<any>("xgb_model.json"),
    fetchJson<NNMeta>("nn_meta.json"),
    fetchJson<{ kShrink: number }>("k_shrink.json").catch(() => ({ kShrink: 8 })),
  ]);

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
  const members = await Promise.all(
    Array.from({ length: nnMeta.ensembleSize }, (_, m) =>
      fetchBuffer(`nn_weights_member${m}.bin`).then((buf) => new Float32Array(buf))
    )
  );
  const nn: NNModel = { meta: nnMeta, members, layerOffset };

  return { xgb, nn, kShrink: kStar.kShrink };
}
