import type { Catalog, Movie, Member, MovieRows, XgbModel, NNMeta, NNModel, Models } from "./types";

const BASE = import.meta.env.BASE_URL + "data/letterboxd/";

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
  const [movies, members, memberIdxBuf, movieIdxBuf, scoreBuf] = await Promise.all([
    fetchJson<Movie[]>("movies.json"),
    fetchJson<Member[]>("members.json"),
    fetchBuffer("ratings_member_idx.bin"),
    fetchBuffer("ratings_movie_idx.bin"),
    fetchBuffer("ratings_score.bin"),
  ]);
  const memberIdx = new Uint16Array(memberIdxBuf);
  const movieIdx = new Uint16Array(movieIdxBuf);
  const score = new Float32Array(scoreBuf);

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

export async function loadModels(): Promise<Models> {
  const [xgbRaw, nnMeta, meta] = await Promise.all([
    fetchJson<any>("xgb_model.json"),
    fetchJson<NNMeta>("nn_meta.json"),
    fetchJson<{ kShrink: number; ratingMin: number; ratingMax: number }>("meta.json"),
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

  return { xgb, nn, kShrink: meta.kShrink, ratingMin: meta.ratingMin, ratingMax: meta.ratingMax };
}
