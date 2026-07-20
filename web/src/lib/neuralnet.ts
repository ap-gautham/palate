import type { NNModel } from "./types";

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

// Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7 -- plenty for a model
// whose own outputs are only ever displayed to 2 decimal places.
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

function gelu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  const invSqrt2 = Math.SQRT1_2;
  for (let i = 0; i < x.length; i++) out[i] = 0.5 * x[i] * (1 + erf(x[i] * invSqrt2));
  return out;
}

function linear(x: Float32Array, W: Float32Array, b: Float32Array, outDim: number, inDim: number): Float32Array {
  const y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let s = b[o];
    const base = o * inDim;
    for (let i = 0; i < inDim; i++) s += W[base + i] * x[i];
    y[o] = s;
  }
  return y;
}

function batchNorm(x: Float32Array, weight: Float32Array, bias: Float32Array,
                    runningMean: Float32Array, runningVar: Float32Array): Float32Array {
  const eps = 1e-5;
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    y[i] = ((x[i] - runningMean[i]) / Math.sqrt(runningVar[i] + eps)) * weight[i] + bias[i];
  }
  return y;
}

function layerNorm(x: Float32Array, weight: Float32Array, bias: Float32Array): Float32Array {
  const eps = 1e-5;
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  variance /= n;
  const denom = Math.sqrt(variance + eps);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = ((x[i] - mean) / denom) * weight[i] + bias[i];
  return y;
}

function addVec(a: Float32Array, b: Float32Array): Float32Array {
  const y = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i];
  return y;
}

/** Standardize a raw feature row into the network's numeric input vector.
 * Mirrors design3_neural/predict.py:predict's preprocessing block exactly:
 * log1p(clip>=0) on log_cols first, then NaN-impute, then z-score. */
export function prepareNumeric(model: NNModel, row: Record<string, number>): Float32Array {
  const { numericCols, logCols, mu, sd, muImpute } = model.meta;
  const logSet = new Set(logCols);
  const out = new Float32Array(numericCols.length);
  for (let i = 0; i < numericCols.length; i++) {
    const name = numericCols[i];
    let v = row[name];
    if (logSet.has(name)) v = Math.log1p(Math.max(v, 0));
    if (Number.isNaN(v)) v = muImpute[i];
    out[i] = (v - mu[i]) / sd[i];
  }
  return out;
}

function layerView(model: NNModel, member: Float32Array, name: string): Float32Array {
  const l = model.layerOffset.get(name)!;
  return member.subarray(l.offset, l.offset + l.size);
}

function forwardOne(model: NNModel, member: Float32Array, numeric: Float32Array, genreId: number): number {
  const { width, embDim, depth } = model.meta;
  const xNorm = batchNorm(
    numeric,
    layerView(model, member, "input_norm.weight"),
    layerView(model, member, "input_norm.bias"),
    layerView(model, member, "input_norm.running_mean"),
    layerView(model, member, "input_norm.running_var")
  );
  const embAll = layerView(model, member, "embedding.weight");
  const emb = embAll.subarray(genreId * embDim, genreId * embDim + embDim);

  const cat = new Float32Array(xNorm.length + emb.length);
  cat.set(xNorm, 0);
  cat.set(emb, xNorm.length);

  let h = linear(cat, layerView(model, member, "proj.weight"), layerView(model, member, "proj.bias"), width, cat.length);

  for (let i = 0; i < depth; i++) {
    const p = `blocks.${i}.net`;
    let z = linear(h, layerView(model, member, `${p}.0.weight`), layerView(model, member, `${p}.0.bias`), width, width);
    z = batchNorm(z, layerView(model, member, `${p}.1.weight`), layerView(model, member, `${p}.1.bias`),
      layerView(model, member, `${p}.1.running_mean`), layerView(model, member, `${p}.1.running_var`));
    z = gelu(z);
    z = linear(z, layerView(model, member, `${p}.4.weight`), layerView(model, member, `${p}.4.bias`), width, width);
    z = batchNorm(z, layerView(model, member, `${p}.5.weight`), layerView(model, member, `${p}.5.bias`),
      layerView(model, member, `${p}.5.running_mean`), layerView(model, member, `${p}.5.running_var`));
    h = gelu(addVec(h, z));
  }

  const ln = layerNorm(h, layerView(model, member, "head.0.weight"), layerView(model, member, "head.0.bias"));
  const out = linear(ln, layerView(model, member, "head.1.weight"), layerView(model, member, "head.1.bias"), 1, width);
  return out[0];
}

/** `range` defaults to the Rotten Tomatoes 0-5 scale; pass [1,10] for
 * Letterboxd. */
export function predictNeuralNet(
  model: NNModel,
  row: Record<string, number>,
  range: [number, number] = [0, 5]
): number {
  const numeric = prepareNumeric(model, row);
  const genreId = row.genre_id;
  let sum = 0;
  for (const member of model.members) sum += forwardOne(model, member, numeric, genreId);
  return clamp(sum / model.members.length, range[0], range[1]);
}
