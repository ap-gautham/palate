import type { XgbModel } from "./types";

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Walks each tree from the native XGBoost JSON dump. Mirrors
 * design2_xgboost/predict.py (xgb.Booster.predict), -1 marks a leaf.
 * `range` defaults to the Rotten Tomatoes 0-5 scale; pass [1,10] for
 * Letterboxd. */
export function predictXgb(
  model: XgbModel,
  featureRow: Record<string, number>,
  range: [number, number] = [0, 5]
): number {
  const x = model.featureColumns.map((name) => featureRow[name]);
  let sum = model.baseScore;
  for (const tree of model.trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      const value = x[tree.splitIdx[node]];
      const cond = tree.splitCond[node];
      const goLeft = Number.isNaN(value) ? tree.defaultLeft[node] === 1 : value < cond;
      node = goLeft ? tree.left[node] : tree.right[node];
    }
    sum += tree.leafValue[node];
  }
  return clamp(sum, range[0], range[1]);
}
