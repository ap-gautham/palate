// Save/load the app's current inputs (seen films + ratings, predict films +
// ratings) as a small JSON-in-a-.txt file, so a visitor can pick up where
// they left off without re-adding every film by hand. Shared by both
// datasets (Rotten Tomatoes and Letterboxd) -- the format only depends on
// each catalog's stable `id` strings, not on dataset-specific rating scales
// or model logic, so unlike the prediction code this is not duplicated per
// project.

export interface SettingsPayload {
  dataset: string;
  seen: { id: string; rating: number }[];
  predict: { id: string; rating: number | null }[];
}

/** Builds the exportable payload from the app's current state. `movies` only
 * needs the stable `id` field (both datasets' Movie type has one). */
export function serializeSettings(
  dataset: string,
  movies: { id: string }[],
  seenIdxs: number[],
  seenRatings: Record<number, number>,
  predictIdxs: number[],
  predictRatings: Record<number, number>
): string {
  const payload: SettingsPayload = {
    dataset,
    seen: seenIdxs
      .filter((idx) => movies[idx] != null)
      .map((idx) => ({ id: movies[idx].id, rating: seenRatings[idx] })),
    predict: predictIdxs
      .filter((idx) => movies[idx] != null)
      .map((idx) => ({ id: movies[idx].id, rating: predictRatings[idx] ?? null })),
  };
  return JSON.stringify(payload, null, 2);
}

/** Triggers a browser download of `text` as `filename` (no server involved). */
export function downloadTextFile(text: string, filename: string): void {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export interface ParsedSettings {
  dataset: string;
  seen: { idx: number; rating: number }[];
  predict: { idx: number; rating: number | null }[];
  /** Entries in the file whose id isn't in this catalog (e.g. a file saved
   * against the other dataset, or an older/newer catalog snapshot). */
  unmatched: number;
}

/** Parses a previously-exported settings file against the current catalog.
 * Returns `{ error }` if the file isn't recognizable JSON in the expected
 * shape at all (a wrong-dataset file still parses -- its ids just won't
 * match, reported via `unmatched`, and the caller decides how to warn). */
export function parseSettingsFile(
  text: string,
  movies: { id: string }[]
): ParsedSettings | { error: string } {
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    return { error: "That file isn't valid -- expected a .txt exported from this app." };
  }
  if (
    !payload || typeof payload !== "object" ||
    !Array.isArray((payload as SettingsPayload).seen) ||
    !Array.isArray((payload as SettingsPayload).predict)
  ) {
    return { error: "That file isn't valid -- expected a .txt exported from this app." };
  }
  const p = payload as SettingsPayload;
  const idToIdx = new Map(movies.map((m, i) => [m.id, i]));
  let unmatched = 0;

  const seen: { idx: number; rating: number }[] = [];
  for (const entry of p.seen) {
    const idx = idToIdx.get(entry?.id);
    if (idx == null || typeof entry.rating !== "number") { unmatched++; continue; }
    seen.push({ idx, rating: entry.rating });
  }
  const predict: { idx: number; rating: number | null }[] = [];
  for (const entry of p.predict) {
    const idx = idToIdx.get(entry?.id);
    if (idx == null) { unmatched++; continue; }
    predict.push({ idx, rating: typeof entry.rating === "number" ? entry.rating : null });
  }
  return { dataset: typeof p.dataset === "string" ? p.dataset : "", seen, predict, unmatched };
}

export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("Couldn't read that file."));
    reader.readAsText(file);
  });
}
