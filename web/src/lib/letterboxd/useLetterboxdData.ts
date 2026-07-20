import { useEffect, useState } from "react";
import type { Catalog, Models } from "./types";
import { loadCatalog, loadModels } from "./data";

export interface LetterboxdData {
  catalog: Catalog;
  models: Models;
}

export type LetterboxdDataState =
  | { status: "loading"; progress: string }
  | { status: "error"; message: string }
  | { status: "ready"; data: LetterboxdData };

/** Lazy: only call this (from the Letterboxd tab) once the user actually
 * opens it, so the ~60MB export isn't fetched until needed. */
export function useLetterboxdData(enabled: boolean): LetterboxdDataState {
  const [state, setState] = useState<LetterboxdDataState>({ status: "loading", progress: "Loading catalog…" });

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    (async () => {
      try {
        const catalog = await loadCatalog();
        if (cancelled) return;
        setState({ status: "loading", progress: "Loading models (~60MB, one-time)…" });
        const models = await loadModels();
        if (cancelled) return;
        setState({ status: "ready", data: { catalog, models } });
      } catch (err) {
        if (cancelled) return;
        setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  return state;
}
