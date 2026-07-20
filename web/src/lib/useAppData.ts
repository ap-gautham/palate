import { useEffect, useState } from "react";
import type { Catalog, Models } from "./types";
import { loadCatalog, loadModels } from "./data";

export interface AppData {
  catalog: Catalog;
  models: Models;
}

export type AppDataState =
  | { status: "loading"; progress: string }
  | { status: "error"; message: string }
  | { status: "ready"; data: AppData };

export function useAppData(): AppDataState {
  const [state, setState] = useState<AppDataState>({ status: "loading", progress: "Loading catalog…" });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const catalog = await loadCatalog();
        if (cancelled) return;
        setState({ status: "loading", progress: "Loading models (~40MB, one-time)…" });
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
  }, []);

  return state;
}
