// Import a visitor's real Letterboxd diary into "films you have seen" by
// username. This is a static, serverless site, and Letterboxd does not
// expose a public, CORS-enabled API, so the browser can't fetch
// letterboxd.com directly -- this goes through a public CORS proxy instead
// (api.allorigins.win). That's a real dependency on a third party staying
// up; if it's down or rate-limits, the import fails with a clear error and
// the visitor can still add films by hand. Only the visitor's *public*,
// *recent* diary is available this way (Letterboxd's per-user RSS feed is
// capped to their latest ~50 entries), not their full rating history.
//
// Shared by both datasets: this module only fetches and parses the feed into
// a dataset-agnostic (title, year, star rating) list; each app page matches
// those against its own catalog and rescales the 0.5-5 star rating to its
// own widget (5-star for Rotten Tomatoes, 1-10 for Letterboxd).

const CORS_PROXY = "https://api.allorigins.win/raw?url=";

export interface LetterboxdDiaryEntry {
  title: string;
  year: number | null;
  /** 0.5-5.0 in half-star steps, or null if the visitor didn't rate it. */
  rating: number | null;
}

/** Fetches and parses a public Letterboxd profile's RSS diary feed. Throws
 * with a message safe to show directly to the visitor. */
export async function fetchLetterboxdDiary(username: string): Promise<LetterboxdDiaryEntry[]> {
  const trimmed = username.trim().replace(/^@/, "");
  if (!trimmed) throw new Error("Enter a Letterboxd username first.");
  const rssUrl = `https://letterboxd.com/${encodeURIComponent(trimmed)}/rss/`;

  let res: Response;
  try {
    res = await fetch(CORS_PROXY + encodeURIComponent(rssUrl));
  } catch {
    throw new Error("Couldn't reach the import service -- check your connection and try again.");
  }
  if (!res.ok) {
    throw new Error(
      res.status === 404
        ? `No Letterboxd profile found for "${trimmed}".`
        : `Couldn't reach Letterboxd (status ${res.status}). The import service may be down -- try again shortly.`
    );
  }
  const text = await res.text();
  const doc = new DOMParser().parseFromString(text, "text/xml");
  if (doc.querySelector("parsererror")) {
    throw new Error("Letterboxd returned something unexpected -- double-check the username and try again.");
  }
  const items = Array.from(doc.getElementsByTagName("item"));
  if (items.length === 0) {
    throw new Error(`No public diary entries found for "${trimmed}" -- check the username and that the profile is public.`);
  }

  const entries: LetterboxdDiaryEntry[] = [];
  for (const item of items) {
    const filmTitle = item.getElementsByTagName("letterboxd:filmTitle")[0]?.textContent?.trim();
    if (!filmTitle) continue;
    const filmYearText = item.getElementsByTagName("letterboxd:filmYear")[0]?.textContent;
    const ratingText = item.getElementsByTagName("letterboxd:memberRating")[0]?.textContent;
    entries.push({
      title: filmTitle,
      year: filmYearText ? Number(filmYearText) : null,
      rating: ratingText ? Number(ratingText) : null,
    });
  }
  return entries;
}

export function normalizeTitle(title: string): string {
  return title
    .toLowerCase()
    .replace(/^(the|a|an)\s+/, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

/** Matches diary entries to catalog positions by normalized (title, year),
 * falling back to title-only when it's unambiguous in this catalog. Skips
 * unrated entries (nothing to seed "seen" with) and keeps only the first
 * match per catalog film. Returns the rating on Letterboxd's native
 * 0.5-5.0 star scale -- callers rescale to their own widget. */
export function matchDiaryToCatalog(
  entries: LetterboxdDiaryEntry[],
  movies: { title: string; year: number | null }[]
): { idx: number; rating: number }[] {
  const byTitleYear = new Map<string, number>();
  const byTitle = new Map<string, number>();
  const titleCounts = new Map<string, number>();
  movies.forEach((m, i) => {
    const key = normalizeTitle(m.title);
    titleCounts.set(key, (titleCounts.get(key) ?? 0) + 1);
    const tyKey = `${key}|${m.year ?? ""}`;
    if (!byTitleYear.has(tyKey)) byTitleYear.set(tyKey, i);
    if (!byTitle.has(key)) byTitle.set(key, i);
  });

  const matched: { idx: number; rating: number }[] = [];
  const claimed = new Set<number>();
  for (const entry of entries) {
    if (entry.rating == null) continue;
    const key = normalizeTitle(entry.title);
    let idx = byTitleYear.get(`${key}|${entry.year ?? ""}`);
    if (idx == null && titleCounts.get(key) === 1) idx = byTitle.get(key);
    if (idx == null || claimed.has(idx)) continue;
    claimed.add(idx);
    matched.push({ idx, rating: entry.rating });
  }
  return matched;
}
