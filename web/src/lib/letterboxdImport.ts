// Import a visitor's real Letterboxd ratings into "films you have seen" from
// their own data export. This is a static, serverless site with no backend,
// and Letterboxd has no public API and blocks unauthenticated scraping (its
// RSS feed is capped to the ~50 most recent diary entries, and its HTML
// pages sit behind a bot-check that a plain fetch can't pass) -- so there is
// no reliable way to pull a visitor's full history live. Letterboxd's own
// "Export Your Data" feature (Settings -> Import & Export) gives the
// visitor a zip with ratings.csv (and diary.csv) covering every film
// they've ever rated. Parsing that file client-side needs no network call
// at all, so it's both more complete and more reliable than any live fetch.
//
// Shared by both datasets: this module only parses the CSV into a
// dataset-agnostic (title, year, star rating) list; each app page matches
// those against its own catalog and rescales the 0.5-5 star rating to its
// own widget (5-star for Rotten Tomatoes, 1-10 for Letterboxd).

export interface LetterboxdDiaryEntry {
  title: string;
  year: number | null;
  /** 0.5-5.0 in half-star steps, or null if the visitor didn't rate it. */
  rating: number | null;
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
      continue;
    }
    if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else {
      field += c;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

/** Parses a Letterboxd export CSV (ratings.csv or diary.csv) into a rating
 * list. Column order isn't assumed -- only that a header row names "Name",
 * "Year" and "Rating" columns, which both of those files have. */
export function parseLetterboxdCsv(text: string): { entries: LetterboxdDiaryEntry[] } | { error: string } {
  const rows = parseCsvRows(text);
  if (rows.length < 2) {
    return { error: "That file doesn't look like a Letterboxd export CSV." };
  }
  const header = rows[0].map((h) => h.trim().toLowerCase());
  const nameIdx = header.indexOf("name");
  const yearIdx = header.indexOf("year");
  const ratingIdx = header.indexOf("rating");
  if (nameIdx === -1 || ratingIdx === -1) {
    return {
      error: "That CSV doesn't have the expected columns -- upload ratings.csv or diary.csv from your Letterboxd data export.",
    };
  }

  const entries: LetterboxdDiaryEntry[] = [];
  for (const r of rows.slice(1)) {
    const title = r[nameIdx]?.trim();
    if (!title) continue;
    const yearText = yearIdx !== -1 ? r[yearIdx]?.trim() : "";
    const ratingText = r[ratingIdx]?.trim();
    entries.push({
      title,
      year: yearText ? Number(yearText) : null,
      rating: ratingText ? Number(ratingText) : null,
    });
  }
  if (entries.length === 0) {
    return { error: "No rated films found in that file." };
  }
  return { entries };
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
