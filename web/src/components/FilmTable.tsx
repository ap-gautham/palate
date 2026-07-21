import { Fragment, useState, type ReactNode } from "react";
import { StarRating } from "./StarRating";

const SUGGEST_TIP = "Rate one or all of these similar films to improve this prediction.";

interface Props<M extends { title: string; year: number | null }> {
  idxs: number[];
  movies: M[];
  ratings: Map<number, number>;
  onRate: (idx: number, value: number) => void;
  onRemove: (idx: number) => void;
  label: (m: M) => string;
  emptyText: string;
  /** Defaults to the 5-star widget (Rotten Tomatoes); Letterboxd passes a
   * 1-10 RatingInput instead. */
  renderRating?: (value: number | null, onChange: (v: number) => void) => ReactNode;
  /** Denominator shown in the score pill ("4/5" vs "7/10"). Defaults to 5. */
  scoreMax?: number;
  /** When given, each row gets an expandable "movies like this one" dropdown
   * -- 20 catalog indices most similar to that row's film. Only meaningful on
   * the "films to predict" table. */
  suggestions?: (idx: number) => number[];
  /** Adds a chosen suggestion to "films you have seen" (with a default rating). */
  onAddSuggestion?: (idx: number) => void;
  /** Catalog indices already in "films you have seen" -- drives the toggle
   * arrow's green/red color and the per-suggestion "already seen" state. */
  seenSet?: Set<number>;
}

export function FilmTable<M extends { title: string; year: number | null }>({
  idxs, movies, ratings, onRate, onRemove, label, emptyText, renderRating, scoreMax = 5,
  suggestions, onAddSuggestion, seenSet,
}: Props<M>) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const hasSuggestions = suggestions != null;
  const colCount = 4 + (hasSuggestions ? 1 : 0);

  if (idxs.length === 0) {
    return <p className="muted small">{emptyText}</p>;
  }

  function toggle(idx: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  return (
    <div className="film-table-wrap">
      <table className="film-table">
        <thead>
          <tr>
            {hasSuggestions && <th></th>}
            <th></th>
            <th>Film</th>
            <th>Your rating</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {idxs.map((idx) => {
            const rating = ratings.get(idx) ?? null;
            const similar = hasSuggestions ? suggestions(idx) : [];
            const hasSeenSimilar = seenSet != null && similar.some((j) => seenSet.has(j));
            return (
              <Fragment key={idx}>
                <tr>
                  {hasSuggestions && (
                    <td>
                      <button
                        type="button"
                        className={"suggest-toggle" + (hasSeenSimilar ? " green" : " red")}
                        title={SUGGEST_TIP}
                        aria-label={SUGGEST_TIP}
                        onClick={() => toggle(idx)}
                      >
                        {expanded.has(idx) ? "▾" : "▸"}
                      </button>
                    </td>
                  )}
                  <td>
                    <button type="button" className="remove-btn" title="Remove" onClick={() => onRemove(idx)}>
                      ✕
                    </button>
                  </td>
                  <td>{label(movies[idx])}</td>
                  <td>
                    {renderRating
                      ? renderRating(rating, (v) => onRate(idx, v))
                      : <StarRating value={rating} onChange={(v) => onRate(idx, v)} />}
                  </td>
                  <td>
                    {rating != null
                      ? <span className="score-pill">{rating}/{scoreMax}</span>
                      : <span className="muted small">not rated</span>}
                  </td>
                </tr>
                {hasSuggestions && expanded.has(idx) && (
                  <tr className="suggest-row">
                    <td colSpan={colCount}>
                      <div className="suggest-panel">
                        <p className="suggest-tip">{SUGGEST_TIP}</p>
                        {similar.length === 0 ? (
                          <p className="muted small">No similar films found.</p>
                        ) : (
                          <ul className="suggest-list">
                            {similar.map((j) => {
                              const alreadySeen = seenSet?.has(j) ?? false;
                              return (
                                <li key={j}>
                                  <span>{label(movies[j])}</span>
                                  {alreadySeen ? (
                                    <span className="suggest-seen">✓ seen</span>
                                  ) : (
                                    <button
                                      type="button"
                                      className="suggest-add"
                                      onClick={() => onAddSuggestion?.(j)}
                                    >
                                      Add
                                    </button>
                                  )}
                                </li>
                              );
                            })}
                          </ul>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
