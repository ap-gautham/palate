import type { ReactNode } from "react";
import { StarRating } from "./StarRating";

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
}

export function FilmTable<M extends { title: string; year: number | null }>({
  idxs, movies, ratings, onRate, onRemove, label, emptyText, renderRating, scoreMax = 5,
}: Props<M>) {
  if (idxs.length === 0) {
    return <p className="muted small">{emptyText}</p>;
  }
  return (
    <div className="film-table-wrap">
      <table className="film-table">
        <thead>
          <tr>
            <th></th>
            <th>Film</th>
            <th>Your rating</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {idxs.map((idx) => {
            const rating = ratings.get(idx) ?? null;
            return (
              <tr key={idx}>
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
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
