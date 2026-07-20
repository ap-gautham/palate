import type { Movie } from "../lib/types";
import { StarRating } from "./StarRating";

interface Props {
  idxs: number[];
  movies: Movie[];
  ratings: Map<number, number>;
  onRate: (idx: number, value: number) => void;
  onRemove: (idx: number) => void;
  label: (m: Movie) => string;
  emptyText: string;
}

export function FilmTable({ idxs, movies, ratings, onRate, onRemove, label, emptyText }: Props) {
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
                  <StarRating value={rating} onChange={(v) => onRate(idx, v)} />
                </td>
                <td>
                  {rating != null ? <span className="score-pill">{rating}/5</span> : <span className="muted small">not rated</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
