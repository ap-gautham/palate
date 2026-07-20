interface Props {
  value: number | null;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
}

/** A 1-10 numeric stepper for Letterboxd ratings (Rotten Tomatoes keeps the
 * 5-star widget; a 10-point scale is too fine-grained for stars). */
export function RatingInput({ value, onChange, min = 1, max = 10 }: Props) {
  const current = value ?? Math.round((min + max) / 2);

  function clamp(v: number) {
    return Math.min(max, Math.max(min, v));
  }

  return (
    <span className="rating-input">
      <button
        type="button"
        className="rating-step"
        aria-label="Decrease rating"
        onClick={() => onChange(clamp(current - 1))}
        disabled={value != null && current <= min}
      >
        −
      </button>
      <input
        type="number"
        className="rating-number"
        min={min}
        max={max}
        step={1}
        value={value ?? ""}
        placeholder="–"
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v) && e.target.value !== "") onChange(clamp(Math.round(v)));
        }}
      />
      <button
        type="button"
        className="rating-step"
        aria-label="Increase rating"
        onClick={() => onChange(clamp(current + 1))}
        disabled={value != null && current >= max}
      >
        +
      </button>
    </span>
  );
}
