interface Props {
  value: number | null;
  onChange: (value: number) => void;
}

export function StarRating({ value, onChange }: Props) {
  return (
    <span className="stars" role="radiogroup" aria-label="Your rating">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          className={"star" + (value != null && n <= value ? " filled" : "")}
          aria-checked={value === n}
          aria-label={`${n} star${n > 1 ? "s" : ""}`}
          role="radio"
          onClick={() => onChange(n)}
        >
          ★
        </button>
      ))}
    </span>
  );
}
