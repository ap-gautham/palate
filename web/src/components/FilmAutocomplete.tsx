import { useMemo, useState, useRef } from "react";

interface Props<M extends { title: string }> {
  placeholder: string;
  orderedIdxs: number[];
  movies: M[];
  excluded: Set<number>;
  onAdd: (idx: number) => void;
  label: (m: M) => string;
}

export function FilmAutocomplete<M extends { title: string }>({
  placeholder, orderedIdxs, movies, excluded, onAdd, label,
}: Props<M>) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    const candidates = orderedIdxs.filter((i) => !excluded.has(i));
    const filtered = q ? candidates.filter((i) => movies[i].title.toLowerCase().includes(q)) : candidates;
    // Keep the entire matching catalog available. The list itself is bounded
    // visually and scrolls, so popular title searches are not silently cut off.
    return filtered;
  }, [query, orderedIdxs, excluded, movies]);

  function choose(idx: number) {
    onAdd(idx);
    setQuery("");
    setOpen(false);
  }

  return (
    <div className="autocomplete" ref={boxRef}>
      <input
        type="text"
        value={query}
        placeholder={placeholder}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
      />
      {open && results.length > 0 && (
        <ul className="autocomplete-list">
          {results.map((i) => (
            <li key={i}>
              <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => choose(i)}>
                {label(movies[i])}
              </button>
            </li>
          ))}
        </ul>
      )}
      {open && query && results.length === 0 && (
        <ul className="autocomplete-list">
          <li className="autocomplete-empty">No matches</li>
        </ul>
      )}
    </div>
  );
}
