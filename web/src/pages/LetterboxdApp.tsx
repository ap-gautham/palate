import { useMemo, useState } from "react";
import type { LetterboxdData } from "../lib/letterboxd/useLetterboxdData";
import { predictAll, topMemberMatches, mse } from "../lib/letterboxd/predict";
import { FilmAutocomplete } from "../components/FilmAutocomplete";
import { FilmTable } from "../components/FilmTable";
import { RatingInput } from "../components/RatingInput";

const MIN_SEEN = 5;
const DEFAULT_RATING = 6;

type SortChoice = "title" | "year" | "reviewed";
const SORT_LABELS: Record<SortChoice, string> = {
  title: "Title (A–Z)",
  year: "Year (newest first)",
  reviewed: "Most rated",
};

function labelOf(m: { title: string; year: number | null }) {
  return `${m.title} (${m.year ?? "n/a"})`;
}

export function LetterboxdApp({ data }: { data: LetterboxdData }) {
  const { catalog, models } = data;

  const [sortChoice, setSortChoice] = useState<SortChoice>("title");
  const [seenIdxs, setSeenIdxs] = useState<number[]>([]);
  const [seenRatings, setSeenRatings] = useState<Record<number, number>>({});
  const [predictIdxs, setPredictIdxs] = useState<number[]>([]);
  const [predictRatings, setPredictRatings] = useState<Record<number, number>>({});
  const [showMatches, setShowMatches] = useState(false);

  const orderedIdxs = useMemo(() => {
    const idxs = catalog.movies.map((_, i) => i);
    if (sortChoice === "title") {
      idxs.sort((a, b) => catalog.movies[a].title.localeCompare(catalog.movies[b].title, undefined, { sensitivity: "base" }));
    } else if (sortChoice === "year") {
      idxs.sort((a, b) => (catalog.movies[b].year ?? -Infinity) - (catalog.movies[a].year ?? -Infinity));
    } else {
      idxs.sort((a, b) => catalog.movies[b].nScores - catalog.movies[a].nScores);
    }
    return idxs;
  }, [catalog, sortChoice]);

  const chosen = useMemo(() => new Set([...seenIdxs, ...predictIdxs]), [seenIdxs, predictIdxs]);

  function addSeen(idx: number) {
    setSeenIdxs((prev) => [...prev, idx]);
    setSeenRatings((prev) => ({ ...prev, [idx]: DEFAULT_RATING }));
  }
  function addPredict(idx: number) {
    setPredictIdxs((prev) => [...prev, idx]);
  }
  function removeSeen(idx: number) {
    setSeenIdxs((prev) => prev.filter((i) => i !== idx));
    setSeenRatings((prev) => {
      const next = { ...prev };
      delete next[idx];
      return next;
    });
  }
  function removePredict(idx: number) {
    setPredictIdxs((prev) => prev.filter((i) => i !== idx));
    setPredictRatings((prev) => {
      const next = { ...prev };
      delete next[idx];
      return next;
    });
  }

  const seenRatingsMap = useMemo(() => new Map(Object.entries(seenRatings).map(([k, v]) => [Number(k), v])), [seenRatings]);

  const seenValues = Object.values(seenRatings);
  const seenMean = seenValues.length ? seenValues.reduce((s, v) => s + v, 0) / seenValues.length : 0;
  const seenStd = seenValues.length
    ? Math.sqrt(seenValues.reduce((s, v) => s + (v - seenMean) ** 2, 0) / seenValues.length)
    : 0;

  const predictions = useMemo(() => {
    if (seenIdxs.length < MIN_SEEN || seenStd < 1e-9 || predictIdxs.length === 0) return null;
    return predictAll(catalog, models, seenRatingsMap, predictIdxs);
  }, [catalog, models, seenRatingsMap, predictIdxs, seenIdxs.length, seenStd]);

  const matches = useMemo(() => {
    if (!showMatches || seenIdxs.length === 0) return [];
    return topMemberMatches(catalog, seenRatingsMap, models.kShrink);
  }, [showMatches, catalog, seenRatingsMap, models.kShrink, seenIdxs.length]);

  return (
    <div className="predict-app">
      <p className="lead-caption">
        Rate films you have seen on Letterboxd's 1–10 scale, then predict your score for others three ways: the
        analytic member-match formula, an XGBoost model, and a neural network — all trained on real member
        histories instead of critic pseudo-users. Score the films you predict to see which method is closest to
        your taste.
      </p>

      <div className="sort-row">
        <span className="sort-label">Sort the search list by</span>
        {(Object.keys(SORT_LABELS) as SortChoice[]).map((key) => (
          <label key={key} className="radio-pill">
            <input type="radio" checked={sortChoice === key} onChange={() => setSortChoice(key)} />
            {SORT_LABELS[key]}
          </label>
        ))}
      </div>

      <section className="card-section">
        <h2>1. Films you have seen</h2>
        <FilmAutocomplete
          placeholder="Add a film you have seen"
          orderedIdxs={orderedIdxs}
          movies={catalog.movies}
          excluded={chosen}
          onAdd={addSeen}
          label={labelOf}
        />
        <FilmTable
          idxs={seenIdxs}
          movies={catalog.movies}
          ratings={seenRatingsMap}
          onRate={(idx, v) => setSeenRatings((prev) => ({ ...prev, [idx]: v }))}
          onRemove={removeSeen}
          label={labelOf}
          emptyText="No films added yet."
          scoreMax={10}
          renderRating={(value, onChange) => <RatingInput value={value} onChange={onChange} />}
        />
      </section>

      <section className="card-section">
        <h2>2. Films to predict</h2>
        <p className="muted small">
          Scoring these is optional. If you rate them, each method's error against your own score is reported below.
        </p>
        <FilmAutocomplete
          placeholder="Add a film to predict"
          orderedIdxs={orderedIdxs}
          movies={catalog.movies}
          excluded={chosen}
          onAdd={addPredict}
          label={labelOf}
        />
        <FilmTable
          idxs={predictIdxs}
          movies={catalog.movies}
          ratings={new Map(Object.entries(predictRatings).map(([k, v]) => [Number(k), v]))}
          onRate={(idx, v) => setPredictRatings((prev) => ({ ...prev, [idx]: v }))}
          onRemove={removePredict}
          label={labelOf}
          emptyText="No films added yet."
          scoreMax={10}
          renderRating={(value, onChange) => <RatingInput value={value} onChange={onChange} />}
        />
      </section>

      <section className="card-section">
        <h2>3. Predictions</h2>
        {seenIdxs.length < MIN_SEEN ? (
          <p className="info-box">
            Rate at least {MIN_SEEN} seen films to build a taste profile (currently {seenIdxs.length}).
          </p>
        ) : seenStd < 1e-9 ? (
          <p className="warn-box">Give your seen films different ratings so similarity is defined.</p>
        ) : predictIdxs.length === 0 ? (
          <p className="info-box">Add at least one film to predict in section 2.</p>
        ) : (
          predictions && (
            <PredictionsView
              predictIdxs={predictIdxs}
              movies={catalog.movies}
              predictions={predictions}
              predictRatings={predictRatings}
              label={labelOf}
            />
          )
        )}

        <details className="matches-expander" onToggle={(e) => setShowMatches((e.target as HTMLDetailsElement).open)}>
          <summary>Your closest members</summary>
          {matches.length === 0 ? (
            <p className="muted small">No overlapping member yet; predictions fall back to consensus.</p>
          ) : (
            <div className="film-table-wrap">
              <table className="film-table">
                <thead>
                  <tr>
                    <th>Member</th>
                    <th>Alignment</th>
                    <th>Scale match</th>
                    <th>Films in common</th>
                  </tr>
                </thead>
                <tbody>
                  {matches.map((m) => (
                    <tr key={m.memberId}>
                      <td>{m.memberId}</td>
                      <td>{m.sim.toFixed(2)}</td>
                      <td>{m.magSim.toFixed(2)}</td>
                      <td>{m.overlap}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </details>
      </section>
    </div>
  );
}

function PredictionsView({
  predictIdxs, movies, predictions, predictRatings, label,
}: {
  predictIdxs: number[];
  movies: LetterboxdData["catalog"]["movies"];
  predictions: NonNullable<ReturnType<typeof predictAll>>;
  predictRatings: Record<number, number>;
  label: (m: { title: string; year: number | null }) => string;
}) {
  const byIdx = new Map(predictions.map((p) => [p.movieIdx, p]));
  const hasAnyScore = predictIdxs.some((i) => predictRatings[i] != null);

  const analyticArr = predictIdxs.map((i) => byIdx.get(i)?.analytic ?? NaN);
  const xgbArr = predictIdxs.map((i) => byIdx.get(i)?.xgboost ?? NaN);
  const nnArr = predictIdxs.map((i) => byIdx.get(i)?.neuralNet ?? NaN);
  const meanArr = predictIdxs.map((i) => byIdx.get(i)?.movieMean ?? NaN);
  const truth = predictIdxs.map((i) => predictRatings[i] ?? NaN);
  const nScored = truth.filter((t) => !Number.isNaN(t)).length;

  const candidates: [string, number[]][] = [
    ["Analytic formula", analyticArr],
    ["XGBoost", xgbArr],
    ["Neural net", nnArr],
    ["Consensus mean", meanArr],
  ];
  const rows = candidates.map(([name, arr]) => [name, mse(arr, truth)] as const);
  const best = rows.reduce((b, r) => (r[1] != null && (b[1] == null || r[1] < b[1]) ? r : b), rows[0]);

  return (
    <>
      <div className="film-table-wrap">
        <table className="film-table predictions-table">
          <thead>
            <tr>
              <th>Film</th>
              <th>Analytic</th>
              <th>XGBoost</th>
              <th>Neural net</th>
              <th>Consensus (mean)</th>
              {hasAnyScore && <th>Your score</th>}
            </tr>
          </thead>
          <tbody>
            {predictIdxs.map((idx) => {
              const p = byIdx.get(idx);
              return (
                <tr key={idx}>
                  <td>{label(movies[idx])}</td>
                  <td>{p?.analytic.toFixed(2)}</td>
                  <td>{p?.xgboost.toFixed(2)}</td>
                  <td>{p?.neuralNet.toFixed(2)}</td>
                  <td>{p?.movieMean.toFixed(2)}</td>
                  {hasAnyScore && <td>{predictRatings[idx] != null ? `${predictRatings[idx]}/10` : "—"}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3 className="best-predictor-heading">Which method predicts <em>you</em> best?</h3>
      {nScored === 0 ? (
        <p className="info-box">
          Rate one or more of your predict films (section 2) to see the mean squared error of each method against
          your own score.
        </p>
      ) : (
        <>
          <p className="muted small">
            MSE against your own score over {nScored} rated film(s). Lower is closer to your taste —{" "}
            <strong>{best[0]}</strong> is closest here.
          </p>
          <div className="mse-cards">
            {rows.map(([name, value]) => (
              <div key={name} className={"mse-card" + (name === best[0] ? " best" : "")}>
                <div className="mse-name">
                  {name}
                  {name === best[0] ? " ✅" : ""}
                </div>
                <div className="mse-value">{value != null ? value.toFixed(3) : "—"}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}
