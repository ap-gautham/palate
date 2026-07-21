import { sentimentClass, sentimentFor, sentimentText } from "../lib/sentiment";

interface HeadlinePrediction {
  movieIdx: number;
  neuralNet: number;
  neuralNetZ: number | null;
  movieMean: number;
}

/** The "main result" for each predicted film: the neural net's raw (and, if
 * available, z-score) prediction, plus a plain-language sentiment read
 * against the movie's consensus mean. Sits above the full method-by-method
 * table, which stays exactly as it was. Shared by both datasets -- the
 * prediction shape is structurally identical, only `groupNoun` differs. */
export function NeuralHeadline({
  predictIdxs, movies, label, predictions, groupNoun, hasAnyZ,
}: {
  predictIdxs: number[];
  movies: { title: string; year: number | null }[];
  label: (m: { title: string; year: number | null }) => string;
  predictions: HeadlinePrediction[];
  groupNoun: string;
  hasAnyZ: boolean;
}) {
  const byIdx = new Map(predictions.map((p) => [p.movieIdx, p]));
  return (
    <div className="nn-headline-grid">
      {predictIdxs.map((idx) => {
        const p = byIdx.get(idx);
        if (!p) return null;
        const diff = p.neuralNet - p.movieMean;
        const sentiment = sentimentFor(p.neuralNet, p.movieMean);
        return (
          <div key={idx} className="nn-headline-card">
            <div className="nn-headline-title">{label(movies[idx])}</div>
            <div className="nn-headline-score">
              {p.neuralNet.toFixed(2)}
              {hasAnyZ && (
                <span className="nn-headline-z">
                  {p.neuralNetZ != null ? ` / z ${p.neuralNetZ.toFixed(2)}` : " / z —"}
                </span>
              )}
            </div>
            <div className={"sentiment " + sentimentClass(sentiment)}>
              {sentimentText(sentiment, diff, groupNoun)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
