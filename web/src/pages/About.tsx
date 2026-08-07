const BASE = import.meta.env.BASE_URL;

export function About({ onOpenApp }: { onOpenApp: () => void }) {
  return (
    <div className="about-page">
      <header className="about-header">
        <p className="lead">
          Palate predicts the rating <b>you</b> would give a film from the scores of other raters — matching you
          to the critics or members whose taste tracks yours. It is really a measurement project:{" "}
          <b>where does personalization actually beat the flat average, and where does it just pretend to?</b>
        </p>
        <div className="cta">
          <button className="btn primary" onClick={onOpenApp}>
            Try the app
          </button>
          <a className="btn" href={`${BASE}assets/palate-report.pdf`}>
            Read the report (PDF)
          </a>
          <a className="btn" href="https://github.com/ap-gautham/palate">
            Source on GitHub
          </a>
        </div>
      </header>

      <section>
        <p className="kicker">The data</p>
        <h2>Two datasets, one pipeline</h2>
        <p>
          Rotten Tomatoes has no real users, so ~2,900 professional critics stand in: some of their ratings are
          "films they have seen," one held-out film is the target. Letterboxd has real people — 7,420 members
          with dense rating histories on a native 1–10 scale. The exact same three-model pipeline runs on both,
          fully isolated, so the comparison is symmetric. Both catalogs are also joined to a movie-metadata dump
          (genre, theme, director, actor…) so the models can learn taste like "this user over-rates A24 films."
        </p>
        <ul className="funnel">
          <li>
            <span>Rotten Tomatoes: raw critic reviews (standardized to 0–5)</span>
            <span>1,444,963</span>
          </li>
          <li>
            <span>Critic pseudo-users (≥10 rated films)</span>
            <span>2,894</span>
          </li>
          <li>
            <span>Letterboxd: member ratings (1–10)</span>
            <span>11,078,045</span>
          </li>
          <li>
            <span>Members (≥5 rated films)</span>
            <span>7,420</span>
          </li>
          <li>
            <span>Films in each interactive catalog</span>
            <span>1,000</span>
          </li>
        </ul>
      </section>

      <section>
        <p className="kicker">Three models</p>
        <h2>How the predictions are made</h2>
        <p>
          Three models compete on identical held-out films in both datasets. Each also has a parallel{" "}
          <b>z-score</b> variant — predicting your deviation from your own average instead of your raw score — a
          direct test of whether personalization is taste-matching or just calibration.
        </p>
        <div className="cards">
          <div className="card d1">
            <h3>Similarity model (analytic)</h3>
            <span className="n">0.801</span>
            <p>An explicit formula: centre each film on its peer mean, add similarity-weighted deviations.</p>
          </div>
          <div className="card d2">
            <h3>XGBoost</h3>
            <span className="n">0.783</span>
            <p>Gradient-boosted trees over 116 engineered features, including per-genre/theme/actor/director taste affinities.</p>
          </div>
          <div className="card d3">
            <h3>Neural network</h3>
            <span className="n">0.789</span>
            <p>A residual MLP ensemble trained on the identical features.</p>
          </div>
        </div>
        <p className="muted small" style={{ marginTop: 14 }}>
          Full-history RMSE, Rotten Tomatoes (lower is better). Every variant runs live in the app tab, entirely
          in your browser.
        </p>
      </section>

      <section>
        <p className="kicker">The headline</p>
        <h2>Personalization vs. the flat average</h2>
        <div className="callout">
          <p style={{ marginTop: 0, marginBottom: 0 }}>
            Against the flat baseline of just quoting each film's mean reviewer score, the best model is{" "}
            <b>7.1% more accurate</b> on Rotten Tomatoes (RMSE 0.783 vs. 0.843) and <b>14.3% more accurate</b>{" "}
            on Letterboxd (1.417 vs. 1.654).
          </p>
        </div>
        <figure>
          <img src={`${BASE}assets/plotA_rmse_vs_n.png`} alt="RMSE versus the number of films you have rated, for each method" />
          <figcaption>
            Rotten Tomatoes: the three models against the flat baseline. The trained models beat it at every
            seen-count; the analytic formula overtakes it once ~20 films are rated.
          </figcaption>
        </figure>
        <figure>
          <img
            src={`${BASE}assets/letterboxd_plotA_rmse_vs_n.png`}
            alt="Letterboxd RMSE versus the number of films rated, for each method"
          />
          <figcaption>Letterboxd: the same sweep, styled identically — the gain over the baseline is larger here.</figcaption>
        </figure>
      </section>

      <section>
        <p className="kicker">The honest findings</p>
        <h2>What the measurement actually says</h2>
        <div className="callout">
          <p style={{ marginTop: 0 }}>
            <b>It is mostly calibration, not taste-matching.</b> Most of the gain is knowing whether you rate
            high or low; similarity-based taste-matching — the project's whole premise — adds comparatively
            little on top, and the z-score track (which strips your level out) confirms it: even where z helps,
            the gain is under 1%.
          </p>
          <p>
            <b>More real data did not mean lower error.</b> Normalized by rating range, the trained models land
            at essentially identical error on both datasets (~0.157–0.158) — Letterboxd's dense real histories
            did not beat sparse critic pseudo-users.
          </p>
          <p style={{ marginBottom: 0 }}>
            <b>Negative results, reported as found:</b> restricting the formula to your 10 most-aligned peers
            never beats the full neighbourhood, and rating a film's nearest content neighbours first does not
            measurably sharpen its prediction.
          </p>
        </div>
        <figure>
          <img src={`${BASE}assets/plotB_gain_vs_dispersion.png`} alt="RMSE improvement over the Tomatometer by how much critics disagree" />
          <figcaption>Where personalization pays, split by how much critics disagree about a film.</figcaption>
        </figure>
      </section>

      <section>
        <p className="kicker">Try it</p>
        <h2>The interactive app</h2>
        <p>
          Rate films with clickable stars (or a 1–10 stepper on Letterboxd) and watch every model variant
          predict the rest live in your browser — no server. If you score your prediction films, the app tells
          you <b>which method is closest to your own taste</b>.
        </p>
        <div className="cta">
          <button className="btn primary" onClick={onOpenApp}>
            Open the app →
          </button>
        </div>
      </section>

      <footer>
        <p>
          Built from ~1.4M Rotten Tomatoes critic reviews and 11.08M Letterboxd member ratings (both Kaggle).
          Every model runs as a client-side TypeScript port of the original Python inference code — an XGBoost
          tree-walker, a hand-written neural-net forward pass, and the movie-facet affinity computation. Full
          methodology and results: <a href={`${BASE}assets/palate-report.pdf`}>the report</a> ·{" "}
          <a href="https://github.com/ap-gautham/palate">github.com/ap-gautham/palate</a>
        </p>
      </footer>
    </div>
  );
}
