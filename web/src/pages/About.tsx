const BASE = import.meta.env.BASE_URL;

export function About({ onOpenApp }: { onOpenApp: () => void }) {
  return (
    <div className="about-page">
      <header className="about-header">
        <p className="lead">
          Palate predicts the rating <b>you</b> would give a film from the scores of other raters — matching you to
          the critics or members whose taste tracks yours. It is really a measurement project:{" "}
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
          Rotten Tomatoes has no real users, so each of ~2,900 professional critics becomes a stand-in: some of
          their ratings are "films they have seen," one held-out film is the target. Letterboxd has real people
          instead — 7,420 members with genuine, dense rating histories. The exact same three-design pipeline runs
          on both, fully isolated (no shared code, data, or models), so the comparison is symmetric.
        </p>
        <ul className="funnel">
          <li>
            <span>Rotten Tomatoes: raw critic reviews</span>
            <span>1,444,963</span>
          </li>
          <li>
            <span>After dropping low-volume critics</span>
            <span>4,393 critics</span>
          </li>
          <li>
            <span>Pseudo-users (≥10 rated films)</span>
            <span>2,894 critics</span>
          </li>
          <li>
            <span>Letterboxd: member ratings</span>
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
        <p className="muted small" style={{ marginTop: 16 }}>
          Every Rotten Tomatoes score — 4-star, letter grade, percentage — is standardized onto one 0–5 scale first.
          Both catalogs are also joined to a third-party movie-metadata dump (genre, theme, studio, director, actor,
          decade, language, country) so the models can learn per-user taste in those facets too — not just "this
          critic and I agree," but "this user likes A24 films" or "this user over-rates 90s comedies."
        </p>
        <figure>
          <img src={`${BASE}assets/score_distributions.png`} alt="Distribution of standardized critic scores" />
          <figcaption>Heterogeneous critic scales collapsed onto six clean levels.</figcaption>
        </figure>
      </section>

      <section>
        <p className="kicker">Three designs, two tracks each</p>
        <h2>How the predictions are made</h2>
        <p>
          Three designs compete on identical held-out films in both datasets. Every design also has a second,
          parallel <b>z-score</b> variant — instead of predicting your raw score, it predicts your deviation from
          your own average, then converts back — a direct test of whether personalization is really about matching
          taste, or just knowing whether you rate high or low.
        </p>
        <div className="cards">
          <div className="card d1">
            <h3>Analytic formula</h3>
            <span className="n">0.801</span>
            <p>
              An explicit rule: centre each film on its peer mean, then add similarity-weighted, magnitude-scaled
              deviations. A second variant restricts this to your 10 most strongly aligned <em>or</em>{" "}
              anti-aligned peers instead of the full neighbourhood.
            </p>
          </div>
          <div className="card d2">
            <h3>XGBoost</h3>
            <span className="n">0.783</span>
            <p>
              Gradient-boosted trees over 116 engineered features: similarity deciles, consensus, dispersion, your
              average, and per-genre, per-theme, per-actor, and per-director taste-affinity blocks built from a
              movie-metadata join.
            </p>
          </div>
          <div className="card d3">
            <h3>Neural network</h3>
            <span className="n">0.789</span>
            <p>A residual MLP ensemble trained on the GPU over the identical features (no separate genre embedding -- the per-genre affinity block already gives it that information directly).</p>
          </div>
        </div>
        <p className="muted small" style={{ marginTop: 14 }}>
          Full-history RMSE, Rotten Tomatoes, raw track (lower is better). Every variant above — both tracks, all
          four methods — runs live in the app tab, entirely in your browser.
        </p>
      </section>

      <section>
        <p className="kicker">The headline</p>
        <h2>Personalization vs. the flat average</h2>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>method</th>
                <th>3</th>
                <th>10</th>
                <th>50</th>
                <th>all</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Tomatometer → score</td>
                <td>0.853</td>
                <td>0.853</td>
                <td>0.853</td>
                <td>0.853</td>
              </tr>
              <tr>
                <td>Mean of all reviewers</td>
                <td>0.843</td>
                <td>0.843</td>
                <td>0.843</td>
                <td>0.843</td>
              </tr>
              <tr>
                <td>Analytic formula</td>
                <td>0.971</td>
                <td>0.874</td>
                <td>0.812</td>
                <td>0.801</td>
              </tr>
              <tr>
                <td>Analytic, top-|sim| variant</td>
                <td>0.993</td>
                <td>0.902</td>
                <td>0.835</td>
                <td>0.811</td>
              </tr>
              <tr>
                <td>XGBoost</td>
                <td>0.830</td>
                <td>0.812</td>
                <td>0.793</td>
                <td className="best">0.783</td>
              </tr>
              <tr>
                <td>Neural network</td>
                <td>0.831</td>
                <td>0.815</td>
                <td>0.794</td>
                <td className="best">0.789</td>
              </tr>
            </tbody>
          </table>
        </div>
        <figure>
          <img src={`${BASE}assets/plotA_rmse_vs_n.png`} alt="RMSE versus the number of films you have rated, for each method" />
          <figcaption>
            The trained models beat every flat baseline once a profile exists. The top-|sim| analytic variant
            trails the full formula slightly at every seen-count — a negative result, reported as found.
          </figcaption>
        </figure>
      </section>

      <section>
        <p className="kicker">The honest finding</p>
        <h2>It is mostly calibration, not taste-matching</h2>
        <div className="callout">
          <p style={{ marginTop: 0 }}>
            Most of the gain over the flat average is <b>knowing whether you rate high or low</b> — anchoring the
            consensus on your own average. An attribution test shows that similarity-based <b>taste-matching</b>,
            the project's whole premise, adds comparatively little on top. Adding real movie-facet features (per-genre,
            theme, actor, and director affinities) shifts XGBoost's error by well under 0.01 — a small effect on
            top of calibration, not a replacement for it.
          </p>
          <p style={{ marginBottom: 0 }}>
            The top-|sim| analytic variant — restricting to your 10 most strongly aligned or anti-aligned
            peers — was tried as a way to sharpen the taste-matching signal. It trails the full formula on Rotten
            Tomatoes and only manages a dead heat on Letterboxd. Reported honestly rather than tuned to manufacture
            a win.
          </p>
        </div>
        <figure>
          <img src={`${BASE}assets/plotB_gain_vs_dispersion.png`} alt="RMSE improvement over the Tomatometer by how much critics disagree" />
          <figcaption>Where personalization pays, split by how much critics disagree about a film.</figcaption>
        </figure>
      </section>

      <section>
        <p className="kicker">Second dataset</p>
        <h2>Does real, dense rating data actually do better?</h2>
        <p>
          Critics are a sparse proxy for real people. The identical pipeline, run on 7,420 real Letterboxd members
          with dense rating histories (native 1–10 scale, no Tomatometer), tests the hypothesis directly. Switch
          datasets at the top of the app tab to try it live.
        </p>
        <div className="cards">
          <div className="card d1">
            <h3>Analytic formula</h3>
            <span className="n">1.502</span>
            <p>Same movie-mean-centred, magnitude-scaled formula (and top-|sim| variant) as Rotten Tomatoes.</p>
          </div>
          <div className="card d2">
            <h3>XGBoost</h3>
            <span className="n">1.417</span>
            <p>Same feature contract as Rotten Tomatoes' Design 2, minus the Tomatometer feature.</p>
          </div>
          <div className="card d3">
            <h3>Neural network</h3>
            <span className="n">1.418</span>
            <p>Same residual-MLP architecture — genuinely trained, not a placeholder.</p>
          </div>
        </div>
        <p className="muted small" style={{ marginTop: 14 }}>
          Full-history RMSE on the 1–10 scale (lower is better). All variants run live in the app, exactly like
          Rotten Tomatoes.
        </p>
        <figure>
          <img
            src={`${BASE}assets/letterboxd_plotA_rmse_vs_n.png`}
            alt="Letterboxd RMSE versus the number of films rated, for each method"
          />
          <figcaption>
            The same seen-history sweep as the Rotten Tomatoes plot above, styled identically.
          </figcaption>
        </figure>
        <div className="callout" style={{ marginTop: 20 }}>
          <p style={{ marginTop: 0, marginBottom: 0 }}>
            <b>Honest cross-dataset finding:</b> raw RMSE isn't comparable across a 0–5 scale and a 1–10 scale, so
            normalizing by rating range (RMSE ÷ range) puts them on the same footing. Rotten Tomatoes normalizes to
            ~0.157–0.162; Letterboxd to ~0.157–0.167 — <b>comparable</b>, with the trained models landing at
            essentially identical normalized error on both, and the analytic formula slightly <i>better</i> on
            Rotten Tomatoes despite its far sparser critic pseudo-user profiles. More real rating data did{" "}
            <b>not</b> translate into a lower normalized error here. Letterboxd's genuine value is dense real-member
            histories, not a lower headline RMSE — reported as found, not tuned to confirm the hypothesis.
          </p>
        </div>
      </section>

      <section>
        <p className="kicker">Isolating scale</p>
        <h2>Is the gain calibration, or genuine taste-matching?</h2>
        <p>
          Both projects also train a parallel <b>z-score track</b>: every rater is standardized to their own scale
          (the model predicts pure <i>deviation</i>, not level), and every prediction is converted back to the raw
          scale before scoring. Both tracks run live in the app; the predictions table shows raw and z-score side
          by side and marks whichever is closest to your own score.
        </p>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>design</th>
                <th>RT raw</th>
                <th>RT z</th>
                <th>LB raw</th>
                <th>LB z</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Analytic formula</td>
                <td>0.801</td>
                <td>0.948</td>
                <td>1.502</td>
                <td>1.607</td>
              </tr>
              <tr>
                <td>Analytic, top-|sim| variant</td>
                <td>0.811</td>
                <td>0.844</td>
                <td>1.501</td>
                <td>1.501</td>
              </tr>
              <tr>
                <td>XGBoost</td>
                <td>0.783</td>
                <td>0.792</td>
                <td>1.417</td>
                <td className="best">1.412</td>
              </tr>
              <tr>
                <td>Neural network</td>
                <td>0.789</td>
                <td>0.793</td>
                <td>1.418</td>
                <td className="best">1.406</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="muted small" style={{ marginTop: 14 }}>
          Full-history RMSE, both tracks on their raw scale (lower is better).
        </p>
        <div className="callout" style={{ marginTop: 20 }}>
          <p style={{ marginTop: 0, marginBottom: 0 }}>
            The result is <b>not uniform</b>. On Rotten Tomatoes the z-track trails raw at every design — the
            full-neighbourhood formula loses the most, since it has no learned capacity to reallocate once the
            level is removed. On Letterboxd the trained models come out <i>flat-to-slightly-better</i> in
            z-space, and even the top-|sim| analytic variant is nearly indifferent. Even where z helps, the gain is
            under 1% — an order of magnitude smaller than raw calibration's gain over the flat baselines above.
            Calibration, not taste-matching, remains the dominant effect. Reported as found, not tuned to confirm
            either hypothesis.
          </p>
        </div>
      </section>

      <section>
        <p className="kicker">Try it</p>
        <h2>The interactive app</h2>
        <p>
          Rate films with clickable stars (or a 1–10 stepper on Letterboxd) and watch all four model variants —
          both tracks — predict the rest live, right here in the browser, no server involved. Filter the search
          list by genre, and if you score your prediction films, the app tells you{" "}
          <b>which method is closest to your own taste</b>.
        </p>
        <div className="cta">
          <button className="btn primary" onClick={onOpenApp}>
            Open the app →
          </button>
        </div>
      </section>

      <footer>
        <p>
          Built from ~1.4M Rotten Tomatoes critic reviews and 11.08M Letterboxd member ratings (both Kaggle), plus
          a third-party movie-metadata dump for genre/theme/studio/cast facets. Each project is a self-contained,
          symmetric Python package; all results regenerate from fixed seeds. Every model on this page runs as a
          client-side TypeScript port (an XGBoost tree-walker, a hand-written neural-net forward pass, and the
          movie-facet affinity computation) of the original Python inference code, for both datasets.{" "}
          <a href="https://github.com/ap-gautham/palate">github.com/ap-gautham/palate</a>
        </p>
      </footer>
    </div>
  );
}
