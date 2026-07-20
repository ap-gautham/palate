const BASE = import.meta.env.BASE_URL;

export function About({ onOpenApp }: { onOpenApp: () => void }) {
  return (
    <div className="about-page">
      <header className="about-header">
        <p className="lead">
          Palate predicts the rating <b>you</b> would give a film from the scores of professional critics —
          matching you to the reviewers whose taste tracks yours. It is really a measurement project:{" "}
          <b>where does personalization actually beat the flat critic average, and where does it just pretend to?</b>
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
          <a className="btn" href="https://github.com/ap-gautham/palate/blob/main/DOCUMENTATION.md">
            Technical docs
          </a>
        </div>
      </header>

      <section>
        <p className="kicker">The idea</p>
        <h2>Critics are the users</h2>
        <p>
          The data has no real users, so each of ~3,700 professional critics becomes a stand-in: some of their
          ratings are "films they have seen," one held-out film is the target, and the target is predicted from
          the rest of the critic pool. Three designs compete on the exact same held-out films:
        </p>
        <div className="cards">
          <div className="card d1">
            <h3>Analytic formula</h3>
            <span className="n">0.814</span>
            <p>
              An explicit rule: centre each film on its critic mean, then add similarity-weighted,
              magnitude-scaled deviations.
            </p>
          </div>
          <div className="card d2">
            <h3>XGBoost</h3>
            <span className="n">0.791</span>
            <p>
              Gradient-boosted trees over 38 engineered features (similarity deciles, consensus, dispersion, your
              average).
            </p>
          </div>
          <div className="card d3">
            <h3>Neural network</h3>
            <span className="n">0.793</span>
            <p>A residual MLP ensemble with a genre embedding, trained on the GPU over the identical features.</p>
          </div>
        </div>
        <p className="muted small" style={{ marginTop: 14 }}>
          Full-history RMSE on the standardized 0–5 rating scale (lower is better). Every model above runs live in
          the app tab, entirely in your browser.
        </p>
      </section>

      <section>
        <p className="kicker">The headline</p>
        <h2>Personalization vs. the flat critic average</h2>
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
                <td>0.841</td>
                <td>0.841</td>
                <td>0.841</td>
                <td>0.841</td>
              </tr>
              <tr>
                <td>Mean of all reviewers</td>
                <td>0.827</td>
                <td>0.827</td>
                <td>0.827</td>
                <td>0.827</td>
              </tr>
              <tr>
                <td>Analytic formula</td>
                <td>0.957</td>
                <td>0.869</td>
                <td>0.825</td>
                <td>0.814</td>
              </tr>
              <tr>
                <td>XGBoost</td>
                <td>0.815</td>
                <td>0.805</td>
                <td>0.796</td>
                <td className="best">0.791</td>
              </tr>
              <tr>
                <td>Neural network</td>
                <td>0.820</td>
                <td>0.809</td>
                <td>0.798</td>
                <td className="best">0.793</td>
              </tr>
            </tbody>
          </table>
        </div>
        <figure>
          <img src={`${BASE}assets/plotA_rmse_vs_n.png`} alt="RMSE versus the number of films you have rated, for each method" />
          <figcaption>
            The trained models beat every flat baseline once a profile exists; the analytic formula only overtakes
            the consensus after ~50 rated films. Baselines are flat because they do not depend on your history.
          </figcaption>
        </figure>
      </section>

      <section>
        <p className="kicker">The honest finding</p>
        <h2>It is calibration, not taste-matching</h2>
        <div className="callout">
          <p style={{ marginTop: 0 }}>
            The ~4% gain over the critic average is almost entirely <b>knowing whether you rate high or low</b> —
            anchoring the consensus on your own average. An attribution test shows that similarity-based{" "}
            <b>taste-matching</b>, the project's whole premise, adds essentially nothing on top.
          </p>
          <p style={{ marginBottom: 0 }}>
            Scaling the network wider, deeper, and on twice the data moved the loss by nothing: on these engineered
            features the ceiling is set by the information in the data, not model capacity. Reported honestly rather
            than tuned to manufacture a difference.
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
          Critics are a sparse proxy for real people. To test the hypothesis directly, the exact same
          three-design pipeline was rebuilt — fully isolated, same feature contract, same evaluation protocol —
          on <b>7,420 real Letterboxd members</b> with genuine dense rating histories (11.08M ratings, native
          1–10 scale, no Tomatometer). Switch datasets at the top of the app tab to try it live.
        </p>
        <div className="cards">
          <div className="card d1">
            <h3>Analytic formula</h3>
            <span className="n">1.507</span>
            <p>Same movie-mean-centred, magnitude-scaled formula as Rotten Tomatoes' Design 1.</p>
          </div>
          <div className="card d2">
            <h3>XGBoost</h3>
            <span className="n">1.501</span>
            <p>Same 37-feature contract as Design 2 (identical minus the Tomatometer feature).</p>
          </div>
          <div className="card d3">
            <h3>Neural network</h3>
            <span className="n">1.515</span>
            <p>Same residual-MLP architecture as Design 3 — genuinely trained, not a placeholder.</p>
          </div>
        </div>
        <p className="muted small" style={{ marginTop: 14 }}>
          Full-history RMSE on the 1–10 scale (lower is better). All three run live in the app, exactly like
          Rotten Tomatoes.
        </p>
        <figure>
          <img
            src={`${BASE}assets/letterboxd_plotA_rmse_vs_n.png`}
            alt="Letterboxd RMSE versus the number of films rated, for each method"
          />
          <figcaption>
            The same seen-history sweep as the Rotten Tomatoes plot above, styled identically. All three designs
            converge to ≈1.50–1.52 at full history.
          </figcaption>
        </figure>
        <div className="callout" style={{ marginTop: 20 }}>
          <p style={{ marginTop: 0, marginBottom: 0 }}>
            <b>Honest cross-dataset finding:</b> raw RMSE isn't comparable across a 0–5 scale and a 1–10 scale, so
            normalizing by rating range (RMSE ÷ range) puts them on the same footing. Rotten Tomatoes normalizes to
            ~0.158–0.163; Letterboxd normalizes to ~0.167–0.168 — <b>comparable</b>, with Rotten Tomatoes actually
            slightly <i>better</i> despite its far sparser critic pseudo-user profiles. More real rating data did{" "}
            <b>not</b> translate into a lower normalized error here. Letterboxd's genuine value is dense real-member
            histories, not a lower headline RMSE — reported as found, not tuned to confirm the hypothesis.
          </p>
        </div>
      </section>

      <section>
        <p className="kicker">The data</p>
        <h2>From 1.4M reviews to a usable matrix</h2>
        <ul className="funnel">
          <li>
            <span>Raw critic reviews</span>
            <span>1,444,963</span>
          </li>
          <li>
            <span>After dropping one-off critics</span>
            <span>4,393 critics</span>
          </li>
          <li>
            <span>Parsed &amp; standardized scores</span>
            <span>992,954</span>
          </li>
          <li>
            <span>Pseudo-users (≥10 rated films)</span>
            <span>3,704 critics</span>
          </li>
          <li>
            <span>Films in the interactive catalog</span>
            <span>1,000</span>
          </li>
        </ul>
        <p className="muted small" style={{ marginTop: 16 }}>
          Every score — 4-star, letter grade, percentage — is standardized onto one 0–5 scale before anything else.
        </p>
        <figure>
          <img src={`${BASE}assets/score_distributions.png`} alt="Distribution of standardized critic scores" />
          <figcaption>Heterogeneous critic scales collapsed onto six clean levels.</figcaption>
        </figure>
      </section>

      <section>
        <p className="kicker">Try it</p>
        <h2>The interactive app</h2>
        <p>
          Rate films with clickable stars and watch all three models predict the rest live — right here in the
          browser, no server involved. If you score your prediction films, it tells you{" "}
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
          Built from ~1.4M Rotten Tomatoes critic reviews and 11.08M Letterboxd member ratings (both Kaggle).
          Each project is a self-contained, symmetric Python package; all results regenerate from fixed seeds.
          Every model on this page runs as a client-side TypeScript port (an XGBoost tree-walker and a
          hand-written neural-net forward pass) of the original Python inference code, for both datasets.{" "}
          <a href="https://github.com/ap-gautham/palate">github.com/ap-gautham/palate</a>
        </p>
      </footer>
    </div>
  );
}
