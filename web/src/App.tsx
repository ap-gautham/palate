import { useEffect, useState } from "react";
import { useAppData } from "./lib/useAppData";
import { About } from "./pages/About";
import { PredictApp } from "./pages/PredictApp";
import "./App.css";

type Tab = "about" | "app";
type Project = "rotten-tomatoes" | "letterboxd";

function tabFromHash(): Tab {
  return window.location.hash === "#about" ? "about" : "app";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash());
  const [project, setProject] = useState<Project>("rotten-tomatoes");
  const state = useAppData();

  useEffect(() => {
    const onHashChange = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function go(next: Tab) {
    window.location.hash = next === "app" ? "#app" : "#about";
    setTab(next);
  }

  return (
    <div className="site">
      <nav className="topnav">
        <div className="topnav-inner">
          <button className="brand" type="button" onClick={() => go("app")} aria-label="Open the app">
            Palate<span className="dot">.</span>
          </button>
          <div className="tabs">
            <button className={"tab" + (tab === "about" ? " active" : "")} onClick={() => go("about")}>
              About
            </button>
            <button className={"tab" + (tab === "app" ? " active" : "")} onClick={() => go("app")}>
              🎬 App
            </button>
          </div>
        </div>
      </nav>

      <main className="site-main">
        {tab === "about" && <About onOpenApp={() => go("app")} />}
        {tab === "app" && (
          <>
            <div className="project-switcher" role="group" aria-label="Rating dataset">
              <span>Dataset</span>
              <button className={project === "rotten-tomatoes" ? "active" : ""} onClick={() => setProject("rotten-tomatoes")}>Rotten Tomatoes</button>
              <button className={project === "letterboxd" ? "active" : ""} onClick={() => setProject("letterboxd")}>Letterboxd</button>
            </div>
            {project === "rotten-tomatoes" ? <AppTab state={state} /> : <LetterboxdTab />}
          </>
        )}
      </main>
    </div>
  );
}

function LetterboxdTab() {
  return (
    <section className="project-pending">
      <p className="kicker">Letterboxd project</p>
      <h1>Community ratings on a 1–10 scale</h1>
      <p>
        This is a separate recommendation pipeline: Letterboxd members replace critics, and half-star ratings are
        represented directly on a 1–10 scale. The downloaded export contains 7,420 eligible members, 286,069 films,
        and 11.08 million ratings after the five-rating quality floor.
      </p>
      <div className="cards project-results">
        <div className="card d1"><h3>Design 1: analytic</h3><span className="n">1.695</span><p>RMSE, 300 deterministic held-out profiles with 50 seen films.</p></div>
        <div className="card d2"><h3>Design 2: XGBoost</h3><span className="n">1.628</span><p>RMSE on 1,435 member-disjoint held-out ratings.</p></div>
        <div className="card d3"><h3>Design 3: neural net</h3><span className="n">Not trained</span><p>The architecture is written but intentionally has not been run.</p></div>
      </div>
      <p className="info-box">
        The analysis uses the full local rating matrix. An interactive browser catalog needs a separate compact export
        of these Letterboxd assets, so Rotten Tomatoes remains the live interactive demo for now.
      </p>
    </section>
  );
}

function AppTab({ state }: { state: ReturnType<typeof useAppData> }) {
  if (state.status === "loading") {
    return (
      <div className="loading-screen">
        <div className="spinner" />
        <p>{state.progress}</p>
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="loading-screen">
        <p className="warn-box">Failed to load: {state.message}</p>
      </div>
    );
  }
  return <PredictApp data={state.data} />;
}
