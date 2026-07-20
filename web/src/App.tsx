import { useEffect, useState } from "react";
import { useAppData } from "./lib/useAppData";
import { useLetterboxdData } from "./lib/letterboxd/useLetterboxdData";
import { About } from "./pages/About";
import { PredictApp } from "./pages/PredictApp";
import { LetterboxdApp } from "./pages/LetterboxdApp";
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
  // Only fetches the ~60MB Letterboxd export once this tab actually mounts
  // (i.e. the user has switched the dataset to Letterboxd).
  const state = useLetterboxdData(true);
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
  return <LetterboxdApp data={state.data} />;
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
