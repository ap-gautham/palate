import { useEffect, useState } from "react";
import { useAppData } from "./lib/useAppData";
import { About } from "./pages/About";
import { PredictApp } from "./pages/PredictApp";
import "./App.css";

type Tab = "about" | "app";

function tabFromHash(): Tab {
  return window.location.hash === "#app" ? "app" : "about";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash());
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
          <span className="brand">
            Palate<span className="dot">.</span>
          </span>
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
        {tab === "app" && <AppTab state={state} />}
      </main>
    </div>
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
