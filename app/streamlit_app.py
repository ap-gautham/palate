"""Critic Match: rate films with clickable stars, then compare the analytic
formula, XGBoost, and neural-net predictions -- and see which is closest to you.

Run: .venv/bin/streamlit run app/streamlit_app.py
Requires the catalog from app_catalog.export and the saved models from
design2_xgboost.train and design3_neural.train.
"""
import json
import os
import sys
from pathlib import Path

# XGBoost and PyTorch each bundle their own libomp; on macOS importing both in
# one process aborts with OpenMP "Error #15" and running both multi-threaded
# deadlocks. Allow the duplicate runtime and pin each to a single thread.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "results" / "models"
sys.path.insert(0, str(ROOT / "src"))

from design1_analytic import predict as d1
from design2_xgboost import predict as d2
from design3_neural import predict as d3

XGB_PATH = MODELS / "design2_xgboost.json"
NN_PATH = MODELS / "design3_mlp.pt"
K_STAR_PATH = ROOT / "results" / "tables" / "k_star.json"
LB_DATA = ROOT / "data" / "letterboxd" / "processed"
LB_MODELS = ROOT / "results" / "letterboxd" / "models"
LB_XGB_PATH = LB_MODELS / "letterboxd_xgboost.json"

MIN_SEEN = 5
MIN_OVERLAP = d1.MIN_OVERLAP
DEFAULT_STARS = 2          # feedback index 2 -> 3 stars
GOLD = {1: "#c9a24a", 2: "#d8a838", 3: "#e8ab20", 4: "#f4ae0c", 5: "#ffbf00"}

st.set_page_config(page_title="Critic Match", page_icon="🎬", layout="wide")


@st.cache_data
def load_letterboxd_catalog():
    """A 1,000-film interactive slice backed by the full processed matrix."""
    ratings = pd.read_parquet(LB_DATA / "ratings_1_to_10.parquet")
    popularity = ratings.groupby("movie_id").size().nlargest(1_000)
    ratings = ratings[ratings["movie_id"].isin(popularity.index)].copy()
    movies = pd.read_parquet(LB_DATA / "movies.parquet").drop_duplicates("movie_id").set_index("movie_id")
    meta = movies.reindex(popularity.index)[["title", "year"]].copy()
    meta["n_scores"] = popularity
    meta["title"] = meta["title"].fillna(meta.index.to_series())
    return ratings, meta


@st.cache_resource
def load_letterboxd_xgb():
    if not LB_XGB_PATH.exists():
        return None
    model = xgb.Booster()
    model.load_model(LB_XGB_PATH)
    return model


def lb_matches(scores: pd.DataFrame, user: pd.Series, k_shrink: int = 8) -> pd.DataFrame:
    """Same shrunk Pearson + magnitude contract as RT Design 1."""
    overlap = scores[scores["movie_id"].isin(user.index)].copy()
    overlap["user_rating"] = overlap["movie_id"].map(user)
    grouped = overlap.groupby("user_id")
    out = grouped.agg(overlap=("rating", "size"), sx=("user_rating", "sum"),
                      sy=("rating", "sum"), sxx=("user_rating", lambda x: float(np.dot(x, x))),
                      syy=("rating", lambda x: float(np.dot(x, x))),
                      sxy=("rating", lambda x: 0.0))
    # Dot products require aligned columns, so compute them before aggregation.
    overlap["xy"] = overlap["user_rating"] * overlap["rating"]
    out["sxy"] = overlap.groupby("user_id")["xy"].sum()
    n = out["overlap"].to_numpy(float)
    cov = out["sxy"].to_numpy() - out["sx"].to_numpy() * out["sy"].to_numpy() / n
    vx = out["sxx"].to_numpy() - out["sx"].to_numpy() ** 2 / n
    vy = out["syy"].to_numpy() - out["sy"].to_numpy() ** 2 / n
    corr = np.divide(cov, np.sqrt(vx * vy), out=np.zeros_like(cov), where=(n >= 2) & (vx > 1e-9) & (vy > 1e-9))
    out["sim"] = corr * np.minimum(n, k_shrink) / k_shrink
    out["mag_sim"] = np.divide(out["sxy"], out["syy"], out=np.ones(len(out)), where=out["syy"] > 1e-12)
    return out[["overlap", "sim", "mag_sim"]]


def run_letterboxd_app():
    if not (LB_DATA / "ratings_1_to_10.parquet").exists():
        st.error("Letterboxd data is not available. Run `python -m letterboxd.preprocess` first.")
        return
    scores, meta = load_letterboxd_catalog()
    model = load_letterboxd_xgb()
    st.title("🎬 Community Match — Letterboxd")
    st.caption("The same three-design workflow as Rotten Tomatoes, using direct 1–10 member ratings. "
               "Design 3 is intentionally shown as untrained.")
    st.info("Full analysis: 7,420 members · 286,069 films · 11.08M ratings. The interactive catalog contains the 1,000 most-rated films.")
    label = {mid: f"{row.title} ({int(row.year) if pd.notna(row.year) else 'n/a'})" for mid, row in meta.iterrows()}
    sort = st.radio("Sort the search list by", ["Title (A–Z)", "Year (newest first)", "Most rated"], horizontal=True, key="lb_sort")
    if sort == "Title (A–Z)": ordered = list(meta.sort_values("title", key=lambda s: s.str.casefold()).index)
    elif sort == "Year (newest first)": ordered = list(meta.sort_values("year", ascending=False, na_position="last").index)
    else: ordered = list(meta.sort_values("n_scores", ascending=False).index)
    for key in ("lb_seen", "lb_predict"):
        st.session_state.setdefault(key, [])
    def add_lb(which):
        mid = st.session_state.get(f"{which}_add")
        if mid and mid not in st.session_state["lb_seen"] and mid not in st.session_state["lb_predict"]:
            st.session_state[which].append(mid)
            if which == "lb_seen": st.session_state[f"lb_rating_{mid}"] = 5
    def chooser(which, heading):
        st.subheader(heading)
        chosen = set(st.session_state["lb_seen"]) | set(st.session_state["lb_predict"])
        st.selectbox("Add a film", [m for m in ordered if m not in chosen], index=None, key=f"{which}_add", on_change=add_lb, args=(which,), format_func=lambda mid: label[mid], placeholder="Type to search all catalog films…")
        for mid in list(st.session_state[which]):
            cols = st.columns([.6, 5, 2, 1])
            cols[0].button("✕", key=f"lb_rm_{which}_{mid}", on_click=lambda w=which, m=mid: st.session_state[w].remove(m))
            cols[1].write(label[mid])
            cols[2].number_input("Your rating", min_value=1, max_value=10, value=int(st.session_state.get(f"lb_rating_{mid}", 5)), step=1, key=f"lb_rating_{mid}", label_visibility="collapsed")
            cols[3].markdown(f"**{st.session_state.get(f'lb_rating_{mid}', '—')}/10**" if f"lb_rating_{mid}" in st.session_state else "not rated")
    chooser("lb_seen", "1. Films you have seen")
    chooser("lb_predict", "2. Films to predict")
    st.subheader("3. Predictions")
    user = pd.Series({mid: st.session_state[f"lb_rating_{mid}"] for mid in st.session_state["lb_seen"]}, dtype=float)
    targets = st.session_state["lb_predict"]
    if len(user) < MIN_SEEN: st.info(f"Rate at least {MIN_SEEN} seen films (currently {len(user)}).")
    elif user.std() < 1e-9: st.warning("Give your seen films different ratings so similarity is defined.")
    elif not targets: st.info("Add at least one film to predict.")
    else:
        matches = lb_matches(scores, user)
        rows = []
        for mid in targets:
            peer = scores[scores.movie_id == mid].join(matches, on="user_id")
            mean, std, count = peer.rating.mean(), peer.rating.std(ddof=0), len(peer)
            weight = peer.sim.abs().fillna(0)
            analytic = mean if weight.sum() == 0 else float((((weight * mean + peer.sim.fillna(0) * (peer.rating - mean)) * peer.mag_sim.fillna(1)).sum()) / weight.sum())
            xgb_pred = np.nan
            if model is not None:
                features = pd.DataFrame([[user.mean(), len(user), mean, std, count]], columns=["user_mean", "user_count", "movie_mean", "movie_std", "movie_count"])
                xgb_pred = float(np.clip(model.predict(xgb.DMatrix(features))[0], 1, 10))
            rows.append({"Film": label[mid], "Analytic": round(float(np.clip(analytic, 1, 10)), 2), "XGBoost": round(xgb_pred, 2), "Neural net": "Not trained", "Consensus (mean)": round(mean, 2), "Your score": st.session_state.get(f"lb_rating_{mid}")})
        table = pd.DataFrame(rows).set_index("Film")
        if table["Your score"].isna().all(): table = table.drop(columns="Your score")
        st.dataframe(table, width="stretch")
        st.caption("Design 1 uses the same shrunken Pearson/magnitude formula as Rotten Tomatoes. Design 2 uses the separately trained Letterboxd XGBoost model; Design 3 is deliberately untrained.")


dataset = st.sidebar.radio("Project", ["Rotten Tomatoes", "Letterboxd"], index=0)
if dataset == "Letterboxd":
    run_letterboxd_app()
    st.stop()


def mse(pred, truth) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    return float(np.mean((pred[ok] - truth[ok]) ** 2)) if ok.any() else float("nan")


@st.cache_data
def load_catalog():
    scores = pd.read_parquet(DATA / "demo_scores.parquet")
    critics = pd.read_parquet(DATA / "demo_critics.parquet").set_index("critic_id")
    meta = (scores.drop_duplicates("movie_id")
            .set_index("movie_id")[["title", "year", "tomatoMeter"]])
    meta["n_scores"] = scores.groupby("movie_id").size()
    k_shrink = 8
    if K_STAR_PATH.exists():
        k_shrink = int(json.loads(K_STAR_PATH.read_text())["k_star"])
    return scores, critics, meta, k_shrink


@st.cache_resource
def load_xgb():
    return d2.load_model(XGB_PATH) if XGB_PATH.exists() else None


@st.cache_resource
def load_nn():
    return d3.load_checkpoint(NN_PATH) if NN_PATH.exists() else None


scores, critics, movie_meta, K_SHRINK = load_catalog()
xgb_model = load_xgb()
nn_ckpt = load_nn()
LABEL = {mid: f"{r['title']} ({int(r['year']) if pd.notna(r['year']) else 'n/a'})"
         for mid, r in movie_meta.iterrows()}

SORT_ORDERS = {
    "Title (A–Z)": lambda m: m.sort_values(
        "title", key=lambda s: s.str.casefold()),
    "Year (newest first)": lambda m: m.sort_values(
        "year", ascending=False, na_position="last"),
    "Most reviewed": lambda m: m.sort_values("n_scores", ascending=False),
}

for key in ("seen", "predict"):
    st.session_state.setdefault(key, [])


def add_movie(list_key: str):
    search_key = f"{list_key}_search"
    mid = st.session_state.get(search_key)
    if mid is not None and mid not in st.session_state["seen"] \
            and mid not in st.session_state["predict"]:
        st.session_state[list_key].append(mid)
        if list_key == "seen":
            st.session_state[f"seen_star_{mid}"] = DEFAULT_STARS
    st.session_state[search_key] = None


def remove_movie(list_key: str, mid):
    if mid in st.session_state[list_key]:
        st.session_state[list_key].remove(mid)


def star_value(list_key: str, mid, default=None):
    """Current 1-5 rating from a feedback widget, or `default` if unrated."""
    idx = st.session_state.get(f"{list_key}_star_{mid}")
    return default if idx is None else int(idx) + 1


def rating_pill(rating):
    if rating is None:
        return "<span style='color:#9a9a9a'>not rated</span>"
    return (f"<span style='color:{GOLD[rating]};font-weight:700;"
            f"font-size:1.05rem'>{rating}/5</span>")


def film_search(list_key: str, prompt: str, ordered_ids):
    chosen = set(st.session_state["seen"]) | set(st.session_state["predict"])
    options = [m for m in ordered_ids if m not in chosen]
    st.selectbox(prompt, options=options, index=None, key=f"{list_key}_search",
                 on_change=add_movie, args=(list_key,),
                 format_func=lambda m: LABEL[m],
                 placeholder="Type to search the catalog...")


def rating_table(list_key: str, seen_defaults: bool):
    """Render one row per chosen film: remove, title, clickable stars, rating."""
    ids = st.session_state[list_key]
    if not ids:
        st.caption("No films added yet.")
        return
    head = st.columns([0.6, 5, 3, 1.4])
    head[1].markdown("**Film**")
    head[2].markdown("**Your rating**")
    head[3].markdown("**Score**")
    for mid in list(ids):
        cols = st.columns([0.6, 5, 3, 1.4], vertical_alignment="center")
        cols[0].button("✕", key=f"rm_{list_key}_{mid}", on_click=remove_movie,
                       args=(list_key, mid), help="Remove")
        cols[1].write(LABEL[mid])
        cols[2].feedback("stars", key=f"{list_key}_star_{mid}")
        rating = star_value(list_key, mid, default=3 if seen_defaults else None)
        cols[3].markdown(rating_pill(rating), unsafe_allow_html=True)


st.title("🎬 Critic Match")
st.caption(
    "Rate films you have seen, then predict your score for others three ways: "
    "the analytic critic formula, an XGBoost model, and a neural network. "
    "Score the films you predict to see which method is closest to your taste.")

sort_choice = st.radio("Sort the search list by", list(SORT_ORDERS),
                       horizontal=True, key="sort_choice")
ORDERED_IDS = list(SORT_ORDERS[sort_choice](movie_meta).index)

# ---- 1. Films you have seen ------------------------------------------------
st.header("1. Films you have seen")
film_search("seen", "Add a film you have seen", ORDERED_IDS)
rating_table("seen", seen_defaults=True)

# ---- 2. Films to predict ---------------------------------------------------
st.header("2. Films to predict")
st.caption("Scoring these is optional. If you rate them, each method's error "
           "against your own score is reported below.")
film_search("predict", "Add a film to predict", ORDERED_IDS)
rating_table("predict", seen_defaults=False)

# ---- 3. Predictions --------------------------------------------------------
st.header("3. Predictions")

seen_ids = st.session_state["seen"]
predict_ids = st.session_state["predict"]
user = pd.Series({mid: star_value("seen", mid, default=3) for mid in seen_ids},
                 dtype=float)

if len(user) < MIN_SEEN:
    st.info(f"Rate at least {MIN_SEEN} seen films to build a taste profile "
            f"(currently {len(user)}).")
elif user.std() < 1e-9:
    st.warning("Give your seen films different ratings so similarity is defined.")
elif not predict_ids:
    st.info("Add at least one film to predict in section 2.")
else:
    target_scores = scores[scores["movie_id"].isin(predict_ids)]
    matches = d1.critic_matches(scores, user, K_SHRINK)     # Design 1 formula + display
    formula = d1.predict(target_scores, matches)
    xgb = d2.predict(scores, user, target_scores, critics, K_SHRINK, xgb_model) \
        if xgb_model is not None else pd.Series(np.nan, index=predict_ids)
    nn = d3.predict(scores, user, target_scores, critics, K_SHRINK, nn_ckpt) \
        if nn_ckpt is not None else pd.Series(np.nan, index=predict_ids)
    means = formula["movie_mean"]
    your = pd.Series({mid: star_value("predict", mid) for mid in predict_ids}, dtype=float)

    table = pd.DataFrame({
        "Film": [LABEL[m] for m in predict_ids],
        "Analytic": formula["prediction"].reindex(predict_ids).round(2).values,
        "XGBoost": xgb.reindex(predict_ids).round(2).values,
        "Neural net": nn.reindex(predict_ids).round(2).values,
        "Consensus (mean)": means.reindex(predict_ids).round(2).values,
        "Tomatometer": movie_meta["tomatoMeter"].reindex(predict_ids).values,
        "Your score": your.reindex(predict_ids).values,
    }).set_index("Film")
    if your.notna().sum() == 0:
        table = table.drop(columns=["Your score"])
    st.dataframe(table, width="stretch")

    # Which predictor is closest to the user's OWN score, over the films they
    # actually rated in section 2.
    truth = your.reindex(predict_ids).to_numpy()
    n_scored = int(np.isfinite(truth).sum())
    st.subheader("Which method predicts *you* best?")
    if n_scored == 0:
        st.info("Rate one or more of your predict films (section 2) to see the "
                "mean squared error of each method against your own score.")
    else:
        candidates = [
            ("Analytic formula", formula["prediction"].reindex(predict_ids)),
            ("XGBoost", xgb.reindex(predict_ids)),
            ("Neural net", nn.reindex(predict_ids)),
            ("Consensus mean", means.reindex(predict_ids)),
        ]
        rows = [(name, mse(series.to_numpy(), truth)) for name, series in candidates]
        best = min(rows, key=lambda r: r[1] if np.isfinite(r[1]) else np.inf)[0]
        st.caption(f"MSE against your own score over {n_scored} rated film(s). "
                   f"Lower is closer to your taste — **{best}** is closest here.")
        mse_cols = st.columns(len(rows))
        for col, (name, value) in zip(mse_cols, rows):
            col.metric(f"{name}{' ✅' if name == best else ''}",
                       f"{value:.3f}" if np.isfinite(value) else "—")

    with st.expander("Your closest critic matches"):
        visible = matches[(matches["overlap"] >= MIN_OVERLAP) & (matches["sim"] != 0)]
        top = visible.sort_values("sim", ascending=False).head(12).join(critics)
        if top.empty:
            st.write("No overlapping critic yet; predictions fall back to consensus.")
        else:
            st.dataframe(pd.DataFrame({
                "Critic": top.index,
                "Publication": top["publicationName"],
                "Alignment": top["sim"].round(2),
                "Scale match": top["mag_sim"].round(2),
                "Films in common": top["overlap"],
            }).set_index("Critic"), width="stretch")
