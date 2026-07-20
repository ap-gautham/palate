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

MIN_SEEN = 5
MIN_OVERLAP = d1.MIN_OVERLAP
DEFAULT_STARS = 2          # feedback index 2 -> 3 stars
GOLD = {1: "#c9a24a", 2: "#d8a838", 3: "#e8ab20", 4: "#f4ae0c", 5: "#ffbf00"}

st.set_page_config(page_title="Critic Match", page_icon="🎬", layout="wide")


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
