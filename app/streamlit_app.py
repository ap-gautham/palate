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
DATA = ROOT / "data" / "rotten_tomatoes" / "processed"
MODELS = ROOT / "results" / "rotten_tomatoes" / "models"
sys.path.insert(0, str(ROOT / "src"))

from rotten_tomatoes.design1_analytic import predict as d1
from rotten_tomatoes.design2_xgboost import predict as d2
from rotten_tomatoes.design3_neural import predict as d3
from letterboxd import features as lbf
from letterboxd.analyze import load_nn as lb_load_nn, nn_predict as lb_nn_predict

XGB_PATH = MODELS / "design2_xgboost.json"
NN_PATH = MODELS / "design3_mlp.pt"
K_STAR_PATH = ROOT / "results" / "rotten_tomatoes" / "tables" / "k_star.json"
LB_DATA = ROOT / "data" / "letterboxd" / "processed"
LB_MODELS = ROOT / "results" / "letterboxd" / "models"
LB_XGB_PATH = LB_MODELS / "letterboxd_xgboost.json"

MIN_SEEN = 5
MIN_OVERLAP = d1.MIN_OVERLAP
DEFAULT_STARS = 2          # feedback index 2 -> 3 stars
GOLD = {1: "#c9a24a", 2: "#d8a838", 3: "#e8ab20", 4: "#f4ae0c", 5: "#ffbf00"}

st.set_page_config(page_title="Critic Match", page_icon="🎬", layout="wide")


def mse(pred, truth) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    return float(np.mean((pred[ok] - truth[ok]) ** 2)) if ok.any() else float("nan")


LB_MIN_OVERLAP = lbf.MIN_APP_OVERLAP


@st.cache_data
def load_letterboxd_catalog():
    """A 1,000-film interactive slice backed by the full processed matrix.

    `members` carries each member's ALL-TIME rating_sum/rating_count (not
    restricted to the 1,000-film catalog), matching the leave-one-out peer
    mean the models were trained on -- the same convention as RT's
    demo_critics.parquet.
    """
    ratings_full = pd.read_parquet(LB_DATA / "ratings_1_to_10.parquet")
    popularity = ratings_full.groupby("movie_id").size().nlargest(1_000)
    scores = ratings_full[ratings_full["movie_id"].isin(popularity.index)].copy()
    movies_full = pd.read_parquet(LB_DATA / "movies.parquet")
    movie_genre, _, unknown_genre_id = lbf.make_genre_maps(movies_full)
    movies_meta = movies_full.drop_duplicates("movie_id").set_index("movie_id")
    meta = movies_meta.reindex(popularity.index)[["title", "year"]].copy()
    meta["n_scores"] = popularity
    meta["title"] = meta["title"].fillna(meta.index.to_series())
    member_ids = pd.Index(scores["user_id"].drop_duplicates())
    members = (ratings_full[ratings_full["user_id"].isin(member_ids)]
               .groupby("user_id")["rating"].agg(["sum", "size"])
               .rename(columns={"sum": "rating_sum", "size": "rating_count"}))
    return scores, meta, members, movie_genre, unknown_genre_id


@st.cache_resource
def load_letterboxd_xgb():
    if not LB_XGB_PATH.exists():
        return None
    model = xgb.Booster()
    model.load_model(LB_XGB_PATH)
    return model


@st.cache_resource
def load_letterboxd_nn():
    return lb_load_nn()


def lb_analytic(target_scores: pd.DataFrame, matches: pd.DataFrame):
    """Movie-mean-centered, magnitude-scaled analytic prediction, clipped to
    [1, 10] -- the same formula as RT Design 1. Returns {movie_id: (pred, mean)}."""
    out = {}
    for mid, group in target_scores.groupby("movie_id"):
        sim = group["user_id"].map(matches["sim"]).fillna(0.0)
        mag = group["user_id"].map(matches["mag_sim"]).fillna(1.0)
        mean = group["rating"].mean()
        weight = sim.abs()
        num = ((weight * mean + sim * (group["rating"] - mean)) * mag).sum()
        pred = mean if weight.sum() == 0 else float(num / weight.sum())
        out[mid] = (float(np.clip(pred, 1, 10)), float(mean))
    return out


def run_letterboxd_app():
    if not (LB_DATA / "ratings_1_to_10.parquet").exists():
        st.error("Letterboxd data is not available. Run `python -m letterboxd.preprocess` first.")
        return
    scores, meta, members, movie_genre, unknown_genre_id = load_letterboxd_catalog()
    xgb_model = load_letterboxd_xgb()
    nn_ckpt = load_letterboxd_nn()
    st.title("🎬 Community Match — Letterboxd")
    st.caption("The same three-design workflow as Rotten Tomatoes -- an analytic member-match formula, "
               "an XGBoost model, and a neural network -- trained on direct 1–10 member ratings instead "
               "of critic pseudo-users, over the same 37-feature similarity-decile contract (no Tomatometer).")
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
            if which == "lb_seen": st.session_state[f"lb_rating_{mid}"] = 6
    def chooser(which, heading):
        st.subheader(heading)
        chosen = set(st.session_state["lb_seen"]) | set(st.session_state["lb_predict"])
        st.selectbox("Add a film", [m for m in ordered if m not in chosen], index=None, key=f"{which}_add", on_change=add_lb, args=(which,), format_func=lambda mid: label[mid], placeholder="Type to search all catalog films…")
        for mid in list(st.session_state[which]):
            cols = st.columns([.6, 5, 2, 1])
            cols[0].button("✕", key=f"lb_rm_{which}_{mid}", on_click=lambda w=which, m=mid: st.session_state[w].remove(m))
            cols[1].write(label[mid])
            cols[2].number_input("Your rating", min_value=1, max_value=10, value=int(st.session_state.get(f"lb_rating_{mid}", 6)), step=1, key=f"lb_rating_{mid}", label_visibility="collapsed")
            cols[3].markdown(f"**{st.session_state.get(f'lb_rating_{mid}', '—')}/10**" if f"lb_rating_{mid}" in st.session_state else "not rated")
    chooser("lb_seen", "1. Films you have seen")
    chooser("lb_predict", "2. Films to predict")
    st.subheader("3. Predictions")
    user = pd.Series({mid: st.session_state[f"lb_rating_{mid}"] for mid in st.session_state["lb_seen"]}, dtype=float)
    targets = st.session_state["lb_predict"]
    if len(user) < MIN_SEEN:
        st.info(f"Rate at least {MIN_SEEN} seen films to build a taste profile (currently {len(user)}).")
    elif user.std() < 1e-9:
        st.warning("Give your seen films different ratings so similarity is defined.")
    elif not targets:
        st.info("Add at least one film to predict in section 2.")
    else:
        target_scores = scores[scores["movie_id"].isin(targets)]
        matches = lbf.app_similarity(scores, user, lbf.K_SHRINK)
        analytic = lb_analytic(target_scores, matches)
        feats, feat_ids = lbf.app_features(target_scores, matches, members, user,
                                           movie_genre, unknown_genre_id)
        feats = feats[lbf.FEATURE_COLS]
        xgb_pred = (pd.Series(np.clip(xgb_model.predict(xgb.DMatrix(feats)), 1, 10), index=feat_ids)
                    if xgb_model is not None else pd.Series(np.nan, index=feat_ids))
        nn_pred = (pd.Series(lb_nn_predict(nn_ckpt, feats), index=feat_ids)
                   if nn_ckpt is not None else pd.Series(np.nan, index=feat_ids))
        means = pd.Series({mid: analytic.get(mid, (np.nan, np.nan))[1] for mid in targets})
        formula = pd.Series({mid: analytic.get(mid, (np.nan, np.nan))[0] for mid in targets})
        your = pd.Series({mid: st.session_state.get(f"lb_rating_{mid}") for mid in targets}, dtype=float)

        table = pd.DataFrame({
            "Film": [label[m] for m in targets],
            "Analytic": formula.reindex(targets).round(2).values,
            "XGBoost": xgb_pred.reindex(targets).round(2).values,
            "Neural net": nn_pred.reindex(targets).round(2).values,
            "Consensus (mean)": means.reindex(targets).round(2).values,
            "Your score": your.reindex(targets).values,
        }).set_index("Film")
        if your.notna().sum() == 0:
            table = table.drop(columns=["Your score"])
        st.dataframe(table, width="stretch")
        st.caption("Design 1 uses the same shrunken Pearson/magnitude formula as Rotten Tomatoes; Design 2 "
                   "(XGBoost) and Design 3 (neural net) are separately trained on member ratings over the "
                   "same 37-feature similarity-decile contract (no Tomatometer).")

        truth = your.reindex(targets).to_numpy()
        n_scored = int(np.isfinite(truth).sum())
        st.subheader("Which method predicts *you* best?")
        if n_scored == 0:
            st.info("Rate one or more of your predict films (section 2) to see the "
                    "mean squared error of each method against your own score.")
        else:
            candidates = [
                ("Analytic formula", formula.reindex(targets)),
                ("XGBoost", xgb_pred.reindex(targets)),
                ("Neural net", nn_pred.reindex(targets)),
                ("Consensus mean", means.reindex(targets)),
            ]
            rows = [(name, mse(series.to_numpy(), truth)) for name, series in candidates]
            best = min(rows, key=lambda r: r[1] if np.isfinite(r[1]) else np.inf)[0]
            st.caption(f"MSE against your own score over {n_scored} rated film(s). "
                       f"Lower is closer to your taste — **{best}** is closest here.")
            mse_cols = st.columns(len(rows))
            for col, (name, value) in zip(mse_cols, rows):
                col.metric(f"{name}{' ✅' if name == best else ''}",
                           f"{value:.3f}" if np.isfinite(value) else "—")

        with st.expander("Your closest members"):
            visible = matches[(matches["overlap"] >= LB_MIN_OVERLAP) & (matches["sim"] != 0)]
            top = visible.sort_values("sim", ascending=False).head(12)
            if top.empty:
                st.write("No overlapping member yet; predictions fall back to consensus.")
            else:
                st.dataframe(pd.DataFrame({
                    "Member": top.index,
                    "Alignment": top["sim"].round(2),
                    "Scale match": top["mag_sim"].round(2),
                    "Films in common": top["overlap"],
                }).set_index("Member"), width="stretch")


dataset = st.sidebar.radio("Project", ["Rotten Tomatoes", "Letterboxd"], index=0)
if dataset == "Letterboxd":
    run_letterboxd_app()
    st.stop()


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
