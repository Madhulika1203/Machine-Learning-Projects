import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests

st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")

# TMDB API KEY
TMDB_API_KEY = "94842568766808b32ae6a7691b035bfb"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
POSTER_PLACEHOLDER = "https://via.placeholder.com/500x750?text=No+Poster"

#polish / equalized cards & posters
st.markdown("""
<style>
  .rec-card {padding:12px; border-radius:16px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);}
  .rec-title {font-weight:700; font-size:1rem; margin:6px 0 8px 0; min-height: 40px;}
  .rec-meta {color:rgba(255,255,255,0.75); font-size:0.85rem; margin-bottom:10px; min-height: 22px;}
  .poster-img {width:100%; height:360px; object-fit:cover; border-radius:12px;}
  .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    movies_obj = pickle.load(open("movies.pkl", "rb"))
    if isinstance(movies_obj, pd.DataFrame):
        movies_df = movies_obj.copy()
    elif isinstance(movies_obj, dict):
        movies_df = pd.DataFrame(movies_obj)
    else:
        movies_df = pd.DataFrame({"title": pd.Series(movies_obj, dtype="string")})

    movies_df.columns = [c.strip().lower() for c in movies_df.columns]
    if "title" not in movies_df.columns:
        raise ValueError("movies.pkl must have a 'title' column (or be a list of titles).")

    if "movie_id" not in movies_df.columns:
        if "id" in movies_df.columns:
            movies_df.rename(columns={"id": "movie_id"}, inplace=True)
        else:
            movies_df["movie_id"] = pd.Series([np.nan] * len(movies_df))

    sim = pickle.load(open("similarity.pkl", "rb"))
    sim = np.asarray(sim, dtype=float)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1] or sim.shape[0] != len(movies_df):
        raise ValueError(
            f"similarity.pkl shape {sim.shape} does not match movies count {len(movies_df)}."
        )

    titles = movies_df["title"].astype(str).tolist()
    return movies_df.reset_index(drop=True), sim, titles

movies, similarity, movies_list = load_data()

# TMDB helpers (with tiny retry + caching)
@st.cache_data(show_spinner=False)
def _tmdb_get(url: str):
    for _ in range(2):  # light retry for transient issues
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {}

def fetch_info_by_id(movie_id):
    if pd.isna(movie_id):
        return {}
    data = _tmdb_get(
        f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_API_KEY}&language=en-US"
    )
    if not data:
        return {}
    title = data.get("title") or data.get("name")
    poster_path = data.get("poster_path")
    year = (data.get("release_date") or "")[:4]
    rating = data.get("vote_average")
    url = f"https://www.themoviedb.org/movie/{int(movie_id)}"
    poster = TMDB_IMAGE_BASE + poster_path if poster_path else None
    return {"title": title, "poster": poster, "year": year, "rating": rating, "url": url}

def fetch_info_by_title(title):
    q = requests.utils.quote(title)
    data = _tmdb_get(
        f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={q}&language=en-US&include_adult=false"
    )
    results = (data or {}).get("results") or []
    if not results:
        return {}
    m = results[0]
    poster_path = m.get("poster_path")
    year = (m.get("release_date") or "")[:4]
    rating = m.get("vote_average")
    mid = m.get("id")
    url = f"https://www.themoviedb.org/movie/{mid}" if mid else None
    poster = TMDB_IMAGE_BASE + poster_path if poster_path else None
    return {"title": m.get("title"), "poster": poster, "year": year, "rating": rating, "url": url}

def fetch_info(movie_id, title):
    info = fetch_info_by_id(movie_id)
    if info:
        return info
    return fetch_info_by_title(title)

# Recommender
def recommend(title, k=5):
    try:
        idx = movies_list.index(title)
    except ValueError:
        return []

    d = similarity[idx].copy()
    d[idx] = -np.inf
    top_idx = np.argsort(-d)[:k]

    items = []
    for i in top_idx:
        name = movies_list[i]
        mid = movies.loc[i, "movie_id"]
        info = fetch_info(mid, name) or {"title": name, "poster": None, "year": "", "rating": None, "url": None}
        # ensure we never return a missing poster
        if not info.get("poster"):
            info["poster"] = POSTER_PLACEHOLDER
        # make sure title is always filled (some APIs return None)
        if not info.get("title"):
            info["title"] = name
        items.append(info)
    return items

# UI
st.title("🎬 Movie Recommender System")
selected_movie = st.selectbox("Pick a movie:", movies_list, index=0)

if st.button("Recommend"):
    items = recommend(selected_movie, k=5)
    if not items:
        st.error("No recommendations found. Check your pickle files.")
    else:
        cols = st.columns(5)
        for col, it in zip(cols, items):
            with col:
                st.markdown('<div class="rec-card">', unsafe_allow_html=True)
                # title + meta (fixed heights so cards align)
                title_line = it.get("title") or "Untitled"
                meta_bits = []
                if it.get("year"):
                    meta_bits.append(it["year"])
                if it.get("rating") is not None:
                    meta_bits.append(f"⭐ {it['rating']:.1f}")
                st.markdown(f'<div class="rec-title">{title_line}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="rec-meta">{" · ".join(meta_bits)}</div>', unsafe_allow_html=True)
                # equal-height poster using HTML for object-fit:cover
                st.markdown(f'<img class="poster-img" src="{it["poster"]}">', unsafe_allow_html=True)
                # link (optional)
                if it.get("url"):
                    st.markdown(f"[TMDB page]({it['url']})")
                st.markdown("</div>", unsafe_allow_html=True)

