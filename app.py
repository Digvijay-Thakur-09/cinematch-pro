
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64
import time
import math
from typing import List, Dict, Tuple, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pathlib


st.set_page_config(
    page_title="CineMatch Pro | AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


LANGUAGE_MAPPING: Dict[str, str] = {
    'en': 'English', 'ja': 'Japanese', 'fr': 'French', 'es': 'Spanish', 
    'ko': 'Korean', 'de': 'German', 'it': 'Italian', 'zh': 'Chinese', 
    'hi': 'Hindi', 'ru': 'Russian', 'pt': 'Portuguese', 'ar': 'Arabic',
    'bn': 'Bengali', 'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch',
    'fi': 'Finnish', 'el': 'Greek', 'he': 'Hebrew', 'id': 'Indonesian',
    'no': 'Norwegian', 'pl': 'Polish', 'ro': 'Romanian', 'sv': 'Swedish',
    'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'vi': 'Vietnamese',
    'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada',
    'mr': 'Marathi', 'gu': 'Gujarati', 'ur': 'Urdu', 'fa': 'Persian',
    'hu': 'Hungarian', 'sk': 'Slovak', 'bg': 'Bulgarian', 'sr': 'Serbian',
    'hr': 'Croatian', 'sl': 'Slovenian', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'et': 'Estonian', 'is': 'Icelandic', 'ga': 'Irish', 'cy': 'Welsh',
    'tl': 'Tagalog', 'ms': 'Malay', 'sw': 'Swahili', 'am': 'Amharic',
    'yo': 'Yoruba', 'ig': 'Igbo', 'zu': 'Zulu', 'xh': 'Xhosa',
    'af': 'Afrikaans', 'sq': 'Albanian', 'hy': 'Armenian', 'az': 'Azerbaijani',
    'eu': 'Basque', 'be': 'Belarusian', 'bs': 'Bosnian', 'ca': 'Catalan',
    'ka': 'Georgian', 'gl': 'Galician', 'kk': 'Kazakh', 'km': 'Khmer',
    'ky': 'Kyrgyz', 'lo': 'Lao', 'mk': 'Macedonian', 'mn': 'Mongolian',
    'ne': 'Nepali', 'ps': 'Pashto', 'si': 'Sinhala', 'so': 'Somali',
    'tg': 'Tajik', 'uz': 'Uzbek', 'xx': 'No Language'
}


GENRE_COLORS: Dict[str, str] = {
    "Action": "#FF4136", "Adventure": "#FF851B", "Animation": "#FFDC00",
    "Comedy": "#2ECC40", "Crime": "#111111", "Documentary": "#AAAAAA",
    "Drama": "#0074D9", "Family": "#39CCCC", "Fantasy": "#B10DC9",
    "History": "#85144b", "Horror": "#000000", "Music": "#F012BE",
    "Mystery": "#3D9970", "Romance": "#FF69B4", "Science Fiction": "#01FF70",
    "TV Movie": "#7FDBFF", "Thriller": "#FF0000", "War": "#8B0000", "Western": "#D2691E"
}


def inject_custom_css() -> None:
    """
    Injects hundreds of lines of custom CSS to override Streamlit's default UI.
    Includes CSS variables, advanced grid layouts, keyframe animations, and
    responsive design media queries.
    """
    custom_css = """
    <style>
    /* --- CSS VARIABLES FOR THEMING --- */
    :root {
        --primary-red: #E50914;
        --primary-hover: #B20710;
        --bg-dark: #0B0B0B;
        --card-bg: #141414;
        --text-main: #FFFFFF;
        --text-muted: #808080;
        --accent-glow: rgba(229, 9, 20, 0.4);
        --glass-bg: rgba(20, 20, 20, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --transition-speed: 0.4s;
    }

    /* --- GLOBAL OVERRIDES --- */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-main);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Remove default padding */
    .css-18e3th9 { padding-top: 1rem; padding-bottom: 1rem; }
    .block-container { max-width: 1600px; padding-top: 2rem; }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-main) !important;
        border-bottom: 3px solid var(--primary-red) !important;
    }

    /* --- SIDEBAR DESIGN --- */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #222;
    }
    
    /* Premium Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--primary-hover) 100%);
        color: white;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: none;
        padding: 14px 24px;
        transition: all var(--transition-speed) cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px var(--accent-glow);
        margin-top: 20px;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.6);
        color: white;
    }
    .stButton>button:active {
        transform: translateY(1px) scale(0.98);
    }

    /* --- KEYFRAME ANIMATIONS --- */
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 10px var(--accent-glow); }
        50% { box-shadow: 0 0 20px rgba(229, 9, 20, 0.8); }
        100% { box-shadow: 0 0 10px var(--accent-glow); }
    }

    /* --- MOVIE CAROUSEL & CARDS --- */
    .movie-carousel {
        display: flex;
        overflow-x: auto;
        gap: 20px;
        padding: 20px 10px 50px 10px; /* Room for hover shadows */
        scroll-behavior: smooth;
        -webkit-overflow-scrolling: touch;
        animation: fadeSlideUp 0.8s ease-out forwards;
    }
    
    /* Custom Scrollbar for Carousel */
    .movie-carousel::-webkit-scrollbar { height: 10px; }
    .movie-carousel::-webkit-scrollbar-track {
        background: #111;
        border-radius: 10px;
        margin: 0 20px;
    }
    .movie-carousel::-webkit-scrollbar-thumb {
        background: var(--primary-red);
        border-radius: 10px;
        border: 2px solid #111;
    }

    .movie-card {
        flex: 0 0 240px; /* Fixed width */
        height: 360px; /* Fixed height for uniformity */
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        cursor: pointer;
        transition: all var(--transition-speed) ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.8);
        background-color: var(--card-bg);
    }
    
    .movie-card:hover {
        transform: translateY(-12px) scale(1.04);
        box-shadow: 0 20px 40px rgba(0,0,0,0.9), 0 0 20px rgba(255,255,255,0.05);
        z-index: 10;
    }
    
    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    
    .movie-card:hover .movie-poster {
        transform: scale(1.08); /* Slow zoom on hover */
    }

    /* --- CARD OVERLAY (The Hover Reveal) --- */
    .card-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 100%;
        background: linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0.8) 40%, rgba(0,0,0,0) 100%);
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        opacity: 0; /* Hidden by default */
        transition: opacity var(--transition-speed) ease;
    }
    
    .movie-card:hover .card-overlay {
        opacity: 1; /* Revealed on hover */
    }

    /* Overlay Typography */
    .movie-title {
        font-size: 1.25rem;
        font-weight: 900;
        color: var(--text-main);
        line-height: 1.2;
        margin-bottom: 8px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .movie-details {
        font-size: 0.9rem;
        color: #CCCCCC;
        font-weight: 500;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .match-score {
        color: #46D369; /* Netflix Green Match Score */
        font-weight: 800;
        font-size: 1.05rem;
    }

    /* --- GLASSMORPHISM TOP BADGES --- */
    .top-badges {
        position: absolute;
        top: 10px;
        right: 10px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        align-items: flex-end;
        z-index: 5;
    }
    
    .badge {
        background: var(--glass-bg);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .badge-star { color: #F5C518; } /* IMDB Yellow */

    /* --- RESPONSIVE MEDIA QUERIES --- */
    @media (max-width: 1200px) {
        .movie-card { flex: 0 0 200px; height: 300px; }
        .movie-title { font-size: 1.1rem; }
    }
    
    @media (max-width: 768px) {
        .movie-card { flex: 0 0 160px; height: 240px; }
        .movie-title { font-size: 0.95rem; }
        .stButton>button { font-size: 1rem; padding: 10px 16px; }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


class MLB_Wrapper(BaseEstimator, TransformerMixin):
    """
    A custom Scikit-Learn Transformer to wrap MultiLabelBinarizer.
    This class is required to exist in the main file so `joblib` can 
    successfully unpickle the saved pipeline architecture.
    """
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X: pd.DataFrame, y=None):
        # Assumes the first column contains the list of genres
        self.mlb.fit(X.iloc[:, 0])
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.mlb.transform(X.iloc[:, 0])


@st.cache_resource(show_spinner="Loading Neural Network...")
def load_model() -> Any:
    """
    Loads the trained Scikit-Learn Pipeline from disk.
    Uses @st.cache_resource to keep the model in memory across sessions.
    """
    try:
        model_path = pathlib.Path('movie_pipeline.joblib')
        if not model_path.exists():
            st.error("🚨 Critical Error: 'movie_pipeline.joblib' not found. Please run the training script (02.py) first.")
            st.stop()
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

@st.cache_data(show_spinner="Loading Cinematic Database...")
def load_data() -> pd.DataFrame:
    """
    Loads the cleaned Pandas DataFrame from disk.
    Uses @st.cache_data to serialize the data for fast lookups.
    """
    try:
        data_path = pathlib.Path('movie_dataframe.joblib')
        if not data_path.exists():
            st.error("🚨 Critical Error: 'movie_dataframe.joblib' not found. Please run the training script (02.py) first.")
            st.stop()
        df = joblib.load(data_path)
        

        if 'popularity' not in df.columns:
            df['popularity'] = 10.0 # Fallback
            
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        st.stop()


def generate_fallback_svg(title: str, year: int, language: str) -> str:
    """
    Generates a beautiful, base64-encoded SVG image to act as a placeholder
    if a movie is missing its official TMDB poster path. This ensures the app
    works entirely offline and doesn't rely on external placeholder services.
    
    Args:
        title: The movie title
        year: The release year
        language: The displayed language name
        
    Returns:
        A data URI string containing the base64 encoded SVG.
    """

    hue = (len(title) * 15) % 360

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="600" viewBox="0 0 400 600">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:hsl({hue}, 70%, 20%);stop-opacity:1" />
                <stop offset="100%" style="stop-color:hsl({hue + 40}, 80%, 10%);stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect width="400" height="600" fill="url(#grad)" />
        <rect width="360" height="560" x="20" y="20" fill="transparent" stroke="rgba(255,255,255,0.1)" stroke-width="2" rx="10" />
        
        <text x="200" y="250" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="white" text-anchor="middle" dominant-baseline="middle">
            <tspan x="200" dy="-40">NO OFFICIAL</tspan>
            <tspan x="200" dy="40">POSTER AVAILABLE</tspan>
        </text>
        
        <path d="M 150 350 L 250 350 L 200 420 Z" fill="rgba(255,255,255,0.2)" />
        
        <text x="200" y="500" font-family="Arial, sans-serif" font-size="16" fill="#AAAAAA" text-anchor="middle">
            {year} • {language}
        </text>
    </svg>
    """
    # Encode to base64
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

def calculate_match_score(distance: float, max_expected_dist: float = 2.0) -> int:
    """
    Converts a raw mathematical KNN distance (usually 0.0 to 1.0 in Cosine)
    into a consumer-friendly percentage score (e.g., 95%).
    """

    if max_expected_dist == 0: return 100
    

    raw_pct = (1.0 - (distance / max_expected_dist)) * 100
    

    score = max(10, min(99, int(raw_pct)))
    

    return min(99, score + 10)


class RecommendationEngine:
    """
    Encapsulates all logic required to process user inputs, vectorize them,
    query the Scikit-Learn model, and execute the Waterfall Fallback filtering algorithm.
    """
    def __init__(self, pipeline: Any, dataset: pd.DataFrame):
        self.pipeline = pipeline
        self.df = dataset
        
    def generate_recommendations(
        self, 
        genres: List[str], 
        year: int, 
        popularity_percentile: float,
        target_langs: List[str],
        strict_year: bool,
        num_results: int
    ) -> Tuple[pd.DataFrame, bool, List[float]]:
        """
        Executes the full pipeline to generate movie recommendations.
        
        Returns:
            Tuple containing:
            1. The final filtered DataFrame of movies
            2. Boolean flag indicating if fallback logic was triggered
            3. List of corresponding match distances
        """

        raw_pop_target = float(self.df['popularity'].quantile(popularity_percentile / 10.0))
        user_df = pd.DataFrame({
            'genre_names': [genres], 
            'release_year': [year],
            'popularity': [raw_pop_target]
        })

        user_vector = self.pipeline.named_steps['preprocessor'].transform(user_df)
        

        pool_size = min(5000, len(self.df))
        distances, indices = self.pipeline.named_steps['knn'].kneighbors(user_vector, n_neighbors=pool_size)

        candidates = self.df.iloc[indices[0]].copy()
        dist_list = distances[0]
        

        candidates['knn_distance'] = dist_list
        

        candidates['lang_name'] = candidates['original_language'].apply(
            lambda x: LANGUAGE_MAPPING.get(x, "Other")
        )
        

        candidates['year_diff'] = abs(candidates['release_year'] - year)
        candidates['pop_diff'] = abs(candidates['popularity'] - raw_pop_target)
        

        final_df, fallback_used = self._execute_waterfall(
            pool=candidates,
            target_langs=target_langs,
            strict_year_flag=strict_year,
            needed_count=num_results
        )
        
        return final_df.head(num_results), fallback_used, final_df['knn_distance'].tolist()
        
    def _execute_waterfall(
        self, 
        pool: pd.DataFrame, 
        target_langs: List[str], 
        strict_year_flag: bool,
        needed_count: int
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Internal logic: Attempts to find the requested amount of movies using strict rules.
        If it fails, it progressively relaxes the rules (waterfall) to guarantee results.
        """
        fallback_triggered = False
        
        def filter_and_sort(df: pd.DataFrame, enforce_lang: bool, enforce_year: bool) -> pd.DataFrame:
            temp = df.copy()
            if enforce_lang and target_langs:
                temp = temp[temp['lang_name'].isin(target_langs)]
            if enforce_year and strict_year_flag:

                temp = temp[temp['year_diff'] <= 10]
                

            return temp.sort_values(by=['year_diff', 'knn_distance', 'pop_diff'], ascending=[True, True, True])

        result = filter_and_sort(pool, enforce_lang=True, enforce_year=True)

        if len(result) < needed_count:
            fallback_triggered = True
            result = filter_and_sort(pool, enforce_lang=True, enforce_year=False)

        if len(result) < needed_count:
            fallback_triggered = True
            result = filter_and_sort(pool, enforce_lang=False, enforce_year=False)
            
        return result, fallback_triggered



def render_movie_card(movie: pd.Series, distance: float) -> str:
    """
    Constructs the complex HTML required for a single, interactive movie card.
    Includes badges, hover states, and safe string escaping.
    """

    title = str(movie['title'])
    safe_title = title.replace("'", "&#39;").replace('"', "&quot;")
    
    year = int(movie.get('release_year', 0))
    vote_avg = float(movie.get('vote_average', 0.0))
    lang_name = movie.get('lang_name', 'Unknown')

    pop_raw = movie.get('popularity', 0.0)
    percentile_rank = (df['popularity'] <= pop_raw).mean()
    pop_display = percentile_rank * 10.0

    match_score = calculate_match_score(distance)
    score_color = "#46D369" if match_score >= 80 else "#E8B708" if match_score >= 60 else "#E50914"

    poster_path = movie.get('poster_path', "")
    if pd.notna(poster_path) and str(poster_path).strip() != "":
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:

        poster_url = generate_fallback_svg(title, year, lang_name)

    html = (
        f"<div class='movie-card'>"
        f"<div class='top-badges'>"
        f"<div class='badge'><span class='badge-star'>★</span> {vote_avg:.1f}</div>"
        f"<div class='badge'>🗣️ {lang_name[:3].upper()}</div>"
        f"</div>"
        f"<img src='{poster_url}' class='movie-poster' alt='Poster for {safe_title}' loading='lazy'>"
        f"<div class='card-overlay'>"
        f"<div class='movie-title'>{safe_title}</div>"
        f"<div class='movie-details'>"
        f"<span class='match-score' style='color:{score_color}'>{match_score}% Match</span>"
        f"<span>{year}</span>"
        f"</div>"
        f"<div class='movie-details' style='font-size:0.8rem; margin-bottom:0;'>"
        f"<span>🌟 {pop_display:.1f}/10 Popularity</span>"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    return html

def render_analytics_dashboard(df: pd.DataFrame):
    """Renders the secondary tab showing dataset statistics."""
    st.header("📊 Database Analytics")
    st.markdown("A real-time look into the cinematic data powering the engine.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies Catalogued", f"{len(df):,}")
    with col2:
        st.metric("Earliest Film", int(df['release_year'].min()))
    with col3:
        st.metric("Latest Film", int(df['release_year'].max()))
    with col4:
        st.metric("Avg Database Rating", f"{df['vote_average'].mean():.2f}/10")
        
    st.divider()
    
    st.subheader("Language Distribution (Top 10)")

    lang_counts = df['original_language'].map(LANGUAGE_MAPPING).value_counts().head(10)
    st.bar_chart(lang_counts, color="#E50914")

def render_architecture_docs():
    """Renders the tertiary tab explaining the system design."""
    st.header("🧠 System Architecture")
    st.markdown("""
    ### How CineMatch Pro Works
    
    **1. The Vectorization (MultiLabelBinarizer)**
    Movies are complex. To help a computer understand them, we turn genres into mathematics. 
    A movie that is an Action and Sci-Fi film is converted into an array of 1s and 0s 
    (e.g., `[1, 0, 0, 1, 0...]`). We do the same for the preferences you select in the sidebar.
    
    **2. K-Nearest Neighbors (KNN)**
    Once you click "Generate", your preferences become a "Fake User Vector" in N-dimensional space.
    Our algorithm calculates the *Cosine Distance* between your preference dot, and the 660,000+ movie dots in our database.
    It grabs the 5,000 closest dots.
    
    **3. The Waterfall Fallback**
    Because mathematics isn't always perfect for human taste (a blockuster from 2020 might be mathematically closer 
    to your vector than an indie film from 1950, even if you asked for 1950), we use a custom Pandas Waterfall sorter.
    We take the 5,000 movies KNN found, and strictly re-sort them based on your exact Year and Language parameters.
    If we can't find enough, the algorithm progressively relaxes the rules to ensure your screen is never empty.
    """)

def main():
    # 1. Initialization
    inject_custom_css()
    global df, pipeline
    df = load_data()
    pipeline = load_model()
    engine = RecommendationEngine(pipeline, df)
    
    st.markdown("<h1 style='text-align: center; font-weight: 900; margin-bottom: 0;'>🍿 CineMatch <span style='color: #E50914;'>PRO</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.2rem; margin-bottom: 2rem;'>Enterprise Recommendation Engine Architecture</p>", unsafe_allow_html=True)
    
    tab_discovery, tab_analytics, tab_about = st.tabs(["🔍 The Engine", "📊 Data Analytics", "🧠 Architecture"])
    
    with tab_discovery:
        
        with st.sidebar:
            st.header("🎛️ Director's Console")
            st.markdown("Configure your algorithmic preferences.")
            
            # Genres
            all_genres = sorted({g for sublist in df['genre_names'] for g in sublist})
            selected_genres = st.multiselect("🎭 Core Genres:", all_genres, default=["Action", "Science Fiction"])
            
            # Languages
            available_langs = sorted(list(set(LANGUAGE_MAPPING.values())))
            selected_langs = st.multiselect("🌐 Spoken Languages:", available_langs, default=["English", "Japanese"])
            
            # Time Period
            min_year = int(df['release_year'].min())
            max_year = int(df['release_year'].max())
            selected_year = st.slider("📅 Target Era (Year):", min_year, max_year, 2010)

            strict_year = st.toggle("🔒 Enforce Strict Era (+/- 10 Yrs)", value=True, help="If disabled, the algorithm will prioritize genres over the exact release year.")

            selected_popularity_level = st.slider(
                "🌟 Mainstream Appeal:", 0.0, 10.0, 7.5, 
                help="0 = Obscure Indie Films. 10 = Global Blockbusters."
            )
            
            num_movies = st.slider("🎞️ Engine Output Size:", 1, 40, 15)
            
            st.markdown("<br>", unsafe_allow_html=True)
            generate_button = st.button("Initialize Engine")
            
            with st.expander("ℹ️ How to use"):
                st.markdown("1. Select your core vibe using the sliders.\n2. Click Initialize Engine.\n3. Hover over the movie cards for detailed match data.")


        if generate_button:
            if not selected_genres:
                st.error("⚠️ System Override: Cannot run inference without at least one target Genre.")
            else:

                with st.spinner("Initializing Vector Space... Calculating Cosine Distances..."):

                    time.sleep(0.5) 
                    

                    final_df, fallback_used, distances = engine.generate_recommendations(
                        genres=selected_genres,
                        year=selected_year,
                        popularity_percentile=selected_popularity_level,
                        target_langs=selected_langs,
                        strict_year=strict_year,
                        num_results=num_movies
                    )
                    

                    if fallback_used:
                        st.warning(f"🔄 **Algorithmic Fallback Engaged:** Unable to find {num_movies} exact matches for this specific Era/Language combination. Search parameters were dynamically expanded.")
                        
                    st.markdown(f"### Output Generated: Top {len(final_df)} Matches")

                    carousel_html = "<div class='movie-carousel'>"
                    for (index, movie), dist in zip(final_df.iterrows(), distances):
                        carousel_html += render_movie_card(movie, dist)
                    carousel_html += "</div>"
                    
                    st.markdown(carousel_html, unsafe_allow_html=True)
        else:

            st.info("👈 Configure your parameters in the Director's Console and Initialize the Engine to begin discovery.")

    with tab_analytics:
        render_analytics_dashboard(df)

    with tab_about:
        render_architecture_docs()

if __name__ == "__main__":
    main()