import requests
import pandas as pd
import time
import ast
import joblib
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURATION ---
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3ZmU4NjBkZTQxNGU1NjYwOWM0ODRiMjJlOTcyNTNiNiIsIm5iZiI6MTc3NzMwNzI2OS42MDMwMDAyLCJzdWIiOiI2OWVmOGU4NTFlMjk2ZmNjODk5MzcwMGUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.VT3H1m0SZ45n2wG5mams9nu6nmGUQsOOIaQwunf1sdA"
MASTER_DATABASE_FILE = "tmdb_file_clean.csv.gz"# Your main dataset

HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

class MLB_Wrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    def fit(self, X, y=None):
        self.mlb.fit(X.iloc[:, 0])
        return self
    def transform(self, X):
        return self.mlb.transform(X.iloc[:, 0])

def fetch_page_with_retry(url, page, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                return None

def fetch_all_weekly_movies():
    print("📡 Phase 1: Fetching past 7 days of movies...")
    today = datetime.today()
    seven_days_ago = today - timedelta(days=7)
    
    base_url = (
        f"https://api.themoviedb.org/3/discover/movie"
        f"?primary_release_date.gte={seven_days_ago.strftime('%Y-%m-%d')}"
        f"&primary_release_date.lte={today.strftime('%Y-%m-%d')}"
        f"&sort_by=popularity.desc"
    )
    
    first_page = fetch_page_with_retry(f"{base_url}&page=1", page=1)
    if not first_page: return pd.DataFrame()
        
    total_pages = min(first_page.get('total_pages', 1), 500) 
    new_movies = []
    
    for page in range(1, total_pages + 1):
        data = first_page if page == 1 else fetch_page_with_retry(f"{base_url}&page={page}", page=page)
        if data and 'results' in data:
            for m in data['results']:
                new_movies.append({
                    'id': m['id'], 'title': m['title'], 'release_date': m.get('release_date', ''),
                    'popularity': m['popularity'], 'vote_average': m['vote_average'],
                    'vote_count': m['vote_count'], 'original_language': m['original_language'],
                    'genre_ids': m.get('genre_ids', []), 'poster_path': m.get('poster_path', '')
                })
    return pd.DataFrame(new_movies)

def clean_and_merge(new_df):
    print(f"🧹 Phase 2: Cleaning and merging {len(new_df)} new movies...")
    new_df['release_date'] = pd.to_datetime(new_df['release_date'], errors='coerce')
    new_df['release_year'] = new_df['release_date'].dt.year
    new_df['release_month'] = new_df['release_date'].dt.month
    
    try:
        old_df = pd.read_csv(MASTER_DATABASE_FILE)
    except FileNotFoundError:
        old_df = pd.DataFrame()

    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    
    # THIS IS THE MAGIC DEDUPLICATION LINE
    combined_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
    
    combined_df.to_csv(MASTER_DATABASE_FILE, index=False)
    return combined_df

def retrain_model(df):
    print("🧠 Phase 3: Retraining the Recommendation Engine...")
    df = df[(df['vote_average'] > 0) & (df['vote_count'] > 5)].copy()
    df = df.dropna(subset=['release_year', 'popularity', 'genre_ids'])
    df['genre_ids'] = df['genre_ids'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)

    genre_map = {
        28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy", 80:"Crime", 99:"Documentary", 
        18:"Drama", 10751:"Family", 14:"Fantasy", 36:"History", 27:"Horror", 10402:"Music",
        9648:"Mystery", 10749:"Romance", 878:"Science Fiction", 10770:"TV Movie", 53:"Thriller", 
        10752:"War", 37:"Western"
    }
    df['genre_names'] = df['genre_ids'].apply(lambda ids: [genre_map.get(i, "Unknown") for i in ids] if isinstance(ids, list) else [])

    preprocessor = ColumnTransformer(transformers=[
        ('genres', MLB_Wrapper(), ['genre_names']),
        ('numbers', MinMaxScaler(), ['release_year', 'popularity'])
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute'))
    ])

    pipeline.fit(df)
    joblib.dump(pipeline, "movie_pipeline.joblib", compress=3)
    joblib.dump(df, 'movie_dataframe.joblib', compress=3)
    print("✅ Model updated and saved successfully!")

if __name__ == "__main__":
    new_data = fetch_all_weekly_movies()
    if not new_data.empty:
        updated_db = clean_and_merge(new_data)
        retrain_model(updated_db)
        print("🎉 Full Pipeline Execution Complete!")
    else:
        print("❌ Pipeline failed during fetch phase.")