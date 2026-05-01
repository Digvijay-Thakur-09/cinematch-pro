🍿 CineMatch Pro: AI Movie Discovery Engine

CineMatch Pro is an end-to-end, production-ready movie recommendation system. It goes beyond simple static datasets by utilizing an automated data pipeline that fetches, cleans, and merges live movie data from the TMDB API every week. 

### 🧠 Core Architecture
* **Machine Learning:** Utilizes a custom Scikit-Learn Pipeline combining `MultiLabelBinarizer` and `MinMaxScaler` to vectorize user preferences.
* **Algorithmic Engine:** Employs K-Nearest Neighbors (KNN) using Cosine Distance to find mathematical similarities between user vectors and a database of over 600,000+ films.
* **Waterfall Fallback Logic:** Features dynamic parameter relaxation. If strict year/language constraints yield zero results, the algorithm progressively relaxes filters to guarantee successful recommendations.

### ⚙️ Automated Data Pipeline (CI/CD)
* **Weekly Scraping:** A robust, automated Python script (`update_data.py`) runs via **GitHub Actions** every Sunday.
* **Network Resilience:** Engineered with exponential backoff and retry loops to bypass aggressive ISP firewalls and connection resets during API fetching.
* **Zero-Downtime Updates:** The script automatically deduplicates records, retrains the Scikit-Learn `.joblib` models, and commits the fresh data back to the repository, instantly updating the live frontend.

### 💻 Tech Stack
* **Frontend:** Python, Streamlit, Custom CSS/HTML
* **Backend Data Pipeline:** Pandas, Requests, GitHub Actions
* **Machine Learning:** Scikit-Learn, NumPy, Joblib
* **Data Source:** TMDB (The Movie Database) API v3/v4

### 🚀 Live Demo

Link -> https://cinematch-pro-kqoewokn4rnyzpvpjcjztr.streamlit.app/
