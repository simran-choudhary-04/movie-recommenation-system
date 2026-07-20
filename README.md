# 🎬 CineMatch — Content-Based Movie Recommendation Engine

A movie recommender built with **pandas, numpy, scikit-learn (TF-IDF + cosine similarity)**
and deployed as an interactive **Streamlit** app.

## 🌐 Live Demo
**Try the application here:**
🔗 https://movie-recommenation-system-3wwklx8pekygjmvwcjpdgp.streamlit.app

## 📸 Screenshots

| 🏠 Home Page | 🎬 Movie Recommendations |
|--------------|--------------------------|
| ![](screenshots/home.png.png) | ![](screenshots/recommendations.png.png) |
| 🧩 Quick Taste Quiz | 🎯 Preference-Based Recommendations |
|---------------------|-------------------------------------|
| ![](screenshots/quiz.png.png) | ![](screenshots/preference.png.png) |

## ✨ Features

- 🎬 Content-Based Movie Recommendation
- 📐 Cosine Similarity Search
- 🔍 TF-IDF Vectorization (Scikit-learn)
- 🎭 Three Recommendation Modes
  - Recommend by a Movie
  - Quick Taste Quiz
  - Recommend by Preferences
- 🖼 Movie Posters with Recommendation Results
- 🌐 Streamlit Web Application
- ☁️ Deployed on Streamlit Community Cloud

## How it works

1. Each movie's **genres, director, cast, keywords, and overview** are combined into one
   text "profile" (`recommender.py → _load_data`). Genres/director/cast are weighted more
   heavily since they're stronger similarity signals than free-text overviews.
2. `TfidfVectorizer` (scikit-learn) turns every movie's profile into a numeric vector.
3. `cosine_similarity` compares vectors to find movies that point in the same "direction"
   in feature space — i.e. movies that are thematically/stylistically similar.
4. Three recommendation modes are exposed in the app:
   - **By a movie I like** — classic "because you watched X" recommendations.
   - **By multiple favorites** — averages the TF-IDF vectors of several liked movies into
     one taste profile, then finds the closest matches.
   - **By my preferences** — creates a profile from selected genres, a favorite
     director/actor, and free-text keywords (no need to have watched anything).

## Project structure

```
movie_recommender/
├── app.py              # Streamlit UI
├── recommender.py       # Core recommendation engine (reusable, no Streamlit dependency)
├── data/
│   └── movies.csv        # Sample dataset (80 movies with genre/cast/director/keywords)
├── requirements.txt
└── README.md
```

## Run it locally

```bash
cd movie_recommender
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Use your own dataset

Swap in a bigger dataset (e.g. the MovieLens or TMDB 5000 datasets) — just make sure your
CSV has these columns, then point `MovieRecommender("data/movies.csv")` at the new file:

| column | description |
|---|---|
| `title` | movie title |
| `year` | release year |
| `genres` | space-separated genres, e.g. `Action Sci-Fi` |
| `director` | director name |
| `cast` | space-separated lead actor names |
| `keywords` | space-separated theme/plot keywords |
| `overview` | short plot summary |
| `rating` | numeric rating (for display only) |

## Deploying

### Option A — Streamlit Community Cloud (free, easiest)
1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select the repo/branch, and set the main file path to `app.py`.
4. Deploy — Streamlit installs `requirements.txt` automatically and gives you a public URL.

## Future Enhancements

- Add a larger movie dataset
- Integrate the TMDB API for richer movie details and posters
- Add user login and watchlist support
- Improve recommendations using user ratings
- Add movie trailers and reviews

## 👩‍💻 Author

**Simran Choudhary**

GitHub: https://github.com/simran-choudhary-04
