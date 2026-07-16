"""
Content-Based Movie Recommendation Engine
-------------------------------------------
Uses TF-IDF vectorization + cosine similarity over movie metadata
(genres, director, cast, keywords, overview) to recommend movies that
are similar to a movie a user likes, or to a custom preference profile
built from selected genres / people / keywords.

Author: Built with Claude
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


class MovieRecommender:
    def __init__(self, csv_path: str):
        self.df = self._load_data(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])
        # Precompute the full similarity matrix once (fine for a few thousand movies).
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.title_to_index = pd.Series(self.df.index, index=self.df["title"].str.lower())

    # ------------------------------------------------------------------ #
    # Data loading / feature engineering
    # ------------------------------------------------------------------ #
    def _load_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        text_cols = ["genres", "director", "cast", "keywords", "overview"]
        for col in text_cols:
            df[col] = df[col].fillna("")

        # Weight genres, director and cast more heavily by repeating them —
        # this nudges the vectorizer to treat them as stronger similarity signals
        # than the free-text overview.
        df["tags"] = (
            (df["genres"] + " ") * 3
            + (df["director"].str.replace(" ", "", regex=False) + " ") * 2
            + (df["cast"].str.replace(" ", "", regex=False) + " ") * 2
            + (df["keywords"] + " ") * 2
            + df["overview"]
        ).str.lower()

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Mode 1: "Because you liked movie X"
    # ------------------------------------------------------------------ #
    def recommend_by_movie(self, title: str, top_n: int = 10) -> pd.DataFrame:
        key = title.lower()
        if key not in self.title_to_index:
            raise ValueError(f"'{title}' not found in the dataset.")

        idx = self.title_to_index[key]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        result = self.df.iloc[[i for i, _ in scores]].copy()
        result["similarity"] = [round(s * 100, 1) for _, s in scores]
        return result[["title", "year", "genres", "director", "rating", "overview", "similarity"]]

    # ------------------------------------------------------------------ #
    # Mode 2: Recommend from several liked movies at once (a "taste profile")
    # ------------------------------------------------------------------ #
    def recommend_by_multiple_movies(self, titles: list, top_n: int = 10) -> pd.DataFrame:
        indices = []
        for t in titles:
            key = t.lower()
            if key in self.title_to_index:
                indices.append(self.title_to_index[key])
        if not indices:
            raise ValueError("None of the provided titles were found.")

        profile_vector = np.asarray(self.tfidf_matrix[indices].mean(axis=0))
        scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranked = [s for s in ranked if s[0] not in indices][:top_n]

        result = self.df.iloc[[i for i, _ in ranked]].copy()
        result["similarity"] = [round(s * 100, 1) for _, s in ranked]
        return result[["title", "year", "genres", "director", "rating", "overview", "similarity"]]

    # ------------------------------------------------------------------ #
    # Mode 3: Recommend from free-form stated preferences
    # (e.g. selected genres + favorite actor/director + keywords)
    # ------------------------------------------------------------------ #
    def recommend_by_preferences(self, genres=None, people=None, keywords=None, top_n: int = 10) -> pd.DataFrame:
        genres = genres or []
        people = people or []
        keywords = keywords or []

        profile_text = " ".join(
            [g.lower() for g in genres] * 3
            + [p.lower().replace(" ", "") for p in people] * 2
            + [k.lower() for k in keywords] * 2
        ).strip()

        if not profile_text:
            raise ValueError("Provide at least one genre, person, or keyword.")

        profile_vector = self.vectorizer.transform([profile_text])
        scores = linear_kernel(profile_vector, self.tfidf_matrix).flatten()

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        result = self.df.iloc[[i for i, _ in ranked]].copy()
        result["similarity"] = [round(s * 100, 1) for _, s in ranked]
        return result[["title", "year", "genres", "director", "rating", "overview", "similarity"]]

    # ------------------------------------------------------------------ #
    def all_titles(self):
        return self.df["title"].tolist()

    def all_genres(self):
        genres = set()
        for g in self.df["genres"]:
            genres.update(g.split())
        return sorted(genres)

    # ------------------------------------------------------------------ #
    # Explainability: why was this movie recommended?
    # ------------------------------------------------------------------ #
    def explain_movie_match(self, base_title: str, rec_title: str) -> dict:
        """Returns the concrete overlapping signals between two movies."""
        b = self.df[self.df["title"].str.lower() == base_title.lower()].iloc[0]
        r = self.df[self.df["title"].str.lower() == rec_title.lower()].iloc[0]

        shared_genres = sorted(set(b["genres"].split()) & set(r["genres"].split()))
        shared_keywords = sorted(set(b["keywords"].split()) & set(r["keywords"].split()))
        same_director = b["director"] if b["director"] == r["director"] and b["director"] else None

        return {
            "genres": shared_genres,
            "keywords": shared_keywords,
            "director": same_director,
        }

    def explain_preference_match(self, genres, keywords, rec_title: str) -> dict:
        """Overlap between a stated preference profile and a recommended movie."""
        r = self.df[self.df["title"].str.lower() == rec_title.lower()].iloc[0]
        rec_genres = set(r["genres"].split())
        rec_keywords = set(r["keywords"].split())
        shared_genres = sorted({g for g in genres if g in rec_genres})
        shared_keywords = sorted({k.lower() for k in keywords if k.lower() in rec_keywords})
        return {"genres": shared_genres, "keywords": shared_keywords, "director": None}

    # ------------------------------------------------------------------ #
    # Model internals — useful for a "how it works" panel
    # ------------------------------------------------------------------ #
    def model_stats(self) -> dict:
        n_movies, vocab_size = self.tfidf_matrix.shape
        nnz = self.tfidf_matrix.nnz
        density = nnz / (n_movies * vocab_size) * 100
        return {
            "n_movies": n_movies,
            "vocab_size": vocab_size,
            "nonzero_entries": nnz,
            "matrix_density_pct": round(density, 3),
            "sparsity_pct": round(100 - density, 3),
        }

    def genre_distribution(self) -> pd.Series:
        all_g = []
        for g in self.df["genres"]:
            all_g.extend(g.split())
        return pd.Series(all_g).value_counts()

    def top_rated(self, n: int = 8) -> pd.DataFrame:
        return self.df.sort_values("rating", ascending=False).head(n)

    def random_sample(self, n: int = 6, seed: int = None) -> pd.DataFrame:
        return self.df.sample(n=n, random_state=seed)


if __name__ == "__main__":
    # Quick smoke test
    rec = MovieRecommender("data/movies.csv")
    print("Because you liked 'Inception':")
    print(rec.recommend_by_movie("Inception", top_n=5), "\n")

    print("Based on preferences (genre=Animation, keyword=family):")
    print(rec.recommend_by_preferences(genres=["Animation"], keywords=["family"], top_n=5))
