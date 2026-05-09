from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_search import parse_natural_language_query


class DatasetError(Exception):
    """Raised when the movie dataset is missing or incorrectly formatted."""


class MovieRecommender:
    """Content-based movie recommender using TF-IDF and cosine similarity."""

    REQUIRED_COLUMNS = [
        "title",
        "genre",
        "language",
        "year",
        "rating",
        "director",
        "cast",
        "description",
    ]

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.movies = self._load_dataset()
        self._preprocess_dataset()
        self._train_model()

    def _load_dataset(self):
        if not self.dataset_path.exists():
            raise DatasetError(f"Dataset file not found: {self.dataset_path}")

        try:
            return pd.read_csv(self.dataset_path)
        except Exception as exc:
            raise DatasetError(f"Could not read dataset CSV: {exc}") from exc

    def _preprocess_dataset(self):
        """Check columns, fill missing values, and build the combined text field."""
        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in self.movies.columns
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise DatasetError(f"Dataset is missing required column(s): {missing}")

        text_columns = ["title", "genre", "language", "director", "cast", "description"]
        for column in text_columns:
            self.movies[column] = self.movies[column].fillna("").astype(str)

        self.movies["year"] = pd.to_numeric(self.movies["year"], errors="coerce").fillna(0)
        self.movies["year"] = self.movies["year"].astype(int)
        self.movies["rating"] = pd.to_numeric(
            self.movies["rating"], errors="coerce"
        ).fillna(0.0)

        # The recommendation model learns from this combined content feature.
        self.movies["combined_features"] = (
            self.movies["title"]
            + " "
            + self.movies["genre"]
            + " "
            + self.movies["language"]
            + " "
            + self.movies["director"]
            + " "
            + self.movies["cast"]
            + " "
            + self.movies["description"]
        )

    def _train_model(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["combined_features"])
        self.cosine_similarities = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _movie_to_dict(self, movie_row):
        """Convert a pandas row into a simple JSON/template friendly dictionary."""
        return {
            "title": movie_row["title"],
            "genre": movie_row["genre"],
            "language": movie_row["language"],
            "year": int(movie_row["year"]),
            "rating": float(movie_row["rating"]),
            "director": movie_row["director"],
            "cast": movie_row["cast"],
            "description": movie_row["description"],
        }

    def get_all_movies(self):
        return [self._movie_to_dict(row) for _, row in self.movies.iterrows()]

    def get_filter_options(self):
        """Return unique genres and languages for dropdowns on the home page."""
        genres = sorted(
            {
                genre.strip()
                for value in self.movies["genre"]
                for genre in str(value).split("/")
                if genre.strip()
            }
        )
        languages = sorted(
            {
                language.strip()
                for language in self.movies["language"]
                if str(language).strip()
            }
        )
        return {"genres": genres, "languages": languages}

    def recommend_by_title(self, movie_title, top_n=5):
        """
        Recommend movies similar to the given title.

        The method first tries an exact title match, then a partial match to keep
        the web form beginner-friendly.
        """
        if not movie_title:
            return []

        title_lower = movie_title.strip().lower()
        exact_matches = self.movies[
            self.movies["title"].str.lower().str.strip() == title_lower
        ]

        if exact_matches.empty:
            exact_matches = self.movies[
                self.movies["title"].str.lower().str.contains(title_lower, regex=False)
            ]

        if exact_matches.empty:
            return []

        movie_index = exact_matches.index[0]
        similarity_scores = list(enumerate(self.cosine_similarities[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)

        recommendations = []
        for index, _score in similarity_scores:
            if index == movie_index:
                continue
            recommendations.append(self._movie_to_dict(self.movies.iloc[index]))
            if len(recommendations) == top_n:
                break

        return recommendations

    def filter_movies(self, filters, limit=None):
        """Filter movies using manual form values or NLP-extracted values."""
        filtered_movies = self.movies.copy()

        genre = filters.get("genre")
        if genre:
            filtered_movies = filtered_movies[
                filtered_movies["genre"].str.contains(str(genre), case=False, na=False)
            ]

        language = filters.get("language")
        if language:
            filtered_movies = filtered_movies[
                filtered_movies["language"].str.contains(str(language), case=False, na=False)
            ]

        director = filters.get("director")
        if director:
            filtered_movies = filtered_movies[
                filtered_movies["director"].str.contains(str(director), case=False, na=False)
            ]

        actor = filters.get("actor")
        if actor:
            filtered_movies = filtered_movies[
                filtered_movies["cast"].str.contains(str(actor), case=False, na=False)
            ]

        year_after = filters.get("year_after")
        if year_after is not None:
            filtered_movies = filtered_movies[filtered_movies["year"] > int(year_after)]

        year_before = filters.get("year_before")
        if year_before is not None:
            filtered_movies = filtered_movies[filtered_movies["year"] < int(year_before)]

        rating_above = filters.get("rating_above")
        if rating_above is not None:
            filtered_movies = filtered_movies[filtered_movies["rating"] >= float(rating_above)]

        filtered_movies = filtered_movies.sort_values(
            by=["rating", "year"], ascending=[False, False]
        )

        if limit:
            filtered_movies = filtered_movies.head(limit)

        return [self._movie_to_dict(row) for _, row in filtered_movies.iterrows()]

    def search_by_query(self, query, limit=None):
        filters = parse_natural_language_query(query)
        return self.filter_movies(filters, limit=limit), filters
