# NLP-Based Intelligent Movie Recommendation System Using FastAPI

This is a complete bachelor-level project that recommends movies using a content-based machine learning approach and supports natural language search using simple regex-based NLP.

The application is offline and uses a local CSV dataset. It does not require a database or any external movie API.

## Features

- FastAPI backend with automatic `/docs` API documentation
- HTML frontend using Jinja2 templates
- Static CSS styling with responsive movie cards
- Local CSV dataset loading from `dataset/movies.csv`
- Dataset validation for required columns
- Missing value handling
- Content-based recommendation model using:
  - TF-IDF Vectorizer
  - Cosine Similarity
- Movie recommendation by title
- Natural language movie search
- Manual filter search
- JSON API endpoints for movies, recommendations, and NLP search
- Clear no-result messages and dataset error handling

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Regex-based NLP
- Jinja2
- HTML
- CSS

## Folder Structure

```text
movie_recommendation_system/
│
├── main.py
├── recommender.py
├── nlp_search.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── movies.csv
│
├── templates/
│   ├── index.html
│   └── results.html
│
└── static/
    └── style.css
```

## Dataset Columns

The CSV file must contain these columns:

- `title`
- `genre`
- `language`
- `year`
- `rating`
- `director`
- `cast`
- `description`

## Installation Steps

1. Open a terminal inside the project folder:

   ```bash
   cd movie_recommendation_system
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   On Linux or macOS:

   ```bash
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

4. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Run the FastAPI server with:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Example Search Queries

Try these in the natural language search form:

- Show me English action movies after 2010 with high rating
- Recommend comedy movies with rating above 7
- Find Korean thriller movies after 2015
- Show sci-fi movies before 2020
- Recommend movies by director Christopher Nolan
- Find movies starring Keanu Reeves

## API Endpoints

### Web Pages

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Home page |
| POST | `/recommend` | Submit movie title recommendation form |
| POST | `/search` | Submit natural language search form |
| POST | `/filter` | Submit manual filter form |

### JSON APIs

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/movies` | Return all movies |
| GET | `/api/recommend/{movie_title}` | Return top 5 recommendations for a movie title |
| GET | `/api/search?query=` | Return NLP search results |

Example:

```text
http://127.0.0.1:8000/api/search?query=Find Korean thriller movies after 2015
```

## How the Recommendation Model Works

1. The application reads `dataset/movies.csv` using pandas.
2. It checks whether all required columns exist.
3. Missing text values are replaced with empty strings.
4. The system combines title, genre, language, director, cast, and description into one text feature.
5. TF-IDF converts the combined text into numeric vectors.
6. Cosine similarity measures how similar each movie is to every other movie.
7. When a user enters a movie title, the system returns the top 5 most similar movies.

## How the NLP Search Works

The NLP module uses regular expressions to extract simple filters from a sentence:

- genre
- language
- year after
- year before
- rating above
- actor
- director

For example:

```text
Recommend movies by director Christopher Nolan
```

The system extracts:

```text
director = Christopher Nolan
```

Then it filters the movie dataset using that extracted value.

## Future Enhancements

- Add user login and saved favorite movies
- Store movies and users in a database
- Add posters or image upload support
- Add pagination for large datasets
- Improve NLP using spaCy or transformer models
- Add collaborative filtering based on user ratings
- Add admin pages for managing the movie dataset
