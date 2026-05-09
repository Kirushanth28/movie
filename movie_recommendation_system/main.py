from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recommender import DatasetError, MovieRecommender


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "movies.csv"

app = FastAPI(
    title="NLP-Based Intelligent Movie Recommendation System",
    description="A simple FastAPI project for content-based movie recommendations and NLP search.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

try:
    recommender = MovieRecommender(DATASET_PATH)
except DatasetError as exc:
    # Failing early gives a clear message if the CSV file is missing or invalid.
    raise RuntimeError(str(exc)) from exc


def _to_int_or_none(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_float_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _home_context(request):
    options = recommender.get_filter_options()
    return {
        "request": request,
        "genres": options["genres"],
        "languages": options["languages"],
    }


def _results_context(request, title, results, message=None, query=None, filters=None):
    return {
        "request": request,
        "title": title,
        "results": results,
        "message": message,
        "query": query,
        "filters": filters,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with all three search forms."""
    return templates.TemplateResponse(request, "index.html", _home_context(request))


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, movie_title: str = Form(...)):
    """Handle recommendation form submission by movie title."""
    results = recommender.recommend_by_title(movie_title, top_n=5)
    message = None

    if not results:
        message = f"No recommendations found. Movie title '{movie_title}' is not in the dataset."

    context = _results_context(
        request=request,
        title="Recommended Movies",
        results=results,
        message=message,
        query=movie_title,
    )
    return templates.TemplateResponse(request, "results.html", context)


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    """Handle natural language movie search form submission."""
    results, filters = recommender.search_by_query(query)
    message = None

    if not results:
        message = "No movies matched your natural language search."

    context = _results_context(
        request=request,
        title="Natural Language Search Results",
        results=results,
        message=message,
        query=query,
        filters=filters,
    )
    return templates.TemplateResponse(request, "results.html", context)


@app.post("/filter", response_class=HTMLResponse)
async def filter_movies(
    request: Request,
    genre: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    year_after: Optional[str] = Form(None),
    year_before: Optional[str] = Form(None),
    rating_above: Optional[str] = Form(None),
):
    """Handle manual filter form submission."""
    filters = {
        "genre": genre,
        "language": language,
        "year_after": _to_int_or_none(year_after),
        "year_before": _to_int_or_none(year_before),
        "rating_above": _to_float_or_none(rating_above),
    }

    results = recommender.filter_movies(filters)
    message = None

    if not results:
        message = "No movies matched the selected filters."

    context = _results_context(
        request=request,
        title="Manual Filter Results",
        results=results,
        message=message,
        filters=filters,
    )
    return templates.TemplateResponse(request, "results.html", context)


@app.get("/api/movies")
async def api_movies():
    """Return all movies as JSON."""
    movies = recommender.get_all_movies()
    return {"count": len(movies), "movies": movies}


@app.get("/api/recommend/{movie_title}")
async def api_recommend(movie_title: str):
    """Return movie recommendations as JSON."""
    results = recommender.recommend_by_title(movie_title, top_n=5)
    response = {
        "query": movie_title,
        "count": len(results),
        "recommendations": results,
    }

    if not results:
        response["message"] = f"Movie title '{movie_title}' was not found in the dataset."

    return response


@app.get("/api/search")
async def api_search(query: str = Query(..., description="Natural language movie query")):
    """Return natural language movie search results as JSON."""
    results, filters = recommender.search_by_query(query)
    response = {
        "query": query,
        "extracted_filters": filters,
        "count": len(results),
        "movies": results,
    }

    if not results:
        response["message"] = "No movies matched your search query."

    return response
