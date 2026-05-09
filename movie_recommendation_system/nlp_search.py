import re


# Common genres and languages used by the sample dataset.
# More values can be added here when the dataset grows.
GENRE_KEYWORDS = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "biography": "Biography",
    "comedy": "Comedy",
    "crime": "Crime",
    "drama": "Drama",
    "fantasy": "Fantasy",
    "horror": "Horror",
    "mystery": "Mystery",
    "romance": "Romance",
    "sci-fi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "thriller": "Thriller",
}

LANGUAGE_KEYWORDS = [
    "english",
    "french",
    "german",
    "hindi",
    "japanese",
    "korean",
    "spanish",
]


def _clean_person_name(name):
    """Remove trailing filter words from a director or actor name."""
    if not name:
        return None

    # Stop when another filter phrase starts, for example:
    # "Christopher Nolan after 2010" -> "Christopher Nolan"
    stop_words = [
        " after ",
        " before ",
        " with rating ",
        " rating ",
        " genre ",
        " language ",
    ]

    cleaned_name = f" {name.strip()} "
    lower_name = cleaned_name.lower()

    for stop_word in stop_words:
        position = lower_name.find(stop_word)
        if position != -1:
            cleaned_name = cleaned_name[:position]
            break

    cleaned_name = cleaned_name.strip(" .,-")
    return cleaned_name.title() if cleaned_name else None


def parse_natural_language_query(query):
    """
    Extract simple movie filters from a natural language search query.

    This is intentionally regex-based so students can explain the logic easily
    during a viva without needing a complex NLP library.
    """
    filters = {
        "genre": None,
        "language": None,
        "year_after": None,
        "year_before": None,
        "rating_above": None,
        "actor": None,
        "director": None,
    }

    if not query:
        return filters

    query_lower = query.lower()

    for keyword, genre in GENRE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
            filters["genre"] = genre
            break

    for language in LANGUAGE_KEYWORDS:
        if re.search(rf"\b{re.escape(language)}\b", query_lower):
            filters["language"] = language.title()
            break

    year_after_match = re.search(
        r"\b(?:after|since|from|released after)\s+(\d{4})\b", query_lower
    )
    if year_after_match:
        filters["year_after"] = int(year_after_match.group(1))

    year_before_match = re.search(
        r"\b(?:before|until|prior to|released before)\s+(\d{4})\b", query_lower
    )
    if year_before_match:
        filters["year_before"] = int(year_before_match.group(1))

    rating_match = re.search(
        r"\b(?:rating|rated|score)\s*(?:above|over|greater than|more than|at least)?\s*(\d+(?:\.\d+)?)\b",
        query_lower,
    )
    if rating_match:
        filters["rating_above"] = float(rating_match.group(1))
    elif "high rating" in query_lower or "highly rated" in query_lower:
        filters["rating_above"] = 7.5

    director_match = re.search(
        r"\b(?:by director|director|directed by)\s+([a-zA-Z .'-]+)", query
    )
    if director_match:
        filters["director"] = _clean_person_name(director_match.group(1))

    actor_match = re.search(
        r"\b(?:actor|starring|featuring)\s+([a-zA-Z .'-]+)", query
    )
    if actor_match:
        filters["actor"] = _clean_person_name(actor_match.group(1))

    return filters
