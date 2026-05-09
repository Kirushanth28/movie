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

# Mood words are mapped to tags stored in dataset/movies.csv.
# A single feeling can map to several useful recommendation tags.
MOOD_MAP = {
    "sad": ["sad", "emotional", "comfort", "feel-good"],
    "upset": ["sad", "comfort", "feel-good"],
    "depressed": ["sad", "comfort", "feel-good"],
    "tired": ["tired", "relaxing", "comfort"],
    "stressed": ["stressed", "relaxing", "comfort"],
    "stress": ["stressed", "relaxing", "comfort"],
    "anxious": ["anxious", "relaxing", "comfort"],
    "anxiety": ["anxious", "relaxing", "comfort"],
    "bored": ["bored", "exciting", "comedy"],
    "boring": ["bored", "exciting", "comedy"],
    "lonely": ["lonely", "feel-good", "romantic", "comfort"],
    "happy": ["happy", "comedy", "feel-good"],
    "excited": ["exciting", "adventure", "action"],
    "bad day": ["comfort", "feel-good", "comedy"],
    "motivated": ["motivated", "inspiring"],
    "motivation": ["motivated", "inspiring"],
    "motivational": ["motivated", "inspiring"],
    "inspired": ["inspiring", "motivated"],
    "family": ["family", "feel-good", "animation"],
    "cry": ["sad", "emotional", "romantic"],
    "romantic": ["romantic", "emotional"],
    "relax": ["relaxing", "comfort"],
    "relaxing": ["relaxing", "comfort"],
    "calm": ["relaxing", "comfort"],
    "funny": ["comedy", "happy", "feel-good"],
    "laugh": ["comedy", "happy", "feel-good"],
    "feel-good": ["feel-good", "comfort", "happy"],
    "comforting": ["comfort", "feel-good"],
}

MOOD_INTENT_WORDS = list(MOOD_MAP.keys())


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


def extract_filters_from_query(query):
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


def parse_natural_language_query(query):
    """Backward-compatible name for the filter extraction function."""
    return extract_filters_from_query(query)


def extract_mood_from_query(query):
    """
    Detect emotional words in the query and return unique mood tags.

    Example:
    "I am tired and sad" becomes
    ["tired", "relaxing", "comfort", "sad", "emotional", "feel-good"].
    """
    if not query:
        return []

    query_lower = query.lower()
    detected_tags = []
    detected_mood_words = []

    for mood_word, mapped_tags in MOOD_MAP.items():
        # Phrases such as "bad day" need a simple substring check.
        if " " in mood_word or "-" in mood_word:
            position = query_lower.find(mood_word)
            found = position != -1
        else:
            match = re.search(rf"\b{re.escape(mood_word)}\b", query_lower)
            found = match is not None
            position = match.start() if match else -1

        if found:
            detected_mood_words.append((position, mood_word, mapped_tags))

    # Process mood words in the order the user typed them.
    detected_mood_words.sort(key=lambda item: item[0])
    for _position, _mood_word, mapped_tags in detected_mood_words:
        for tag in mapped_tags:
            if tag not in detected_tags:
                detected_tags.append(tag)

    return detected_tags


def detect_user_intent(query):
    """
    Decide which type of search the user wants.

    Mood words are handled first so the existing natural language box can also
    recommend movies for emotional prompts like "I feel stressed".
    """
    if not query:
        return "filter_search"

    mood_tags = extract_mood_from_query(query)
    if mood_tags:
        return "mood_recommendation"

    query_lower = query.lower()
    title_words = ["similar to", "like movie", "like the movie", "recommend like"]
    if any(words in query_lower for words in title_words):
        return "title_recommendation"

    return "filter_search"
