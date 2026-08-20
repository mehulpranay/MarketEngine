"""
Section 2: BollywoodHungama Cast & Genre Enrichment

For each movie from Section 1, fetches its BH cast page and adds
genre, lead cast, and a confirmed language field.
"""

import logging
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("FanGram.CastEnrichment")

CAST_PAGE_TEMPLATE = "https://www.bollywoodhungama.com/movie/{slug}/cast/"


def fetch_cast_page(slug: str, timeout: float = 10.0) -> Optional[str]:
    """Fetches the raw HTML of a movie's BH cast page."""
    url = CAST_PAGE_TEMPLATE.format(slug=slug)
    try:
        response = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "FanGram-CastEnrichment/1.0"}
        )
        response.raise_for_status() # raise an exception for 4xx/5xx responses (without it , the requests will just return the html page with 404/500 error message, which will break the parsing logic)
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch cast page for '{slug}': {e}")
        return None


def _find_section_items(soup: BeautifulSoup, heading_text: str) -> List[str]: # heading_text is the text of the heading to look for, e.g., "Genre", "Primary Starcast", "Language"
    """Finds a heading matching `heading_text` (case-insensitive) and
    returns the text of each item in the list that follows it.

    NOTE: I only saw a markdown-rendered version of this page, not raw
    HTML, so this matches by heading *text* rather than a specific CSS
    class/tag — more likely to survive not knowing the real structure,
    but still needs confirming against a live run.
    """
    heading = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4", "dt")
        and tag.get_text(strip=True).lower() == heading_text.lower()
    )
    if not heading:
        return []

    list_container = heading.find_next(["ul", "dd"])
    if not list_container:
        return []

    if list_container.name == "ul":
        return [li.get_text(strip=True) for li in list_container.find_all("li")]
    return [list_container.get_text(strip=True)]


def enrich_with_cast_info(movie: Dict) -> Dict:
    """Takes a movie dict (from Section 1) and returns a copy with
    genre, primary_cast"""
    enriched = dict(movie)  # don't mutate the caller's dict

    html = fetch_cast_page(movie["movie_slug"])
    if not html:
        enriched["genre"] = []
        enriched["primary_cast"] = []
        return enriched

    soup = BeautifulSoup(html, "html.parser")

    enriched["genre"] = _find_section_items(soup, "Genre")
    enriched["primary_cast"] = _find_section_items(soup, "Primary Starcast")

    return enriched


def enrich_movies(movies: List[Dict]) -> List[Dict]:
    """Runs enrich_with_cast_info over a list of movies from Section 1."""
    return [enrich_with_cast_info(m) for m in movies]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_movie = {"movie_name": "Pushpa", "movie_slug": "pushpa", "language": "Indian"}
    result = enrich_with_cast_info(test_movie)
    print(result)