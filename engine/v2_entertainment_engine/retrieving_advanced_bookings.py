"""
Advance Booking Enrichment (TrackMyShow)

Best-effort lookup of a pre-release advance booking figure for a movie.
TrackMyShow doesn't track every film (smaller/regional releases are
often absent, or present with 0 tickets) — this returns None whenever
there's no confident, usable signal, rather than guessing.
"""

import re
import logging
from difflib import SequenceMatcher
from typing import Optional, Dict

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("FanGram.AdvanceBooking")

FILMS_LISTING_URL = "https://trackmyshow.in/films/"
NAME_MATCH_THRESHOLD = 0.85  # how close a fuzzy match must be to trust it


def _fetch_html(url: str, timeout: float = 10.0) -> Optional[str]:
    try:
        response = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "FanGram-AdvanceBooking/1.0"}
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch '{url}': {e}")
        return None


BASE_URL = "https://trackmyshow.in"

def _find_film_url(movie_name: str) -> Optional[str]:
    html = _fetch_html(FILMS_LISTING_URL)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    best_match = None
    best_score = 0.0

    for link in soup.find_all("a", href=True):
        if "-box-office-collection" not in link["href"]:
            continue
        candidate_name = link.get_text(strip=True)
        if not candidate_name:
            continue

        score = SequenceMatcher(None, movie_name.lower(), candidate_name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = link["href"]

    if best_score >= NAME_MATCH_THRESHOLD:
        if best_match.startswith("/"):
            best_match = BASE_URL + best_match
        return best_match

    logger.info(f"No confident TrackMyShow match for '{movie_name}' (best score: {best_score:.2f})")
    return None


def fetch_advance_booking_estimate(movie_name: str) -> Optional[Dict]:
    """Returns {'tickets': int, 'source_snippet': str} for a pre-release
    advance booking figure, or None if unavailable."""
    film_url = _find_film_url(movie_name)
    if not film_url:
        return None

    html = _fetch_html(film_url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    match = re.search(
        r"Advance bookings reached ([\d.]+[KM]?) tickets before release",
        page_text,
    )
    if not match:
        logger.info(f"'{movie_name}' found on TrackMyShow but no advance-booking record present.")
        return None

    raw_value = match.group(1)
    multiplier = 1000 if raw_value.endswith("K") else (1_000_000 if raw_value.endswith("M") else 1)
    tickets = int(float(raw_value.rstrip("KM")) * multiplier)

    return {"tickets": tickets, "source_snippet": match.group(0)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for name in ["Awarapan 2", "Harrd Disk"]:
        result = fetch_advance_booking_estimate(name)
        print(f"{name}: {result}")