"""
Section 1: BollywoodHungama Release Calendar Ingestion

Fetches the BH "Movie Release Dates" page, finds movies releasing within
the next few days that we haven't already reported, and remembers what's
already been reported (on disk) so re-runs don't repeat themselves.
"""

""" INFORMATION FLOW (for context):
[BollywoodHungama Web Page]
       │
       ▼ (HTTP GET via requests)
   [Raw HTML]
       │
       ▼ (BeautifulSoup parsing)
  [Raw Table Rows] ──► [Filter: Vague Dates / Far Future / Past Dates]
       │
       ▼
 [Cleaned Rows]    ──► [Filter: Digital-Only Releases]
       │
       ▼
 [Signature Hash]  ──► [Deduplication Check against 'seen_release_hashes.json']
       │
       ▼ (If New)
  [Final New Movies List] ──► Saved to Disk & Returned to Main Pipeline
  
  """

"""OUTPUT EXAMPLE (for one movie):
[
  {
    "movie_name": "Stree 3",
    "language": "Indian",
    "digital_only": false,
    "movie_slug": "stree-3",
    "release_date": "2026-08-27"
  }
]

"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("FanGram.ReleaseCalendar")

RELEASE_DATES_URL = "https://www.bollywoodhungama.com/movie-release-dates/"
WINDOW_DAYS = 3            # report movies releasing within this many days
FAR_FUTURE_CUTOFF_DAYS = 60   # stop scanning rows once clearly irrelevant (efficiency only)


def load_seen_hashes(path: str) -> set:
    """Loads previously-reported movie hashes from a JSON file on disk."""

    if not isinstance(path, str):
        logger.error(f"Seen-hashes path must be a string, got {type(path)}")
        return set()
    
    if not os.path.exists(path):
        logger.warning(f"Seen-hashes file '{path}' does not exist; starting fresh.")
        return set()
    try:
        with open(path, "r") as f:
            return set(json.load(f))
    except Exception as e:
        logger.error(f"Failed to load seen hashes from '{path}': {e}")
        return set()


def save_seen_hashes(path: str, seen_hashes: set) -> None:
    """Saves the current set of reported movie hashes back to disk."""
    try:
        with open(path, "w") as f:
            json.dump(sorted(seen_hashes), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save seen hashes to '{path}': {e}")


def compute_signal_hash(movie_name: str, release_date_iso: str) -> str:
    """MD5 hash of normalized movie name + release date, for dedup.
    Combining both means a postponed movie (new date) naturally gets
    treated as new again."""
    clean_text = re.sub(r'[^a-z0-9]', '', f"{movie_name}{release_date_iso}".lower())
    return hashlib.md5(clean_text.encode("utf-8")).hexdigest()


def parse_release_date(date_str: str) -> Optional[datetime.date]:
    """Parses 'DD Month YYYY' into a date. Returns None for vague dates
    like 'Expected in October 2026' — those get skipped, not guessed at."""
    try:
        return datetime.strptime(date_str.strip(), "%d %B %Y").date()
    except ValueError:
        logger.warning(f"Skipping malformed date: '{date_str}'") # Expected in October 2026, etc.
        return None


def fetch_release_page(url: str, timeout: float = 10.0) -> Optional[str]:
    """Fetches the raw HTML of the release-dates page."""
    try:
        response = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "FanGram-ReleaseCalendar/1.0"} # avoid 403 from BH   
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch release page: {e}")
        return None


def parse_release_rows(html: str) -> List[Dict]:
    """Parses the release-dates table into raw row dicts.

    NOTE: assumes a standard <table>/<tr>/<td> layout with columns
    [Release Date, Movie Name, Audience Score]. I haven't seen the raw
    HTML (only a markdown-rendered version), so verify this against the
    live page and adjust the selectors if it doesn't match.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    table = soup.find("table")
    if not table:
        logger.error("Could not find release-dates table on page.")
        return rows

    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue  # header row or malformed row

        raw_date = cells[0].get_text(strip=True)
        movie_cell = cells[1]
        raw_name = movie_cell.get_text(strip=True)

        link = movie_cell.find("a")
        movie_slug = link["href"].rstrip("/").split("/")[-1] if link and link.get("href") else None

        rows.append({"raw_date": raw_date, "raw_name": raw_name, "movie_slug": movie_slug})

    logger.info(f"Found {len(rows)} raw rows (upcoming movies) in {RELEASE_DATES_URL}.")
    return rows


def clean_row(row: Dict) -> Dict:
    """Extracts language + digital-only flag from the raw movie name."""
    raw_name = row["raw_name"]

    digital_only = "[digital release only]" in raw_name.lower()
    language = "English" if "(english)" in raw_name.lower() else "Indian"

    clean_name = re.sub(r"\(english\)", "", raw_name, flags=re.IGNORECASE)
    clean_name = re.sub(r"\[digital release only\]", "", clean_name, flags=re.IGNORECASE)
    clean_name = clean_name.strip()

    return {
        "movie_name": clean_name,
        "language": language,
        "digital_only": digital_only,
        "movie_slug": row["movie_slug"],
    }


def get_upcoming_movies(
    url: str = RELEASE_DATES_URL,
    seen_hashes_path: str = "seen_release_hashes.json",
    window_days: int = WINDOW_DAYS,
) -> List[Dict]:
    """Main entry point: returns new movies releasing within `window_days`,
    and updates the on-disk seen-hashes file."""

    html = fetch_release_page(url)
    if not html:
        return []

    seen_hashes = load_seen_hashes(seen_hashes_path)
    today = datetime.now().date()
    window_end = today + timedelta(days=window_days)
    far_future = today + timedelta(days=FAR_FUTURE_CUTOFF_DAYS)

    new_movies = []

    for raw_row in parse_release_rows(html):
        release_date = parse_release_date(raw_row["raw_date"])

        if release_date is None:
            continue  # vague date ("Expected in October 2026") — skip

        if release_date > far_future:
            break  # rows are sorted ascending; nothing further matters today

        if release_date < today or release_date > window_end:
            continue  # outside our 3-day window — leave seen_hashes untouched

        row = clean_row(raw_row)
        if row["digital_only"]:
            continue  # not a theatrical release — drop

        release_date_iso = release_date.isoformat()
        sig_hash = compute_signal_hash(row["movie_name"], release_date_iso)

        if sig_hash in seen_hashes:
            continue  # already reported before

        row["release_date"] = release_date_iso
        new_movies.append(row)
        seen_hashes.add(sig_hash)  # only marked seen now — at report time

    save_seen_hashes(seen_hashes_path, seen_hashes)
    return new_movies


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    movies = get_upcoming_movies()
    print(f"Found {len(movies)} new movie(s) releasing within {WINDOW_DAYS} days:\n")
    for m in movies:
        print(m)

