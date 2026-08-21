"""
Section 6: Box Office Resolution Parser (Tier 1 - Deterministic)

Parses a movie's BollywoodHungama box-office page and extracts the
INDIA BOX OFFICE COLLECTION table for comparison against a market's
threshold. Tier 1 only — deterministic parse-and-compare. Escalation
to Tier 2 (web search + LLM) is a decision made by the caller when
this returns PENDING past a grace period, not handled in here.
"""

import re
import logging
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("FanGram.Resolver")


def fetch_box_office_page(slug: str, timeout: float = 10.0) -> Optional[str]: # takes in a movie slug like "awarapan-2" and returns the HTML of its box office page, or None on failure
    url = f"https://www.bollywoodhungama.com/movie/{slug}/box-office/"
    try:
        response = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "FanGram-Resolver/1.0"}
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch box office page for '{slug}': {e}")
        return None


def _extract_amount(text: str) -> Optional[float]: # Parses a table cell's amount text into a float
    """Parses a table cell's amount text into a float. Returns None
    for missing data ('No Data Found') or a range ('182 - 185 cr.') —
    neither is a single comparable value, so both are treated as
    'not usable yet' rather than guessed at."""
    text = text.strip()

    if "no data" in text.lower():
        return None

    if "-" in text.replace("cr.", "").strip():
        logger.warning(f"Amount looks like a range, not a single value: '{text}'")
        return None

    match = re.search(r"[\d.]+", text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_box_office_table(html: str) -> Dict[str, Optional[float]]: # Parses the HTML of a BollywoodHungama box-office page and extracts the INDIA BOX OFFICE COLLECTION table into a dictionary of label -> amount. Returns an empty dict on failure.
    """Finds the INDIA BOX OFFICE COLLECTION table specifically — the
    page has several tables (Day Wise, Week Wise, Weekend too), so we
    locate this one by its heading text, not by grabbing the first
    <table> on the page."""
    soup = BeautifulSoup(html, "html.parser")

    heading = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and "india box office collection" in tag.get_text(strip=True).lower()
    )
    if not heading:
        logger.error("Could not find 'INDIA BOX OFFICE COLLECTION' heading on page.")
        return {}

    table = heading.find_next("table")
    if not table:
        logger.error("Found heading but no table followed it.")
        return {}

    values = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)   # handles the nested <a> tag fine
        amount_text = cells[1].get_text(strip=True)
        values[label] = _extract_amount(amount_text)

    return values


def resolve_market(market: Dict) -> Dict:
    """Attempts Tier 1 deterministic resolution for one market.

    Returns one of:
      {'status': 'RESOLVED', 'value': float, 'outcome': 'YES'/'NO'}
      {'status': 'PENDING'}   — data not published yet, try again later
      {'status': 'ERROR'}     — fetch or parse failed outright
    """
    html = fetch_box_office_page(market["movie_slug"])
    if not html:
        return {"status": "ERROR"}

    table_values = parse_box_office_table(html)
    if not table_values:
        return {"status": "ERROR"}

    value = table_values.get(market["resolution_field"])

    if value is None:
        logger.info(f"'{market['movie_name']}' — no usable value yet for '{market['resolution_field']}'.")
        return {"status": "PENDING"}

    outcome = "YES" if value >= market["threshold"] else "NO"
    logger.info(
        f"RESOLVED '{market['movie_name']}' [{market['metric_id']}]: "
        f"{value} vs threshold {market['threshold']} -> {outcome}"
    )
    return {"status": "RESOLVED", "value": value, "outcome": outcome}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_market = {
        "movie_name": "Dhurandhar: The Revenge",
        "movie_slug": "dhurandhar-the-revenge",
        "metric_id": "day1_threshold",
        "resolution_field": "Opening Day",
        "threshold": 80.0,  # real value is 21.89 -> expect YES
    }
    print(resolve_market(test_market))