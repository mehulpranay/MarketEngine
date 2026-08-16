import re
import hashlib
import logging
from typing import List, Set, Optional
from datetime import datetime, timezone
import feedparser
import requests
from pydantic import BaseModel, Field

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FanGram.Ingester")

class RawSignal(BaseModel):
    """Pydantic model defining the data contract for ingested news signals."""
    signal_hash: str = Field(description="MD5 hash of normalized title for basic deduplication")
    title: str = Field(description="Raw headline text from RSS feed")
    link: str = Field(description="Source URL link")
    published_utc: str = Field(description="Standardized ISO 8601 UTC timestamp")
    source_name: str = Field(default="Google News")

class IngestionEngine:
    """Production RSS Ingester built with network timeouts, feedparser error isolation, 

    and lightweight hash-based deduplication.
    """

    DEFAULT_RSS_URL = (
        "https://news.google.com/rss/search?"
        "q=Bollywood+or+Movie+or+Cinema+when:24h&hl=en-IN&gl=IN&ceid=IN:en"
    )

    def __init__(self, seen_hashes: Optional[Set[str]] = None):
        self.seen_hashes: Set[str] = seen_hashes if seen_hashes is not None else set()

    @staticmethod
    def generate_signal_hash(title: str) -> str:
        """Normalizes headline text to create a simple collision-resistant hash."""
        clean_text = re.sub(r'[^a-z0-9]', '', title.lower())
        return hashlib.md5(clean_text.encode('utf-8')).hexdigest()

    def fetch_rss_feed(
        self, rss_url: str = DEFAULT_RSS_URL, timeout: float = 5.0
    ) -> List[RawSignal]:
        """Fetches and validates RSS items with strict network timeouts and parsing boundaries."""
        signals: List[RawSignal] = []

        try:
            logger.info(f"Fetching RSS feed: {rss_url}")
            response = requests.get(
                rss_url,
                timeout=timeout,
                headers={"User-Agent": "FanGram-Ingester/1.0"}
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"HTTP request failed during RSS fetch: {e}")
            return []  # Defensive degradation: return empty list on network error

        # Parse content using feedparser
        feed = feedparser.parse(response.content)

        if feed.bozo:
            logger.warning(
                f"Feedparser flagged structural formatting issues: "
                f"{feed.get('bozo_exception')}"
            )

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()

            # Defense 1: Reject entries missing core attributes
            if not title or not link:
                continue

            # Defense 2: Deduplication via exact title hash
            sig_hash = self.generate_signal_hash(title)
            if sig_hash in self.seen_hashes:
                logger.debug(f"Skipping duplicate title hash: '{title[:40]}...'")
                continue

            # Defensive Timestamp Parsing
            published_utc = datetime.now(timezone.utc).isoformat()
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    published_utc = dt.isoformat()
                except Exception:
                    pass  # Fall back to current UTC timestamp on parse failure

            try:
                signal = RawSignal(
                    signal_hash=sig_hash,
                    title=title,
                    link=link,
                    published_utc=published_utc,
                    source_name=getattr(entry, "source", {}).get("title", "Google News")
                )
                signals.append(signal)
                self.seen_hashes.add(sig_hash)
            except Exception as e:
                logger.error(f"Pydantic data contract validation failed for entry: {e}")
                continue

        logger.info(f"Ingested {len(signals)} valid, non-duplicate signals.")
        return signals