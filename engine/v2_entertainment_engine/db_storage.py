"""
Section 4/5: Market Storage (SQLite)

One table, plain functions — no class, since there's no state to hold
between calls beyond the db file itself (same as seen_hashes_path in
Section 1: passed as a parameter, not stored on an object).
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger("FanGram.MarketDB")

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    movie_name TEXT NOT NULL,
    movie_slug TEXT NOT NULL,
    metric_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    display_title TEXT,
    threshold REAL NOT NULL,
    initial_yes_probability REAL NOT NULL,
    resolution_source_url TEXT NOT NULL,
    resolution_field TEXT NOT NULL,
    resolution_question_text TEXT NOT NULL,
    betting_lock_utc TEXT NOT NULL,
    resolution_deadline_utc TEXT NOT NULL,
    market_status TEXT NOT NULL DEFAULT 'OPEN',
    resolved_value REAL,
    resolution_outcome TEXT,
    resolution_tier TEXT,
    brier_score REAL,
    days_overdue INTEGER NOT NULL DEFAULT 0,
    void_reason TEXT,
    created_at_utc TEXT NOT NULL
);
"""


def init_db(db_path: str) -> None:
    """Creates the markets table if it doesn't already exist. Safe to
    call every run — CREATE TABLE IF NOT EXISTS is a no-op if present."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)


def save_market(db_path: str, market: Dict) -> None:
    """Inserts one new market, status defaults to OPEN."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO markets (
                market_id, movie_name, movie_slug, metric_id, question_text,
                display_title, threshold, initial_yes_probability,
                resolution_source_url, resolution_field, resolution_question_text,
                betting_lock_utc, resolution_deadline_utc,
                market_status, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
            """,
            (
                market["market_id"], market["movie_name"], market["movie_slug"],
                market["metric_id"], market["question_text"], market["display_title"],
                market["threshold"], market["initial_yes_probability"],
                market["resolution_source_url"], market["resolution_field"],
                market["resolution_question_text"], market["betting_lock_utc"], market["resolution_deadline_utc"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def lock_due_markets(db_path: str) -> int:
    """Moves OPEN markets whose betting_lock_utc has passed into LOCKED.
    Returns how many were locked."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE markets SET market_status = 'LOCKED' "
            "WHERE market_status = 'OPEN' AND betting_lock_utc <= ?",
            (now,),
        )
        return cursor.rowcount


def get_markets_ready_to_resolve(db_path: str) -> List[Dict]:
    """Returns LOCKED markets whose resolution_deadline_utc has passed."""
    today = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM markets WHERE market_status = 'LOCKED' "
            "AND resolution_deadline_utc <= ?",
            (today,),
        ).fetchall()
        return [dict(row) for row in rows]


def mark_resolved(
    db_path: str,
    market_id: str,
    resolved_value: float,
    resolution_outcome: str,  # 'YES' or 'NO'
    resolution_tier: str,     # 'TIER_1_DETERMINISTIC' or 'TIER_2_LLM_SEARCH'
) -> None:
    """Marks a market resolved and computes its Brier score against the
    initial_yes_probability stored at creation time."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT initial_yes_probability FROM markets WHERE market_id = ?",
            (market_id,),
        ).fetchone()
        if not row:
            logger.error(f"Cannot resolve unknown market_id '{market_id}'")
            return

        initial_prob = row[0]
        actual = 1.0 if resolution_outcome == "YES" else 0.0
        brier_score = (initial_prob - actual) ** 2

        conn.execute(
            """
            UPDATE markets SET
                market_status = 'RESOLVED',
                resolved_value = ?,
                resolution_outcome = ?,
                resolution_tier = ?,
                brier_score = ?
            WHERE market_id = ?
            """,
            (resolved_value, resolution_outcome, resolution_tier, brier_score, market_id),
        )
        logger.info(f"Resolved '{market_id}': {resolution_outcome} (Brier: {brier_score:.3f})")


def mark_unresolvable(db_path: str, market_id: str, reason: str) -> None:
    """Tier 3 fallback: no clean data, no LLM+search rescue either."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE markets SET market_status = 'NEEDS_REVIEW', resolution_tier = ? WHERE market_id = ?",
            (f"UNRESOLVED: {reason}", market_id),
        )
        logger.warning(f"Market '{market_id}' could not be resolved: {reason}")


def increment_days_overdue(db_path: str, market_id: str) -> int:
    """Bumps days_overdue by 1 on any non-resolved check, regardless
    of whether it was PENDING or ERROR. Returns the new count."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE markets SET days_overdue = days_overdue + 1 WHERE market_id = ?",
            (market_id,),
        )
        row = conn.execute(
            "SELECT days_overdue FROM markets WHERE market_id = ?", (market_id,)
        ).fetchone()
        return row[0]


def mark_void(db_path: str, market_id: str, reason: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE markets SET market_status = 'VOID', void_reason = ? WHERE market_id = ?",
            (reason, market_id),
        )
        logger.warning(f"Market '{market_id}' voided: {reason}")


def mark_needs_review(db_path: str, market_id: str, reason: str) -> None:
    """Persistent ERROR streak — likely means the parser broke, not
    that data is unavailable. Flag for a human, don't void."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE markets SET market_status = 'NEEDS_REVIEW', void_reason = ? WHERE market_id = ?",
            (reason, market_id),
        )
        logger.error(f"Market '{market_id}' needs review: {reason}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db("fangram_markets.db")
    print("DB initialized.")