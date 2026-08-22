"""
Daily Resolution Job

Run once a day (after the market-creation job). Locks due markets,
then attempts Tier 1 resolution on everything ready to check.
"""

import logging
from db_storage import (
    lock_due_markets, get_markets_ready_to_resolve,
    mark_resolved, mark_void, mark_needs_review,
    increment_days_overdue
)
from tier1_resolver import resolve_market

logger = logging.getLogger("FanGram.DailyResolve")

MAX_OVERDUE_DAYS = 3  # hard ceiling — never more than this many unresolved checks

def run_daily_resolution(db_path: str) -> None:
    locked_count = lock_due_markets(db_path)
    logger.info(f"Locked {locked_count} market(s) whose betting window closed.")

    due_markets = get_markets_ready_to_resolve(db_path)
    logger.info(f"Checking {len(due_markets)} market(s) at/past resolution deadline.")

    for market in due_markets:
        result = resolve_market(market)

        if result["status"] == "RESOLVED":
            mark_resolved(
                db_path, market["market_id"],
                resolved_value=result["value"],
                resolution_outcome=result["outcome"],
                resolution_tier="TIER_1_DETERMINISTIC",
            )
            continue

        # PENDING or ERROR — both advance the same shared clock
        streak = increment_days_overdue(db_path, market["market_id"])

        if streak >= MAX_OVERDUE_DAYS:
            if result["status"] == "ERROR":
                mark_needs_review(
                    db_path, market["market_id"],
                    reason=f"Fetch/parse failed; unresolved after {streak} day(s) overdue.",
                )
            else:  # PENDING
                mark_void(
                    db_path, market["market_id"],
                    reason=f"No data for '{market['resolution_field']}' "
                           f"after {streak} day(s) overdue.",
                )
        else:
            logger.info(
                f"'{market['movie_name']}' [{market['metric_id']}] still unresolved "
                f"({result['status']}, {streak}/{MAX_OVERDUE_DAYS} days overdue)."
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_resolution("fangram_markets.db")