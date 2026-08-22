"""
test_pipeline.py — Daily Market Creation

Run once a day. Discovers new upcoming movies, enriches them, generates
calibrated questions (with two-tier advance-booking grounding), and
saves each as a new OPEN market.
"""

import os
import uuid
import logging

from openai import OpenAI
from tavily import TavilyClient

from movie_extractor import get_upcoming_movies              # Section 1
from context_enrichment import enrich_with_cast_info         # Section 2
from ques_gen import generate_questions_for_movie, build_final_questions  # Section 3
from metric_registry import METRIC_REGISTRY
from db_storage import init_db, save_market

logger = logging.getLogger("FanGram.DailyCreate")


def _registry_lookup(metric_id: str) -> dict:
    """Finds a metric's full registry entry by its id."""
    for entry in METRIC_REGISTRY:
        if entry["metric_id"] == metric_id:
            return entry
    raise KeyError(f"Unknown metric_id '{metric_id}' — not in METRIC_REGISTRY.")


def run_daily_market_creation(db_path: str, llm_client: OpenAI, tavily_client: TavilyClient) -> None:
    init_db(db_path)

    new_movies = get_upcoming_movies()
    logger.info(f"Found {len(new_movies)} new movie(s) to process.")

    for movie in new_movies:
        movie = enrich_with_cast_info(movie)

        generated = generate_questions_for_movie(movie, METRIC_REGISTRY, llm_client, tavily_client)
        if generated is None:
            logger.error(f"Skipping '{movie['movie_name']}' — question generation failed.")
            continue

        final_questions = build_final_questions(movie, generated, METRIC_REGISTRY)

        for question in final_questions:
            registry_entry = _registry_lookup(question["metric_id"])

            market = {
                "market_id": f"mkt_{uuid.uuid4().hex[:10]}",
                "movie_name": movie["movie_name"],
                "movie_slug": movie["movie_slug"],
                "metric_id": question["metric_id"],
                "question_text": question["question_text"],
                "display_title": question["display_title"],
                "threshold": question["threshold"],
                "initial_yes_probability": question["initial_yes_probability"],
                "resolution_source_url": question["resolution_source_url"],
                "resolution_field": registry_entry["resolution_field"],
                "resolution_question_text": registry_entry["resolution_question_text"].format(
                    movie=movie["movie_name"]
                ),
                "betting_lock_utc": question["betting_lock_utc"],
                "resolution_deadline_utc": question["resolution_deadline_utc"],
            }

            save_market(db_path, market)
            logger.info(f"Saved market '{market['market_id']}': {market['question_text']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    run_daily_market_creation("fangram_markets.db", llm_client, tavily_client)