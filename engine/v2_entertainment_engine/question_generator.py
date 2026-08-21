"""
Section 3: Question Generation

Step A (LLM): calibrate a ₹ Cr threshold per metric for one movie's scale.
Step B (pure Python): fill the registry's question template, and compute
betting_lock_utc / resolution_deadline_utc via date math — no LLM involved
in either the wording or the timestamps.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from pydantic import BaseModel, Field
from openai import OpenAI
from retrieving_advanced_bookings import fetch_advance_booking_estimate

logger = logging.getLogger("FanGram.QuestionGeneration")

IST_OFFSET = timedelta(hours=5, minutes=30)


class QuestionSelection(BaseModel):
    """One LLM-calibrated threshold for one registry metric."""
    metric_id: str = Field(description="Must match one of the metric_ids provided in the prompt.")
    threshold: float = Field(description="Sensible ₹ Cr threshold for THIS movie's scale, based on genre/cast/language provided.")
    initial_yes_probability: float = Field(
        description="Your honest calibrated probability (0.05-0.95) that the actual result will meet or exceed the threshold. Do NOT default to 0.5 — reason about it genuinely."
    )
    reasoning: str = Field(description="One sentence: why this threshold fits this movie's expected scale.")
    display_title: str = Field(description="Short, catchy market title for UI display.")

class GeneratedQuestionSet(BaseModel):
    movie_name: str
    selections: List[QuestionSelection]


ADVANCE_TICKET_RATIO_LOW, ADVANCE_TICKET_RATIO_HIGH = 0.0995, 0.1409



SYSTEM_PROMPT = """You are calibrating box-office prediction thresholds for an Indian entertainment market platform.

You will be given a movie's name, language, genre, primary cast, and a list of metric_ids with their question templates. For EACH metric_id, output a threshold (in ₹ Cr) that would make an INTERESTING, genuinely uncertain YES/NO question for THIS SPECIFIC movie — not a threshold so low it's obviously YES, or so high it's obviously NO.

Use these signals to judge scale:
- Big-name lead cast (established stars) implies higher realistic thresholds.
- English-language/Hollywood titles typically post much lower India Net figures than major Hindi star vehicles.
- Genre alone is a weaker signal than cast — an action/franchise title with an unknown cast should not automatically get a high threshold.

If an ADVANCE BOOKING ANCHOR is provided in the prompt, treat it as your strongest signal — it's a real, grounded estimate of Day 1 India Net revenue for THIS movie, not a guess. Use it to calibrate the day1_threshold directly (set it near, not far below, the anchor's low end — the point is genuine uncertainty, not a guaranteed pass). Then extrapolate opening_weekend, week1, and lifetime thresholds from that Day 1 anchor using typical Bollywood trajectory patterns (weekend is usually 3-4x Day 1 for a strong opener; lifetime typically continues to build over the following weeks for a film with real legs). If no advance booking anchor is provided, fall back to cast/genre/language judgment alone, as described above.
For EACH metric, also output an initial_yes_probability — your honest estimate of how likely the actual result is to meet or exceed the threshold you chose. This is NOT automatically 0.50 just because you tried to pick an "uncertain" threshold — a well-chosen threshold can still be, say, 65% likely or 40% likely; the point is calibration, not forced neutrality. If an ADVANCE BOOKING ANCHOR is available, use where your threshold sits relative to the anchor's estimated range to inform this: a threshold near the low end of the range implies a HIGHER probability of hitting it; a threshold near the high end implies a LOWER probability. Without an anchor, reason from cast/genre/language as before, but still commit to a genuine number, not a placeholder.

Do NOT write the question sentence yourself — only provide the threshold, a one-sentence reasoning, probability, and a short display title.


"""

def generate_questions_for_movie(
    movie: Dict,
    metric_registry: List[Dict],
    llm_client: OpenAI,
) -> Optional[GeneratedQuestionSet]:
    """Calls the LLM once to calibrate thresholds for all confirmed
    registry metrics for one movie. Returns None on failure."""

    confirmed_metrics = [
        {"metric_id": m["metric_id"], "question_template": m["question_template"]}
        for m in metric_registry
        if m.get("source_url_template")
    ]

    advance = fetch_advance_booking_estimate(movie["movie_name"])
    if advance:
        tickets_k = advance["tickets"] / 1000
        low_est = round(tickets_k * ADVANCE_TICKET_RATIO_LOW, 2)
        high_est = round(tickets_k * ADVANCE_TICKET_RATIO_HIGH, 2)
        anchor_line = (
            f"ADVANCE BOOKING ANCHOR: {advance['tickets']:,} tickets sold before release "
            f"(source: TrackMyShow). Estimated Day 1 India Net range: ₹{low_est} Cr - ₹{high_est} Cr."
        )
        logger.info(f"Advance booking anchor found for '{movie['movie_name']}': {advance['tickets']:,} tickets")
    else:
        anchor_line = "ADVANCE BOOKING ANCHOR: Not available for this movie."
        logger.info(f"No advance booking anchor for '{movie['movie_name']}' — falling back to cast/genre reasoning.")

    user_prompt = (
        f"Movie: {movie['movie_name']}\n"
        f"Language: {movie['language']}\n"
        f"Genre: {', '.join(movie.get('genre', [])) or 'Unknown'}\n"
        f"Primary Cast: {', '.join(movie.get('primary_cast', [])) or 'Unknown'}\n"
        f"{anchor_line}\n\n"
        f"Generate a threshold for EACH of these metrics:\n{confirmed_metrics}"
    )

    try:
        response = llm_client.beta.chat.completions.parse(
            model="gpt-5.6-terra",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=GeneratedQuestionSet,
        )
        result = response.choices[0].message.parsed
        logger.info(f"Generated {len(result.selections)} question(s) for '{movie['movie_name']}'")
        return result
    except Exception as e:
        logger.error(f"Question generation failed for '{movie['movie_name']}': {e}")
        return None

def _format_threshold(value: float) -> str:
    """100.0 -> '100', 87.5 -> '87.5' — avoids ugly '.0' in question text."""
    return str(int(value)) if value == int(value) else str(value)


def _betting_lock_utc(release_date) -> str:
    """1 day before release, 23:59 IST, converted to UTC."""
    lock_day = release_date - timedelta(days=1)
    lock_ist = datetime.combine(lock_day, datetime.min.time()) + timedelta(hours=23, minutes=59)
    lock_utc = lock_ist - IST_OFFSET
    return lock_utc.replace(tzinfo=timezone.utc).isoformat()


def build_final_questions(
    movie: Dict,
    generated: GeneratedQuestionSet,
    metric_registry: List[Dict],
) -> List[Dict]:
    """Fills in the registry's template text and computes timestamps —
    pure Python, no LLM involved in this step at all."""

    registry_by_id = {m["metric_id"]: m for m in metric_registry}
    release_date = datetime.strptime(movie["release_date"], "%Y-%m-%d").date()
    betting_lock = _betting_lock_utc(release_date)

    final_questions = []
    for selection in generated.selections:
        entry = registry_by_id.get(selection.metric_id)
        if not entry:
            logger.warning(f"LLM returned unknown metric_id '{selection.metric_id}' — skipping.")
            continue

        question_text = entry["question_template"].format(
            movie=movie["movie_name"],
            threshold=_format_threshold(selection.threshold),
        )
        resolution_date = release_date + timedelta(days=entry["resolution_days_after_release"])

        final_questions.append({
            "movie_name": movie["movie_name"],
            "metric_id": selection.metric_id,
            "question_text": question_text,
            "display_title": selection.display_title,
            "threshold": selection.threshold,
            "initial_yes_probability": selection.initial_yes_probability,  # new
            "reasoning": selection.reasoning,
            "betting_lock_utc": betting_lock,
            "resolution_deadline_utc": resolution_date.isoformat(),
            "resolution_source_url": entry["source_url_template"].format(slug=movie["movie_slug"]),
            "resolution_question_text": entry["resolution_question_text"].format(movie=movie["movie_name"]),
        })

    return final_questions


if __name__ == "__main__":

    import os
    logging.basicConfig(level=logging.INFO)

    from metric_registry import METRIC_REGISTRY  
    test_movie = {
        "movie_name": "Awarapan 2", "movie_slug": "awarapan-2",
        "language": "Indian", "release_date": "2026-08-14",
        "genre": ["Action", "Romance"], "primary_cast": ["Emraan Hashmi", "Disha Patani"],
    }

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    generated = generate_questions_for_movie(test_movie, METRIC_REGISTRY, client)
    if generated:
        for q in build_final_questions(test_movie, generated, METRIC_REGISTRY):
            print(q)