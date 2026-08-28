import json
import logging
from typing import List, Optional

from pydantic import BaseModel, Field
from openai import OpenAI
from tavily import TavilyClient

from tier1_bookings import fetch_advance_booking_estimate  # Tier 1
logger = logging.getLogger("FanGram.AdvanceBookingSearch")


"""
[Movie Name & Details] ──► Try Tier-1 Parser (`fetch_advance_booking_estimate`)
                                     │
                             ┌───────┴───────┐
                     (Found) │               │ (Miss / None)
                             ▼               ▼
                      [Return Result]   [Escalate to Agentic Search]
                                             │
                                             ▼
                                    [LLM generates query]
                                             │
                                             ▼ (via Tavily Search Tool)
                                    [Fetch Live Web Results]
                                             │
                                             ▼ (Up to 3 turns)
                                    [LLM parses & reasons across sources]
                                             │
                                             ▼
                                    [Structured Pydantic Model: AdvanceBookingContext]
"""
"""
{
  "data_found": true,
  "summary": "Sacnilk reports 145,000 tickets sold across national chains as of Day 2 of the 5-day advance booking window. Strong traction observed in Hindi multi-plexes with an average ticket price (ATP) of ₹240.",
  "estimated_day1_india_net_cr": 18.5,
  "confidence": "high",
  "sources_used": [
    "https://www.sacnilk.com/news/toxic_2026_box_office_collection",
    "https://www.bollywoodhungama.com/news/box-office-special-features/toxic-advance-booking-tracker"
  ]
}
"""


SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Performs targeted web search to find advance booking figures, ticket sales, or trade box-office estimates.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Hyper-specific search query focused on advance booking or box office prediction data."
                }
            },
            "required": ["query"]
        }
    }
}

class AdvanceBookingContext(BaseModel):
    """Grounded summary of advance booking activity for one movie,
    built from live search results — not from one fixed site's parser."""

    data_found: bool = Field(description="True if any real advance booking figures were found.")
    summary: str = Field(description="Written summary in your own words: ticket counts, gross figures, language breakdown if multi-language, how far into the advance window this data point is (e.g. 'Day 1 of a 5-day window'), and any explicit comparisons to other films.")
    estimated_day1_india_net_cr: Optional[float] = Field(
        default=None,
        description="Your own best-effort Day 1 India Net estimate in ₹ Cr, reasoned from whatever ticket/gross/ATP data was found — not a fixed formula, use judgment based on what the sources actually report."
    )
    confidence: str = Field(description="'low', 'medium', or 'high' — based on how directly the sources answer this, and how recent/complete the data is.")
    sources_used: List[str] = Field(default_factory=list, description="URLs actually used.")


ADVANCE_BOOKING_SYSTEM_PROMPT = """You are an agent researching advance booking activity for an upcoming Indian film release, for a prediction market platform.

YOUR OBJECTIVE:
Find real, current advance ticket booking data for the given movie — ticket counts, gross figures, or explicit trade estimates — using the web_search tool. Do NOT rely on prior knowledge; only use what your searches actually return.

SEARCH GUIDANCE:
- Try queries like '{movie} advance booking tickets sold', '{movie} First Day Advance Booking Report ', '{movie} box office prediction for Day 1'.
- Multiple outlets (Sacnilk, TrackMyShow, BookMyShow trackers, trade news sites) may report different numbers at different points in the advance window — note this rather than picking one arbitrarily.
- If the movie has multiple language versions, capture the breakdown if reported, not just a blended total.
- Note explicitly how far into the advance-booking window the data is (e.g. 'Day 1 of a 5-day window') — a same-day early snapshot means much less than a full pre-release total.

RECONCILING CONFLICTING SOURCES:
- When multiple sources report different figures, do not just pick the highest, most recent, or most dramatic number. Reason across all of them.
- Sacnilk is the most reliable primary source for Indian box office/advance data — when sources conflict, weight Sacnilk-reported figures more heavily than other outlets.
- Advance booking totals are CUMULATIVE and only grow while the booking window is open. If you find multiple figures from around the same time period, check their timestamps: a larger, LATER figure is not "inconsistent" with an earlier, smaller one — it is very likely simply the same running total measured later. Prefer the most recent complete figure as your baseline, rather than an earlier partial one, even if the numbers differ substantially.
- When you attribute a claim to a specific outlet in your summary, use the EXACT outlet name as it appears in the source snippet you retrieved. Never guess, assume, or infer which outlet said something.

STRICT RULES:
1. Only use information found via web_search. Never estimate purely from memory of the film, cast, or genre.
2. If no real advance booking data exists yet (too early, or a small film with no coverage), set data_found=False rather than guessing.
3. Limit yourself to 3 search turns maximum.
"""


def _execute_search_tool(tavily_client: TavilyClient, query: str, max_results: int = 3) -> str:
    logger.info(f"Executing search: '{query}'")
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
        )
        results = response.get("results", [])
        if not results:
            return "No search results found."

        entries = []
        for r in results:
            text = r.get("raw_content") or r.get("content", "")
            entries.append(f"- [{r.get('url', 'unknown source')}] {text}")
        return "\n".join(entries)
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Search execution failed: {str(e)}"

def fetch_advance_booking_via_search(
    movie_name: str,
    release_date: str,
    llm_client: OpenAI,
    tavily_client: TavilyClient,
    known_cast: Optional[List[str]] = None,
) -> Optional[AdvanceBookingContext]:
    """Agentic search for advance booking data — the resilient fallback
    when the TrackMyShow parser finds no confident match."""

    cast_line = f"Known cast: {', '.join(known_cast)}" if known_cast else "Known cast: Unknown"

    user_prompt = (
        f"Movie: {movie_name}\n"
        f"Release Date: {release_date}\n"
        f"{cast_line}\n\n"
        f"Find current advance booking data using web_search."
    )

    messages = [
        {"role": "system", "content": ADVANCE_BOOKING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    max_turns = 3
    for i in range(max_turns):
        logger.info(f"Advance booking search turn {i+1}/{max_turns} for '{movie_name}'")

        try:
            response = llm_client.chat.completions.create(
                model="gpt-5.6-terra",
                messages=messages,
                tools=[SEARCH_TOOL_SPEC],  # reuse from grounding.py
                tool_choice="auto",
                reasoning_effort="none",
            )
        except Exception as e:
            logger.error(f"Advance booking search call failed for '{movie_name}': {e}")
            return None

        msg = response.choices[0].message
        if not msg.tool_calls:
            break

        messages.append(msg)
        for call in msg.tool_calls:
            if call.function.name == "web_search":
                args = json.loads(call.function.arguments)
                query = args.get("query", "")
                result_text = _execute_search_tool(tavily_client, query)  
                messages.append({
                    "tool_call_id": call.id, 
                    "role": "tool",
                    "name": "web_search", 
                    "content": result_text,
                })

    try:
        final = llm_client.beta.chat.completions.parse(
            model="gpt-5.6-terra",
            messages=messages + [{"role": "user", 
                                  "content": "Summarize your findings as AdvanceBookingContext."}],
            response_format=AdvanceBookingContext,
        )
        return final.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Failed to finalize advance booking context for '{movie_name}': {e}")
        return None


def get_advance_booking_signal(movie_name, release_date, llm_client, tavily_client, known_cast=None):
    result = fetch_advance_booking_estimate(movie_name)
    if result:
        return result

    logger.info(f"TrackMyShow miss for '{movie_name}' — escalating to search.")
    return fetch_advance_booking_via_search(movie_name, release_date, llm_client, tavily_client, known_cast)

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    result = fetch_advance_booking_via_search("Toxic", "2026-08-26", llm_client, tavily_client)
    print(result)

