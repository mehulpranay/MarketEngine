import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from .ingester import RawSignal

logger = logging.getLogger("FanGram.Extractor")


class ExtractedEvent(BaseModel):
    """Pydantic schema enforcing structured LLM extraction for Section 2."""

    has_anchor_event: bool = Field(
        description="True if headline refers to a specific, concrete proper-noun event or entity."
    )
    anchor_event: Optional[str] = Field(
        default=None,
        description="Normalized proper-noun Anchor Event (e.g., 'Stree 2 Theatrical Release')."
    )
    candidate_endpoints: List[str] = Field(
        default_factory=list,
        description="1 to 3 objective, measurable YES/NO endpoints for this anchor event."
    )
    has_future_uncertainty: bool = Field(
        description="True if the event has unresolved, unpredictable future binary outcomes."
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Explanation if dropped (e.g., 'No concrete event', 'Event completely in past')."
    )


class EventExtractorEngine:
    """Extracts Anchor Events from raw signals and verifies objective future binary endpoints."""

    SYSTEM_PROMPT = """You are an anchor event extractor for an Indian prediction market engine.
Your job is to parse news headlines and extract structured anchor events for market creation.

CRITICAL RULES FOR ENTERTAINMENT HEADLINES:
1. UPCOMING MOVIE FILTERING:
   - The headline MUST anchor around a specific, concrete UPCOMING MOVIE (e.g., 'Awarapan 2 Theatrical Release').
   - Articles discussing past movies, genre retrospectives, or music trends (e.g., 'From Awarapan to Gangster...', 'Why 2007 films flopped') -> REJECT with reason: "Past movie retrospective or general music/genre editorial".
   - Multi-movie listicles or roundups (e.g., '10 films that released on...') -> REJECT with reason: "Listicle or compilation".
   - General industry news or non-movie events (e.g., film body raids, star-studded award shows) -> REJECT with reason: "No specific movie anchor".

2. NON-ENTERTAINMENT HEADLINES (Sports, Space, Tech, Business):
- Require a concrete, proper-noun event with unresolved future binary outcomes.


3. DEDUCE UPCOMING/FUTURE UNCERTAINTY:
   - The headline must refer to an active upcoming event or unresolved future outcome (e.g., theatrical release date, advance booking performance, box office clash).

4. POLYMARKET-STYLE CANDIDATE ENDPOINTS:
   - All candidate_endpoints MUST be formatted as strict, objective, binary YES/NO prediction questions in Polymarket style.
   - Example 1: "Will 'Awarapan 2' gross over ₹15 Cr India Net on Day 1?"
   - Example 2: "Will 'Awarapan 2' outgross 'Batwara 1947' in opening day collections?"
   - Do NOT use generic phrases or topic names like 'Awarapan 2 box office performance'.

    ======================================================================
    ENTERTAINMENT & MOVIE CANDIDATE QUESTION RULES
    ======================================================================
    When the anchor event is an upcoming movie release or movie clash, generate 
    candidate binary questions using standard box office metrics:

    1. Opening Day Target:
    - "Will '[Movie Title]' gross over ₹[X] Cr India Net on Opening Day/Day 1?"

    2. Opening Weekend Target:
    - "Will '[Movie Title]' gross over ₹[Y] Cr India Net in its Opening Weekend?"


    4. Head-to-Head Clash (if headline mentions a rival movie):
    - "Will '[Movie A]' outgross '[Movie B]' in Day 1 India Net collections?"

    DO NOT generate listicle questions, celebrity gossip polls, or past retrospectives.

5. OUTPUT REQUIREMENTS:
- If valid entertainment headline, `anchor_event` MUST be anchored to the specific MOVIE (e.g., 'Awarapan 2 Theatrical Release').
- If rejected, set `has_anchor_event = False` and provide a clear `rejection_reason`.
"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract(self, signal: RawSignal) -> ExtractedEvent:
        """Processes a RawSignal and returns a structured ExtractedEvent contract."""
        user_prompt = f"Headline: '{signal.title}'\nSource: '{signal.source_name}'\nPublished UTC: {signal.published_utc}"

        if not self.llm_client:
            # Fallback mock for testing without active LLM API credentials
            logger.info(f"Extracting event for signal: '{signal.title[:40]}...'")
            return self._mock_extraction(signal)

        try:
            # Production path using OpenAI / Structured Outputs API
            response = self.llm_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=ExtractedEvent,
                temperature=0.1,
            )
            extracted: ExtractedEvent = response.choices[0].message.parsed
            
            if not extracted.has_anchor_event or not extracted.has_future_uncertainty:
                logger.warning(
                    f"REJECTED EVENT: '{signal.title[:40]}...' -> {extracted.rejection_reason}"
                )
            else:
                logger.info(
                    f"EXTRACTED: '{extracted.anchor_event}' with {len(extracted.candidate_endpoints)} endpoints."
                )

            return extracted

        except Exception as e:
            logger.error(f"Structured API extraction call failed for '{signal.title[:40]}...': {e}")
            return ExtractedEvent(
                has_anchor_event=False,
                has_future_uncertainty=False,
                rejection_reason=f"LLM Extraction exception: {str(e)}"
            )

    def _mock_extraction(self, signal: RawSignal) -> ExtractedEvent:
        """Deterministic fallback mock for testing pipeline execution."""
        title_lower = signal.title.lower()

        if "stree 2" in title_lower:
            return ExtractedEvent(
                has_anchor_event=True,
                anchor_event="Stree 2 Theatrical Release",
                candidate_endpoints=[
                    "Stree 2 opens to ₹40 Cr or more India Net on Day 1?",
                    "Stree 2 crosses ₹100 Cr India Net in opening weekend?"
                ],
                has_future_uncertainty=True
            )

        return ExtractedEvent(
            has_anchor_event=False,
            has_future_uncertainty=False,
            rejection_reason="Mock filter: No recognized anchor event in test title."
        )