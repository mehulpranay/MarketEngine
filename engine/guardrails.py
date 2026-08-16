import re
import unicodedata
import logging
from typing import Tuple, Optional
from .ingester import RawSignal

logger = logging.getLogger("FanGram.Guardrails")

class Tier1SafetyGuardrail:
    """Two-Tier Pre-LLM Guardrail.

    Tier 1: High-speed Regex regex blacklist with word boundary controls.
    Tier 2: Disambiguation evaluation for pop-culture titles flagged by regex.
    """

    # Exact word boundaries \b prevent partial string matches
    BLACK_PATTERNS = [
        r"\b(election|elections|bjp|congress|aap|modi|rahul|minister|court|jail|arrest|police|cbi|ed|fir|judge)\b",
        r"\b(died|dead|death|accident|crash|tragedy|killed|murder|suicide|rape|assault|victim|injury)\b",
        r"\b(temple|mosque|church|religion|caste|communal|riot|protest|slur|fatwa)\b",
        r"\b(divorce|scandal|cheating|allegation|defamation|leak|leaked)\b"
    ]

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalizes unicode characters, case, and dot/dash obfuscations."""
        text = unicodedata.normalize('NFKD', text)
        text = text.lower()
        # Remove dots/dashes used to obfuscate blacklisted terms (e.g., "M.o.d.i")
        text = re.sub(r'[\.\-\_\*]', '', text)
        return text

    @classmethod
    def _tier2_llm_disambiguate(cls, title: str, matched_pattern: str) -> bool:
        """Tier 2 fallback evaluation for flagged headlines.

        Checks if a flagged phrase is part of a benign pop-culture or movie title.
        """
        # Prompt pattern for lightweight LLM call (e.g., gpt-4o-mini / claude-3-haiku)
        system_prompt = (
            "You are a strict safety classifier for a movie prediction platform. "
            "Determine if the headline refers to an actual real-world tragedy, crime, "
            "or political conflict, OR if it is simply a fictional movie/song title or movie news."
        )
        user_prompt = f"Headline: '{title}'\nFlagged Keyword: '{matched_pattern}'\nIs this safe entertainment news? Reply YES or NO."
        
        # Stub returning False for safety during test mode. Replace with client.chat.completions call.
        logger.info(f"Tier 2 Disambiguation triggered for: '{title}' (Pattern: {matched_pattern})")
        return False

    @classmethod
    def evaluate(cls, signal: RawSignal, use_tier2: bool = False) -> Tuple[bool, str]:
        """Evaluates a RawSignal against safety rules.

        Returns (is_safe, rejection_reason).
        """
        clean_title = cls.normalize_text(signal.title)

        for pattern in cls.BLACK_PATTERNS:
            match = re.search(pattern, clean_title)
            if match:
                matched_word = match.group(0)
                
                # If Tier 2 LLM disambiguation is enabled, pass to secondary checker
                if use_tier2:
                    is_safe_entertainment = cls._tier2_llm_disambiguate(signal.title, matched_word)
                    if is_safe_entertainment:
                        logger.info(f"ACCEPTED (Tier 2 Passed): '{signal.title[:50]}...'")
                        return True, "Passed Tier 2 Disambiguation"

                reason = f"Flagged by Guardrail (Matched: '{matched_word}')"
                logger.warning(f"REJECTED: '{signal.title[:50]}...' -> {reason}")
                return False, reason

        logger.info(f"ACCEPTED: '{signal.title[:50]}...'")
        return True, "Passed Guardrail"