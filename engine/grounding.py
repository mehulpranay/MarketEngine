import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from tavily import TavilyClient
from .extractor import ExtractedEvent

logger = logging.getLogger("FanGram.Grounding")


class GroundedContext(BaseModel):
    """Pydantic schema enforcing structured output for verified real-world schedule."""
    release_date_found: bool = Field(
        description="True if an explicit event/release date was found in the search snippets."
    )
    exact_date_str: Optional[str] = Field(
        default=None,
        description="The exact date/time found in text (e.g., 'August 21, 2026'). Null if not found."
    )
    temporal_reasoning: Optional[str] = Field(
        default=None,
        description="Step-by-step reasoning comparing Current UTC Time with extracted date: 'Today is X. Event is Y. Therefore, event is [past/future].'"
    )
    is_future_event: bool = Field(
        description="True ONLY if extracted date is strictly in the future compared to Today's Date. False if in the past or missing."
    )
    grounding_snippet: Optional[str] = Field(
        default=None,
        description="Copy-paste of 10-15 exact words from search snippets proving the schedule."
    )


class GroundingEngine:
    """Agentic Grounding Engine using native OpenAI tool-calling and Tavily Search API."""

    SEARCH_TOOL_SPEC = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs targeted web search to find official release dates, event schedules, match start times, or launch windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Hyper-specific search query focused on schedule or release data (e.g., 'Mirzapur The Movie official release date India sacnilk')."
                    }
                },
                "required": ["query"]
            }
        }
    }

    SYSTEM_PROMPT = """You are an autonomous Fact-Checking and Temporal Verification Agent for an Indian prediction platform.

    YOUR OBJECTIVE:
    Given an Anchor Event and Headline, verify if the event is real, currently active/upcoming, and scheduled for a concrete future date relative to Current UTC Time.

    AVAILABLE TOOLS:
    - `web_search(query: string)`: Searches the web for official event schedules, release dates, or match start times.

    SEARCH QUERY GUIDELINES:
    1. DYNAMIC QUERY FORMULATION: Match the search query precisely to the entity type:
    - For Movies: '[Exact Movie Name] release date India' or '[Exact Movie Name] movie box office sacnilk'
    - For Sports: '[Team A vs Team B] match schedule start time espncricinfo'
    - For Space/Tech: '[Mission Name] launch schedule pib isro'
    2. DO NOT POLLUTE QUERIES: Keep search terms minimal and accurate. Do not force unnecessary years or extra keywords if they pollute search context.
    3. PREVENT SEQUEL/PREQUEL CONFLATION: Ensure search queries match the exact movie name (e.g., distinguish 'Awarapan' from 'Awarapan 2').

    STRICT OPERATING RULES:
    1. TARGETED SEARCHES: Formulate short, precise queries focused strictly on dates and schedules. Limit to 1 or 2 search turns maximum.
    2. TEMPORAL GROUNDING & STEP-BY-STEP REASONING:
    - You MUST ONLY rely on dates found in search snippets returned by `web_search`. Do NOT use pre-trained memory for dates.
    - In your temporal reasoning, explicitly state:
        a) Current UTC Time provided in prompt.
        b) Exact event/release date extracted from search snippets.
        c) Step-by-step comparison proving whether extracted date is strictly in the FUTURE.
    """

    def __init__(self, tavily_api_key: str, client: Optional[OpenAI] = None):
        self.client = client
        self.tavily = TavilyClient(api_key=tavily_api_key)

    def _execute_search_tool(self, query: str, max_results: int = 3) -> str:
        """Executes Tavily search tool call made by the LLM."""
        logger.info(f"Agent Executing Tool `web_search`: '{query}'")
        try:
            response = self.tavily.search(query=query, max_results=max_results, search_depth="basic")
            results = response.get("results", [])
            if not results:
                logger.warning(f"Tool `web_search` returned empty results for: '{query}'")
                return "No search results found."

            compiled_snippets = "\n".join([f"- {r.get('content', '')}" for r in results])
            return compiled_snippets
        except Exception as e:
            logger.error(f"Tavily Tool execution error: {e}")
            return f"Search execution failed: {str(e)}"

    def verify_event(self, event: ExtractedEvent) -> Optional[GroundedContext]:
        """Runs agentic tool-calling loop and outputs structured GroundedContext."""
        logger.info(f"Starting Agentic Grounding for: '{event.anchor_event}'")

        if not self.client:
            logger.warning("No OpenAI client provided. Returning mock grounding response.")
            return GroundedContext(
                release_date_found=True,
                exact_date_str="August 21, 2026",
                temporal_reasoning="Mock reasoning: Event date is in the future.",
                is_future_event=True,
                grounding_snippet="scheduled to release in theaters on August 21, 2026"
            )

        current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current UTC Time: {current_utc}\n"
                    f"Anchor Event: {event.anchor_event}\n"
                    f"Candidate Endpoints: {event.candidate_endpoints}\n\n"
                    "Use the `web_search` tool to verify the official schedule/release date."
                )
            }
        ]

        # Agent Tool Loop (Max 2 tool turns to enforce latency/cost boundary)
        max_turns = 2
        turns_taken = 0

        while turns_taken < max_turns:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=[self.SEARCH_TOOL_SPEC],
                    tool_choice="auto"
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # If model chooses to call tool
                if tool_calls:
                    messages.append(response_message)  # Add assistant's tool-call decision to history

                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        if function_name == "web_search":
                            function_args = json.loads(tool_call.function.arguments)
                            search_query = function_args.get("query", "")

                            # Run Tavily tool
                            tool_result = self._execute_search_tool(search_query)

                            # Append tool result back to message history
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": tool_result
                            })

                    turns_taken += 1
                else:
                    # Model decided it has enough info without making further tool calls
                    break

            except Exception as e:
                logger.error(f"Error during agentic tool loop: {e}")
                return None

        # Final Pass: Force structured Pydantic output using conversation context
        try:
            logger.info("Finalizing Agent Grounding into structured Pydantic schema...")
            final_messages = messages + [
                {
                    "role": "user",
                    "content": "Evaluate all retrieved search results and output the final GroundedContext JSON."
                }
            ]

            final_response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=final_messages,
                response_format=GroundedContext
            )

            result: GroundedContext = final_response.choices[0].message.parsed

            if not result.release_date_found:
                logger.warning(f"GROUNDING REJECTED: No release date found for '{event.anchor_event}'.")
                return result
            if not result.is_future_event:
                logger.warning(f"GROUNDING REJECTED: Event '{event.anchor_event}' is in the past ({result.exact_date_str}).")
                return result

            logger.info(f"GROUNDING SUCCESS: '{event.anchor_event}' confirmed for {result.exact_date_str}.")
            return result

        except Exception as e:
            logger.error(f"Failed to finalize structured GroundedContext output: {e}")
            return None











import logging
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field
from tavily import TavilyClient
from openai import OpenAI
from .extractor import ExtractedEvent

logger = logging.getLogger("FanGram.Grounding")

class GroundedContext(BaseModel):
    """Pydantic schema for the LLM to confirm real-world schedule from search results."""
    release_date_found: bool = Field(
        description="True if an explicit release date was found in the search text."
    )
    exact_date_str: Optional[str] = Field(
        default=None,
        description="The exact date found in the text (e.g., 'August 15, 2026'). Null if not found."
    )
    temporal_reasoning: Optional[str] = Field(
        default=None,
        description="Write a one-sentence step-by-step comparison: 'Today's date is X. The release date is Y. Therefore, the release is in the [past/future].'"
    )
    is_future_event: bool = Field(
        description="True ONLY if the extracted date is strictly in the future compared to Today's Date."
    )
    grounding_snippet: Optional[str] = Field(
        default=None,
        description="Copy-paste the exact 10-15 words from the text that proves the date."
    )

class GroundingEngine:
    """Verifies anchor events against live web search results using Tavily Search API."""

    SYSTEM_PROMPT = """You are a fact-checking and grounding assistant. 
Your job is to read search engine snippets and verify the official release/schedule date of an event.

STRICT RULES:
1. You must ONLY use the provided Search Snippets. Do not use outside knowledge.
2. If the snippets do not contain a concrete date, set release_date_found to False.
3. Compare the found date against 'Today's Date' provided in the prompt. If the event date has already passed, is_future_event MUST be False.
"""

    def __init__(self, tavily_api_key: str, client: Optional[OpenAI] = None):
        self.client = client
        self.tavily = TavilyClient(api_key=tavily_api_key)

    def _fetch_search_snippets(self, query: str, max_results: int = 3) -> str:
        """Fetches live search snippets reliably via Tavily API."""
        logger.info(f"Executing Tavily Web Search: '{query}'")
        try:
            response = self.tavily.search(query=query, max_results=max_results, search_depth="basic")
            results = response.get("results", [])
            
            if not results:
                logger.warning(f"No search results returned for query: '{query}'")
                return ""
            
            # Extract content snippets cleanly
            compiled_text = "\n".join([f"- {r.get('content', '')}" for r in results])
            return compiled_text
        except Exception as e:
            logger.error(f"Tavily Search API error: {e}")
            return ""

    def verify_event(self, event: ExtractedEvent) -> Optional[GroundedContext]:
        """Runs the Search and Grounding evaluation."""
        
        # Clean up the anchor event name for better search precision
        clean_name = event.anchor_event.replace("Theatrical Release", "").strip()
        # FIXED: Added 'India' to force local release date fetching
        search_query = f"{clean_name} movie release date India 2026"
        

        # 1. Execute Web Search via Tavily
        search_text = self._fetch_search_snippets(search_query)
        
        # Defense 1: Empty Search Check
        if not search_text.strip():
            logger.warning(f"GROUNDING FAILED: Empty search payload for '{search_query}'")
            return None

        # 2. Inject Current UTC Date
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        user_prompt = (
            f"Anchor Event: {event.anchor_event}\n"
            f"Today's Date: {current_date}\n\n"
            f"Search Snippets:\n{search_text}\n\n"
            "Extract the exact release date and verify if it is in the future."
        )

        if not self.client:
            logger.warning("No OpenAI client provided. Running in mock grounding mode.")
            return GroundedContext(
                release_date_found=True, 
                exact_date_str="August 21, 2026", 
                is_future_event=True, 
                grounding_snippet="scheduled to release on August 21, 2026"
            )

        # 3. LLM Grounding Evaluation
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=GroundedContext
            )
            
            result: GroundedContext = response.choices[0].message.parsed
            
            if not result.release_date_found:
                logger.warning(f"GROUNDING REJECTED: No release date found in search text for '{event.anchor_event}'.")
                return None
            if not result.is_future_event:
                logger.warning(f"GROUNDING REJECTED: Event '{event.anchor_event}' is in the past ({result.exact_date_str}).")
                return None
                
            logger.info(f"GROUNDING SUCCESS: '{event.anchor_event}' confirmed for {result.exact_date_str}.")
            return result

        except Exception as e:
            logger.error(f"LLM Grounding parse error: {e}")
            return None