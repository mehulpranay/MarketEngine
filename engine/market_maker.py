import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from tavily import TavilyClient

from .extractor import ExtractedEvent
from .grounding import GroundedContext

logger = logging.getLogger("FanGram.MarketMaker")


# =====================================================================
# PYDANTIC SCHEMAS FOR PASS 1 & PASS 2
# =====================================================================

class Pass1Intent(BaseModel):
    """Pydantic schema for Pass 1 output: Intent extraction & single movie-agnostic query formulation."""
    candidate_question: str = Field(
        description="The candidate binary question from Section 2."
    )
    primary_domain: str = Field(
        default="bollywoodhungama.com",
        description="Approved source domain from registry (e.g., 'bollywoodhungama.com', 'boxofficeindia.com', 'espncricinfo.com', 'youtube.com')."
    )
    metric_type: str = Field(
        description="The exact metric to verify on target domain (e.g., 'Opening Day India Net Box Office', 'Opening Weekend', 'Trailer View Count')."
    )
    metric_verification_query: str = Field(
        description="MOVIE-AGNOSTIC search query to test if domain publishes this metric format (e.g., 'site:bollywoodhungama.com \"Opening Day\" \"India Net\" box office'). MUST NOT include unreleased movie title."
    )


class ResolutionRules(BaseModel):
    """Pydantic schema defining exact, dispute-free resolution conditions."""
    primary_source: str = Field(
        description="Official primary oracle URL / Domain (e.g., 'BollywoodHungama.com (India Net Box Office Report)')."
    )
    fallback_source: str = Field(
        description="Secondary backup oracle if primary source is temporarily down."
    )
    resolution_criteria: str = Field(
        description="Strict mathematical criteria required for YES payout."
    )
    void_clause: str = Field(
        description="Exact terms for market VOID refund (e.g., release postponed > 30 days)."
    )


class MarketContract(BaseModel):
    """Production-grade Polymarket prediction market contract schema for FanGram."""
    market_id: str = Field(
        default_factory=lambda: f"mkt_{uuid.uuid4().hex[:10]}",
        description="Unique system market contract identifier."
    )
    anchor_event: str = Field(description="Normalized proper-noun anchor event name.")
    market_title: str = Field(description="High-converting UI market title.")
    binary_question: str = Field(description="Finalized objective binary YES/NO question.")
    category: str = Field(default="Bollywood", description="Market category.")
    
    # Timestamps & Clocks
    verified_release_date: str = Field(description="Verified release date string from Section 3 Grounding.")
    betting_lock_utc: str = Field(description="ISO 8601 UTC timestamp when betting closes (Day -1 at 23:59 IST / 18:29 UTC).")
    resolution_deadline_utc: str = Field(description="ISO 8601 UTC timestamp when market resolves (Day +1 or +2 at 18:29 UTC).")
    
    # Rules & Calibration
    resolution_rules: ResolutionRules = Field(description="Dispute-free resolution terms.")
    initial_yes_probability: float = Field(
        default=0.50, description="Calibrated starting probability between 0.05 and 0.95."
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp."
    )


class Pass2ResolvabilityResponse(BaseModel):
    """Pydantic schema for Pass 2 Gatekeeper decision and contract output."""
    gate_2_metric_supported: bool = Field(
        description="True ONLY if search snippets prove that target domain standardly publishes this metric format."
    )
    is_resolvable: bool = Field(
        description="True ONLY if Gate 2 is True."
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Detailed explanation if rejected (e.g., 'Gate 2 Failed: bollywoodhungama.com does not track popcorn sales')."
    )
    market_contract: Optional[MarketContract] = Field(
        default=None,
        description="Full MarketContract JSON if is_resolvable is True, otherwise null."
    )


# =====================================================================
# ENGINE IMPLEMENTATION
# =====================================================================

class MarketMakerEngine:
    """Section 4: Streamlined Market Creation Engine with Single Movie-Agnostic Search and Gatekeeper Evaluation."""

    PASS1_SYSTEM_PROMPT = """You are a Prediction Market Intent Architect for FanGram.
Given a candidate binary question and anchor event, analyze the question and select the primary domain registry and format query:

RULES FOR `metric_verification_query`:
1. MUST BE MOVIE-AGNOSTIC / ITEM-AGNOSTIC. Do NOT include the unreleased movie name or specific subject name in the query.
2. Search strictly for the metric format on the target domain to prove the domain standardly publishes it.
   - Good Example: '"Opening Day" "India Net" box office collection report'
   - Bad Example:  'site:bollywoodhungama.com Opening Day India Net'  <-- Do not add 'site:'

PRIMARY DOMAINS REGISTRY:
- Indian Box Office: bollywoodhungama.com or boxofficeindia.com
- Cricket / Sports: espncricinfo.com
- YouTube / Media: youtube.com
- Tech / Space: isro.gov.in
"""

    PASS2_SYSTEM_PROMPT = """You are the Chief Resolvability Gatekeeper for FanGram Prediction Markets.
Your job is to evaluate search snippets returned from a Movie-Agnostic Metric Verification Search and decide if the market is resolvable.

GATEKEEPER RULE:
- Gate 2 (Metric Supported on Domain?): Do the search snippets contain explicit proof that the target domain standardly publishes tabular or report formats for the requested `metric_type`?

TIMESTAMP CALCULATION RULES:
- `betting_lock_utc`: Day -1 before `verified_release_date` at 23:59 IST (which equals 18:29 UTC). Format as ISO 8601 string.
- `resolution_deadline_utc`: Day +1 or +2 after `verified_release_date` at 18:29 UTC. Format as ISO 8601 string.

DECISION MATRIX:
- If search snippets DO NOT prove the domain tracks this metric format: set `is_resolvable = False`, state `rejection_reason`, and set `market_contract = null`.
- If search snippets DO prove the domain tracks this metric format: set `is_resolvable = True`, generate `market_contract` with exact timestamps and resolution rules.
"""

    def __init__(self, tavily_api_key: str, llm_client: Optional[OpenAI] = None):
        self.client = llm_client
        self.tavily = TavilyClient(api_key=tavily_api_key)

    def _execute_search(self, query: str, primary_domain: Optional[str] = None) -> str:
        """Executes a single Tavily web search using native domain isolation."""
        domains = [primary_domain] if primary_domain else None
        logger.info(f"Executing Movie-Agnostic Metric Search: '{query}' (Domain Target: {domains})")
        
        try:
            response = self.tavily.search(
                query=query,
                include_domains=domains,  # Native API domain restriction
                max_results=3,
                search_depth="basic"
            )
            results = response.get("results", [])
            if not results:
                return "No search results returned."
            return "\n".join([f"- {r.get('content', '')}" for r in results])
        except Exception as e:
            logger.error(f"Search API execution error for '{query}': {e}")
            return f"Search execution failed: {str(e)}"
        

    def run_pass1(self, candidate_question: str, anchor_event: str) -> Optional[Pass1Intent]:
        """Pass 1: Formulates movie-agnostic metric query and selects primary domain."""
        if not self.client:
            logger.warning("No OpenAI client provided. Returning Mock Pass 1 Intent.")
            return Pass1Intent(
                candidate_question=candidate_question,
                primary_domain="bollywoodhungama.com",
                metric_type="Opening Day India Net Box Office Collection",
                metric_verification_query='site:bollywoodhungama.com "Opening Day" "India Net" box office collection'
            )

        user_prompt = (
            f"Anchor Event: '{anchor_event}'\n"
            f"Candidate Question: '{candidate_question}'\n\n"
            "Analyze candidate question and generate the Pass1Intent JSON with a MOVIE-AGNOSTIC metric_verification_query."
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.PASS1_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=Pass1Intent,
                temperature=0.1,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Pass 1 LLM extraction failed: {e}")
            return None

    def run_pass2(
        self, 
        candidate_question: str, 
        anchor_event: str, 
        grounded_date: str,
        pass1_intent: Pass1Intent, 
        metric_snippets: str
    ) -> Optional[Pass2ResolvabilityResponse]:
        """Pass 2: Evaluates Gate 2 against search snippets and constructs full MarketContract if valid."""
        if not self.client:
            logger.warning("No OpenAI client provided. Running Mock Pass 2 evaluation.")
            return self._mock_pass2_response(anchor_event, candidate_question, grounded_date)

        current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        user_prompt = (
            f"Current UTC Time: {current_utc}\n"
            f"Anchor Event: '{anchor_event}'\n"
            f"Candidate Question: '{candidate_question}'\n"
            f"Grounded Verified Release Date: '{grounded_date}'\n\n"
            f"--- TARGET METRIC TO VERIFY ---\n"
            f"Primary Domain: '{pass1_intent.primary_domain}'\n"
            f"Target Metric Type: '{pass1_intent.metric_type}'\n"
            f"Metric Search Query Executed: '{pass1_intent.metric_verification_query}'\n\n"
            f"--- SEARCH SNIPPETS FROM PRIMARY DOMAIN ---\n"
            f"{metric_snippets}\n\n"
            f"Evaluate if the search snippets prove '{pass1_intent.primary_domain}' standardly tracks and reports '{pass1_intent.metric_type}'. "
            "Output the final Pass2ResolvabilityResponse JSON."
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.PASS2_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=Pass2ResolvabilityResponse,
                temperature=0.1,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Pass 2 LLM evaluation failed: {e}")
            return None

    def create_market(self, event: ExtractedEvent, grounding: GroundedContext) -> Optional[MarketContract]:
        """Full Pipeline Wrapper: Runs Pass 1 -> Movie-Agnostic Tavily Search -> Pass 2 Gatekeeper Check."""
        if not event.candidate_endpoints:
            logger.warning(f"No candidate endpoints available in ExtractedEvent for '{event.anchor_event}'.")
            return None

        candidate_question = event.candidate_endpoints[0]
        grounded_date = grounding.exact_date_str
        logger.info(f"Section 4 Processing Candidate Question: '{candidate_question}' (Release Date: {grounded_date})")

        # Step 1: Pass 1 Intent & Movie-Agnostic Query Generation
        pass1_intent = self.run_pass1(candidate_question, event.anchor_event)
        if not pass1_intent:
            logger.error("Pass 1 execution failed. Aborting market creation.")
            return None

        # Step 2: Execute Single Tavily Search for Metric Trackability using native domain filtering
        metric_snippets = self._execute_search(
            query=pass1_intent.metric_verification_query,
            primary_domain=pass1_intent.primary_domain
        )

        # Step 3: Pass 2 Gatekeeper Decision & Contract Construction
        pass2_response = self.run_pass2(
            candidate_question, 
            event.anchor_event, 
            grounded_date,
            pass1_intent, 
            metric_snippets
        )

        if not pass2_response:
            logger.error("Pass 2 execution returned empty response.")
            return None

        if not pass2_response.is_resolvable or not pass2_response.market_contract:
            logger.warning(
                f"❌ MARKET CREATION REJECTED (Section 4 Gatekeeper): "
                f"[Gate2={pass2_response.gate_2_metric_supported}] "
                f"Reason: {pass2_response.rejection_reason}"
            )
            return None

        contract = pass2_response.market_contract
        logger.info(
            f"🔥 MARKET CONTRACT CREATED SUCCESS: '{contract.market_title}' ({contract.market_id}) "
            f"Lock: {contract.betting_lock_utc} | Resolve: {contract.resolution_deadline_utc}"
        )
        return contract

    def _mock_pass2_response(
        self, anchor_event: str, candidate_question: str, grounded_date: str
    ) -> Pass2ResolvabilityResponse:
        """Deterministic mock for unit testing."""
        contract = MarketContract(
            anchor_event=anchor_event,
            market_title=f"{anchor_event} Day 1 Box Office",
            binary_question=candidate_question,
            category="Bollywood",
            verified_release_date=grounded_date,
            betting_lock_utc="2026-08-13T18:29:00Z",
            resolution_deadline_utc="2026-08-15T18:29:00Z",
            resolution_rules=ResolutionRules(
                primary_source="BollywoodHungama.com (India Net Box Office Report)",
                fallback_source="BoxOfficeIndia.com Trade Archives",
                resolution_criteria="Resolves to YES if BollywoodHungama.com official report states Day 1 India Net collections >= ₹15.00 Cr. Otherwise resolves to NO.",
                void_clause="If theatrical release is postponed beyond 30 days from release date, market resolves VOID."
            )
        )
        return Pass2ResolvabilityResponse(
            gate_2_metric_supported=True,
            is_resolvable=True,
            market_contract=contract
        )

# import uuid
# import logging
# from datetime import datetime, timezone, timedelta
# from typing import Optional, List, Tuple
# from concurrent.futures import ThreadPoolExecutor
# from pydantic import BaseModel, Field
# from openai import OpenAI
# from tavily import TavilyClient

# from .extractor import ExtractedEvent
# from .grounding import GroundedContext

# logger = logging.getLogger("FanGram.MarketMaker")


# # =====================================================================
# # PYDANTIC SCHEMAS FOR PASS 1 & PASS 2
# # =====================================================================

# class Pass1Intent(BaseModel):
#     """Pydantic schema for Pass 1 output: Intent extraction & parallel query formulation."""
#     candidate_question: str = Field(
#         description="The candidate binary question from Section 2."
#     )
#     primary_domain: str = Field(
#         default="bollywoodhungama.com",
#         description="Approved source domain from registry (e.g., 'bollywoodhungama.com', 'boxofficeindia.com', 'espncricinfo.com', 'youtube.com')."
#     )
#     metric_type: str = Field(
#         description="The exact metric to verify on the target domain (e.g., 'Day 1 India Net Box Office', 'Opening Weekend', 'Trailer View Count')."
#     )
#     schedule_query: str = Field(
#         description="Hyper-focused search query to find the official release date or event schedule (e.g., 'Awarapan 2 official release date India')."
#     )
#     metric_verification_query: str = Field(
#         description="Site-constrained search query to test if domain publishes this metric format (e.g., 'site:bollywoodhungama.com Awarapan 2 Day 1 India Net box office')."
#     )


# class ResolutionRules(BaseModel):
#     """Pydantic schema defining exact, dispute-free resolution conditions."""
#     primary_source: str = Field(
#         description="Official primary oracle URL / Domain (e.g., 'BollywoodHungama.com (India Net Box Office Report)')."
#     )
#     fallback_source: str = Field(
#         description="Secondary backup oracle if primary source is temporarily down."
#     )
#     resolution_criteria: str = Field(
#         description="Strict mathematical criteria required for YES payout."
#     )
#     void_clause: str = Field(
#         description="Exact terms for market VOID refund (e.g., release postponed > 30 days)."
#     )


# class MarketContract(BaseModel):
#     """Production-grade Polymarket prediction market contract schema for FanGram."""
#     market_id: str = Field(
#         default_factory=lambda: f"mkt_{uuid.uuid4().hex[:10]}",
#         description="Unique system market contract identifier."
#     )
#     anchor_event: str = Field(description="Normalized proper-noun anchor event name.")
#     market_title: str = Field(description="High-converting UI market title.")
#     binary_question: str = Field(description="Finalized objective binary YES/NO question.")
#     category: str = Field(default="Bollywood", description="Market category.")
    
#     # Timestamps & Clocks
#     verified_release_date: str = Field(description="Verified release date string (e.g., '14 August 2026').")
#     betting_lock_utc: str = Field(description="ISO 8601 UTC timestamp when betting closes (Day -1 at 23:59 IST).")
#     resolution_deadline_utc: str = Field(description="ISO 8601 UTC timestamp when market resolves (Day +1 or +2).")
    
#     # Rules & Calibration
#     resolution_rules: ResolutionRules = Field(description="Dispute-free resolution terms.")
#     initial_yes_probability: float = Field(
#         default=0.50, description="Calibrated starting probability between 0.05 and 0.95."
#     )
#     created_at_utc: str = Field(
#         default_factory=lambda: datetime.now(timezone.utc).isoformat(),
#         description="Creation timestamp."
#     )


# class Pass2ResolvabilityResponse(BaseModel):
#     """Pydantic schema for Pass 2 Dual-Gate decision and contract output."""
#     gate_1_release_date_found: bool = Field(
#         description="True ONLY if search snippets contain an explicit official event/release date."
#     )
#     gate_2_metric_supported: bool = Field(
#         description="True ONLY if search snippets prove that target domain standardly publishes this metric format."
#     )
#     is_resolvable: bool = Field(
#         description="True ONLY if BOTH Gate 1 AND Gate 2 are True."
#     )
#     rejection_reason: Optional[str] = Field(
#         default=None,
#         description="Detailed explanation if rejected (e.g., 'Gate 2 Failed: bollywoodhungama.com does not track popcorn sales')."
#     )
#     market_contract: Optional[MarketContract] = Field(
#         default=None,
#         description="Full MarketContract JSON if is_resolvable is True, otherwise null."
#     )


# # =====================================================================
# # ENGINE IMPLEMENTATION
# # =====================================================================

# class MarketMakerEngine:
#     """Section 4: Two-Pass Market Creation Engine with Parallel Tool Execution and Dual-Gate Resolvability."""

#     PASS1_SYSTEM_PROMPT = """You are a Prediction Market Intent Architect for FanGram.
# Given a candidate binary question and anchor event, analyze the question and generate TWO distinct search queries:

# 1. `schedule_query`: Formulate a query to extract the official theatrical release or match start date.
# 2. `metric_verification_query`: Formulate a site-constrained search query targeting approved trade archives (e.g., 'site:bollywoodhungama.com', 'site:boxofficeindia.com', 'site:espncricinfo.com') to empirically test if the domain standardly publishes this metric format.

# PRIMARY DOMAINS REGISTRY:
# - Indian Box Office: bollywoodhungama.com or boxofficeindia.com
# - Cricket / Sports: espncricinfo.com
# - YouTube / Media: youtube.com
# - Tech / Space: isro.gov.in
# """

#     PASS2_SYSTEM_PROMPT = """You are the Chief Resolvability Gatekeeper for FanGram Prediction Markets.
# Your job is to evaluate search snippets returned from TWO parallel searches (Schedule Search & Metric Trackability Search) and apply a strict Dual-Gate Evaluation Matrix:

# DUAL-GATE EVALUATION MATRIX:
# - Gate 1 (Release Date Found?): Do search snippets explicitly state the official future release date or start schedule?
# - Gate 2 (Metric Supported on Domain?): Do search snippets contain proof that the target domain standardly publishes tabular or explicit reports for this exact metric format (e.g., Day 1 India Net, Opening Weekend, Trailer Views)?

# STRICT RULES:
# 1. Both Gate 1 AND Gate 2 MUST be True for `is_resolvable` to be True.
# 2. If EITHER gate fails, set `is_resolvable = False`, provide a clear `rejection_reason`, and set `market_contract = null`.
# 3. If BOTH gates pass, compute timestamps:
#    - `betting_lock_utc`: 1 Day before release date at 23:59 IST (converted to ISO 8601 UTC).
#    - `resolution_deadline_utc`: 1 or 2 Days after release date to allow trade reports to publish.
# 4. Finalize and construct the complete `market_contract`.
# """

#     def __init__(self, tavily_api_key: str, llm_client: Optional[OpenAI] = None):
#         self.client = llm_client
#         self.tavily = TavilyClient(api_key=tavily_api_key)

#     def _execute_single_search(self, query: str) -> str:
#         """Executes a single Tavily web search turn."""
#         logger.info(f"Executing Search Query: '{query}'")
#         try:
#             response = self.tavily.search(query=query, max_results=3, search_depth="basic")
#             results = response.get("results", [])
#             if not results:
#                 return "No search results returned."
#             return "\n".join([f"- {r.get('content', '')}" for r in results])
#         except Exception as e:
#             logger.error(f"Search API execution error for '{query}': {e}")
#             return f"Search execution failed: {str(e)}"

#     def execute_parallel_searches(self, schedule_query: str, metric_query: str) -> Tuple[str, str]:
#         """Executes Schedule Query and Metric Verification Query concurrently using ThreadPoolExecutor."""
#         logger.info("⚡ Launching Parallel Tavily Searches for Temporal Date & Metric Trackability...")
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             future_schedule = executor.submit(self._execute_single_search, schedule_query)
#             future_metric = executor.submit(self._execute_single_search, metric_query)
            
#             schedule_snippets = future_schedule.result()
#             metric_snippets = future_metric.result()
            
#         logger.info("✅ Parallel Search Execution Complete.")
#         return schedule_snippets, metric_snippets

#     def run_pass1(self, candidate_question: str, anchor_event: str) -> Optional[Pass1Intent]:
#         """Pass 1: Emits dual search queries (Schedule + Metric Verification) in a single LLM output."""
#         if not self.client:
#             logger.warning("No OpenAI client provided. Returning Mock Pass 1 Intent.")
#             return Pass1Intent(
#                 candidate_question=candidate_question,
#                 primary_domain="bollywoodhungama.com",
#                 metric_type="Day 1 India Net Box Office",
#                 schedule_query=f"{anchor_event} official release date India",
#                 metric_verification_query=f"site:bollywoodhungama.com {anchor_event} Day 1 India Net box office"
#             )

#         user_prompt = (
#             f"Anchor Event: '{anchor_event}'\n"
#             f"Candidate Question: '{candidate_question}'\n\n"
#             "Analyze candidate question and generate the Pass1Intent JSON with schedule_query and metric_verification_query."
#         )

#         try:
#             response = self.client.beta.chat.completions.parse(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": self.PASS1_SYSTEM_PROMPT},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 response_format=Pass1Intent,
#                 temperature=0.1,
#             )
#             return response.choices[0].message.parsed
#         except Exception as e:
#             logger.error(f"Pass 1 LLM extraction failed: {e}")
#             return None

#     def run_pass2(
#         self, 
#         candidate_question: str, 
#         anchor_event: str, 
#         pass1_intent: Pass1Intent, 
#         schedule_snippets: str, 
#         metric_snippets: str
#     ) -> Optional[Pass2ResolvabilityResponse]:
#         """Pass 2: Evaluates search snippets against Gate 1 & Gate 2, outputting final MarketContract if valid."""
#         if not self.client:
#             logger.warning("No OpenAI client provided. Running Mock Pass 2 evaluation.")
#             return self._mock_pass2_response(anchor_event, candidate_question, pass1_intent)

#         current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

#         user_prompt = (
#             f"Current UTC Time: {current_utc}\n"
#             f"Anchor Event: '{anchor_event}'\n"
#             f"Candidate Question: '{candidate_question}'\n"
#             f"Target Primary Domain: '{pass1_intent.primary_domain}'\n\n"
#             f"--- SEARCH RESULT 1 (Schedule Query: '{pass1_intent.schedule_query}') ---\n"
#             f"{schedule_snippets}\n\n"
#             f"--- SEARCH RESULT 2 (Metric Trackability Query: '{pass1_intent.metric_verification_query}') ---\n"
#             f"{metric_snippets}\n\n"
#             "Apply the Dual-Gate Evaluation Matrix and output the final Pass2ResolvabilityResponse JSON."
#         )

#         try:
#             response = self.client.beta.chat.completions.parse(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": self.PASS2_SYSTEM_PROMPT},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 response_format=Pass2ResolvabilityResponse,
#                 temperature=0.1,
#             )
#             return response.choices[0].message.parsed
#         except Exception as e:
#             logger.error(f"Pass 2 LLM evaluation failed: {e}")
#             return None

#     def create_market(self, event: ExtractedEvent, grounding: GroundedContext) -> Optional[MarketContract]:
#         """Full Pipeline Wrapper: Runs Pass 1 -> Parallel Tavily Search -> Pass 2 Dual-Gate Check."""
#         # Use first candidate endpoint generated by ExtractorEngine
#         if not event.candidate_endpoints:
#             logger.warning(f"No candidate endpoints available in ExtractedEvent for '{event.anchor_event}'.")
#             return None

#         candidate_question = event.candidate_endpoints[0]
#         logger.info(f"Section 4 Processing Candidate Question: '{candidate_question}'")

#         # Step 1: Pass 1 Dual Query Generation
#         pass1_intent = self.run_pass1(candidate_question, event.anchor_event)
#         if not pass1_intent:
#             logger.error("Pass 1 execution failed. Aborting market creation.")
#             return None

#         # Step 2: Parallel Tavily Search Execution
#         schedule_snippets, metric_snippets = self.execute_parallel_searches(
#             pass1_intent.schedule_query, 
#             pass1_intent.metric_verification_query
#         )

#         # Step 3: Pass 2 Dual-Gate Resolvability Decision
#         pass2_response = self.run_pass2(
#             candidate_question, 
#             event.anchor_event, 
#             pass1_intent, 
#             schedule_snippets, 
#             metric_snippets
#         )

#         if not pass2_response:
#             logger.error("Pass 2 execution returned empty response.")
#             return None

#         if not pass2_response.is_resolvable or not pass2_response.market_contract:
#             logger.warning(
#                 f"❌ MARKET CREATION REJECTED (Section 4 Gatekeeper): "
#                 f"[Gate1={pass2_response.gate_1_release_date_found}, Gate2={pass2_response.gate_2_metric_supported}] "
#                 f"Reason: {pass2_response.rejection_reason}"
#             )
#             return None

#         contract = pass2_response.market_contract
#         logger.info(
#             f"🔥 MARKET CONTRACT CREATED SUCCESS: '{contract.market_title}' ({contract.market_id}) "
#             f"Lock: {contract.betting_lock_utc} | Resolve: {contract.resolution_deadline_utc}"
#         )
#         return contract

#     def _mock_pass2_response(
#         self, anchor_event: str, candidate_question: str, pass1_intent: Pass1Intent
#     ) -> Pass2ResolvabilityResponse:
#         """Deterministic mock for unit testing."""
#         contract = MarketContract(
#             anchor_event=anchor_event,
#             market_title=f"{anchor_event} Day 1 Box Office",
#             binary_question=candidate_question,
#             category="Bollywood",
#             verified_release_date="14 August 2026",
#             betting_lock_utc="2026-08-13T18:29:00Z",
#             resolution_deadline_utc="2026-08-15T12:00:00Z",
#             resolution_rules=ResolutionRules(
#                 primary_source="BollywoodHungama.com (India Net Box Office Report)",
#                 fallback_source="BoxOfficeIndia.com Trade Archives",
#                 resolution_criteria="Resolves to YES if BollywoodHungama.com official report states Day 1 India Net collections >= ₹15.00 Cr. Otherwise resolves to NO.",
#                 void_clause="If theatrical release is postponed beyond 30 days from 14 August 2026, market resolves VOID."
#             )
#         )
#         return Pass2ResolvabilityResponse(
#             gate_1_release_date_found=True,
#             gate_2_metric_supported=True,
#             is_resolvable=True,
#             market_contract=contract
#         )


    
# # import logging
# # from typing import List, Optional
# # from pydantic import BaseModel, Field
# # from openai import OpenAI
# # from tavily import TavilyClient
# # from .extractor import ExtractedEvent

# # logger = logging.getLogger("FanGram.MarketMaker")

# # # ---------------------------------------------------------------------------
# # # DATA CONTRACTS
# # # ---------------------------------------------------------------------------

# # class Pass1EndpointIntent(BaseModel):
# #     """Pass 1: Generates targeted search query and identifies domain registry source."""
# #     candidate_endpoint: str = Field(
# #         description="The candidate endpoint question being processed."
# #     )
# #     data_intent_query: str = Field(
# #         description="Targeted search query to find the exact schedule/event time for this specific endpoint."
# #     )
# #     primary_domain: str = Field(
# #         description="Approved domain from registry (e.g., 'sacnilk.com', 'youtube.com', 'espncricinfo.com')."
# #     )

# # class PolymarketContract(BaseModel):
# #     """Pass 2: Complete, production-ready market contract schema."""
# #     is_resolvable: bool = Field(
# #         description="True ONLY if exact Betting Lock and Resolution dates were discovered from search text."
# #     )
# #     unresolvable_reason: Optional[str] = Field(
# #         default=None,
# #         description="Reason for dropping the question if is_resolvable is False."
# #     )
# #     market_title: Optional[str] = Field(
# #         default=None, 
# #         description="Short, direct title in Polymarket style (e.g., 'Stree 2: Day 1 Box Office ₹50 Cr+')."
# #     )
# #     final_question: Optional[str] = Field(
# #         default=None, 
# #         description="Unambiguous YES/NO binary statement."
# #     )
# #     category: str = Field(default="Cinema / Box Office")
    
# #     # Timestamps extracted by LLM from search context
# #     lock_timestamp_display: Optional[str] = Field(
# #         default=None, 
# #         description="User-facing display string (e.g., 'August 14, 2026 at 11:59 PM IST')."
# #     )
# #     lock_timestamp_utc: Optional[str] = Field(
# #         default=None, 
# #         description="ISO 8601 UTC timestamp when betting closes (e.g., '2026-08-14T18:29:59Z')."
# #     )
# #     resolution_timestamp_utc: Optional[str] = Field(
# #         default=None, 
# #         description="ISO 8601 UTC timestamp when market resolves (e.g., '2026-08-17T12:00:00Z')."
# #     )
    
# #     # Resolution Rules & Sources
# #     source_of_truth_url: Optional[str] = Field(
# #         default=None, 
# #         description="Authoritative domain URL used to verify outcome."
# #     )
# #     resolution_rules: List[str] = Field(
# #         default_factory=list, 
# #         description="Bullet-point legal resolution criteria."
# #     )
# #     void_clause: Optional[str] = Field(
# #         default=None, 
# #         description="Explicit cancellation rule if event is delayed, cancelled, or abandoned."
# #     )
# #     initial_probability: float = Field(
# #         default=0.50, 
# #         description="Calibrated initial probability bounded between 0.05 and 0.95."
# #     )


# # # ---------------------------------------------------------------------------
# # # TWO-PASS MARKET MAKER ENGINE
# # # ---------------------------------------------------------------------------

# # class MarketMakerEngine:
# #     """Two-Pass Market Creation Engine with Resolvability Gatekeeping."""

# #     APPROVED_REGISTRY = {
# #         "sacnilk.com": "Box Office Collections (India Net)",
# #         "bollywoodhungama.com": "Bollywood Releases & Box Office",
# #         "youtube.com": "Trailer Views & Video Metrics",
# #         "espncricinfo.com": "Cricket Match Schedules & Results",
# #         "imdb.com": "Ratings & Movie Details"
# #     }

# #     PASS1_SYSTEM_PROMPT = """You are a prediction market search query architect.
# # Your job is to take a candidate prediction endpoint and generate a hyper-specific 'data_intent_query' 
# # to find the exact date, start time, or release schedule required to set market deadlines.

# # Rules:
# # 1. Focus the search query on finding official dates, start times, or schedule windows.
# # 2. Select the single best primary domain from approved sources: sacnilk.com, bollywoodhungama.com, youtube.com, espncricinfo.com, imdb.com.
# # """

# #     PASS2_SYSTEM_PROMPT = """You are a legal contract generator for a prediction platform following strict Polymarket guidelines.

# # Your job is to read search snippets and evaluate if the endpoint is RESOLVABLE with hard timestamps.

# # STRICT RESOLVABILITY RULES:
# # 1. If the search snippets DO NOT contain enough schedule information to determine when the event starts or ends, set is_resolvable = False and provide unresolvable_reason.
# # 2. If schedule info is found, calculate BOTH:
# #    - Lock Timestamp (Betting Lock): Must be set BEFORE the event starts (e.g., midnight before release or match start time).
# #    - Resolution Timestamp: Set AFTER official data is published (e.g., 24-48 hours post-release).
# # 3. Provide ISO 8601 UTC timestamps (e.g. '2026-08-14T18:29:59Z').
# # 4. Define binding resolution rules referencing the primary domain.
# # 5. Define an explicit Void Clause for cancellations/postponements.
# # """

# #     def __init__(self, tavily_api_key: str, client: Optional[OpenAI] = None):
# #         self.client = client
# #         self.tavily = TavilyClient(api_key=tavily_api_key)

# #     def _execute_registry_search(self, query: str, domain: str, max_results: int = 2) -> str:
# #         """Executes a search query scoped to approved domains."""
# #         full_query = f"site:{domain} {query}" if domain else query
# #         logger.info(f"Executing Registry Search: '{full_query}'")
# #         try:
# #             response = self.tavily.search(query=full_query, max_results=max_results, search_depth="basic")
# #             results = response.get("results", [])
# #             if not results:
# #                 # Fallback search without site restriction if restricted query returns empty
# #                 response = self.tavily.search(query=query, max_results=max_results, search_depth="basic")
# #                 results = response.get("results", [])

# #             return "\n".join([f"- {r.get('content', '')}" for r in results])
# #         except Exception as e:
# #             logger.error(f"Tavily search error for '{query}': {e}")
# #             return ""

# #     def process_endpoint(self, anchor_event: str, endpoint: str) -> Optional[PolymarketContract]:
# #         """Executes Pass 1 -> Web Search -> Pass 2 Resolvability Check for a single candidate endpoint."""
# #         logger.info(f"Processing Endpoint: '{endpoint}'")

# #         if not self.client:
# #             logger.warning("No OpenAI client provided. Returning mock contract.")
# #             return self._mock_contract(endpoint)

# #         # -------------------------------------------------------------------
# #         # PASS 1: DRAFT INTENT & QUERY GENERATION
# #         # -------------------------------------------------------------------
# #         try:
# #             p1_prompt = f"Anchor Event: {anchor_event}\nCandidate Endpoint: {endpoint}"
# #             p1_response = self.client.beta.chat.completions.parse(
# #                 model="gpt-4o-mini",
# #                 messages=[
# #                     {"role": "system", "content": self.PASS1_SYSTEM_PROMPT},
# #                     {"role": "user", "content": p1_prompt}
# #                 ],
# #                 response_format=Pass1EndpointIntent
# #             )
# #             intent: Pass1EndpointIntent = p1_response.choices[0].message.parsed
# #             logger.info(f"Pass 1 Query Generated: '{intent.data_intent_query}' (Domain: {intent.primary_domain})")

# #         except Exception as e:
# #             logger.error(f"Pass 1 Execution Failed for endpoint '{endpoint}': {e}")
# #             return None

# #         # -------------------------------------------------------------------
# #         # TOOL EXECUTION: REGISTRY WEB SEARCH
# #         # -------------------------------------------------------------------
# #         search_snippets = self._execute_registry_search(
# #             query=intent.data_intent_query, 
# #             domain=intent.primary_domain
# #         )

# #         if not search_snippets.strip():
# #             logger.warning(f"RESOLVABILITY FAILED: No search snippets returned for '{intent.data_intent_query}'")
# #             return PolymarketContract(
# #                 is_resolvable=False,
# #                 unresolvable_reason=f"No search results found on registry domain '{intent.primary_domain}'."
# #             )

# #         # -------------------------------------------------------------------
# #         # PASS 2: RESOLVABILITY GATE & CONTRACT FINALIZATION
# #         # -------------------------------------------------------------------
# #         try:
# #             p2_prompt = (
# #                 f"Anchor Event: {anchor_event}\n"
# #                 f"Candidate Endpoint: {endpoint}\n"
# #                 f"Primary Domain: {intent.primary_domain}\n\n"
# #                 f"Search Snippets:\n{search_snippets}\n\n"
# #                 "Determine if hard timestamps can be extracted and finalize the Polymarket contract."
# #             )

# #             p2_response = self.client.beta.chat.completions.parse(
# #                 model="gpt-4o-mini",
# #                 messages=[
# #                     {"role": "system", "content": self.PASS2_SYSTEM_PROMPT},
# #                     {"role": "user", "content": p2_prompt}
# #                 ],
# #                 response_format=PolymarketContract
# #             )
# #             contract: PolymarketContract = p2_response.choices[0].message.parsed
            
# #             if not contract.is_resolvable:
# #                 logger.warning(f"DROPPED QUESTION (Unresolvable): '{endpoint}' -> {contract.unresolvable_reason}")
# #                 return contract

# #             contract.source_of_truth_url = f"https://www.{intent.primary_domain}"
# #             logger.info(f"PASS 2 SUCCESS: Contract Created -> '{contract.market_title}'")
# #             return contract

# #         except Exception as e:
# #             logger.error(f"Pass 2 Execution Failed: {e}")
# #             return None

# #     def _mock_contract(self, endpoint: str) -> PolymarketContract:
# #         """Fallback mock contract for offline testing."""
# #         return PolymarketContract(
# #             is_resolvable=True,
# #             market_title="Mock Movie: Day 1 Collection",
# #             final_question=endpoint,
# #             category="Cinema / Box Office",
# #             lock_timestamp_display="August 20, 2026 at 11:59 PM IST",
# #             lock_timestamp_utc="2026-08-20T18:29:59Z",
# #             resolution_timestamp_utc="2026-08-23T12:00:00Z",
# #             source_of_truth_url="https://www.sacnilk.com",
# #             resolution_rules=[
# #                 "Market resolves to YES if official Sacnilk India Net Day 1 collection meets or exceeds threshold.",
# #                 "Market resolves to NO if official collection is below threshold."
# #             ],
# #             void_clause="If release is postponed past August 31, 2026, market resolves to VOID."
# #         )