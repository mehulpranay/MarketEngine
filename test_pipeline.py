import os
import json
import logging
from openai import OpenAI

from engine.ingester import IngestionEngine
from engine.guardrails import Tier1SafetyGuardrail
from engine.extractor import EventExtractorEngine
from engine.grounding import GroundingEngine
from engine.market_maker import MarketMakerEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FanGram.PipelineTest")


def run_full_fangram_pipeline():
    print("=" * 75)
    print("      FANGRAM END-TO-END PREDICTION MARKET PIPELINE (SECTIONS 1 - 4)      ")
    print("=" * 75 + "\n")

    # Load API Keys securely from environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not openai_key or not tavily_key:
        logger.error(
            "Missing environment variables! Please set OPENAI_API_KEY and TAVILY_API_KEY."
        )
        return


    openai_client = OpenAI(api_key=openai_key)

    # Initialize Engine Components
    ingester = IngestionEngine()
    extractor = EventExtractorEngine(llm_client=openai_client)
    grounder = GroundingEngine(tavily_api_key=tavily_key, client=openai_client)
    market_maker = MarketMakerEngine(tavily_api_key=tavily_key, llm_client=openai_client)

    # -------------------------------------------------------------------
    # SECTION 1: INGESTION & SAFETY GUARDRAILS
    # -------------------------------------------------------------------
    print("--- SECTION 1: RSS INGESTION & TIER-1 GUARDRAILS ---")
    raw_signals = ingester.fetch_rss_feed()
    print(f"Total Raw RSS Signals Ingested: {len(raw_signals)}")

    if not raw_signals:
        print("No RSS signals fetched. Exiting pipeline.")
        return

    safe_signals = []
    for sig in raw_signals:
        is_safe, reason = Tier1SafetyGuardrail.evaluate(sig)
        if is_safe:
            safe_signals.append(sig)

    print(f"Passed Tier-1 Guardrails: {len(safe_signals)} / {len(raw_signals)}\n")

    # Limit batch to first 5 safe signals for the test run
    test_batch = safe_signals[:10]
    finalized_contracts = []

    # -------------------------------------------------------------------
    # SECTIONS 2, 3 & 4: EXTRACTION, GROUNDING & MARKET MAKER
    # -------------------------------------------------------------------
    print("--- SECTIONS 2, 3 & 4: PROCESSING HEADLINES TO MARKET CONTRACTS ---")

    for idx, signal in enumerate(test_batch, start=1):
        print("\n" + "=" * 75)
        print(f"[{idx}/{len(test_batch)}] HEADLINE: {signal.title}")
        print("=" * 75)

        # ---------------------------------------------------------------
        # SECTION 2: ANCHOR EXTRACTION & BINARY CANDIDATES
        # ---------------------------------------------------------------
        extracted_event = extractor.extract(signal)

        if not extracted_event.has_anchor_event or not extracted_event.has_future_uncertainty:
            print(f"  └─► ❌ REJECTED (Section 2 Extractor): {extracted_event.rejection_reason}")
            continue

        print(f"  └─► ✅ ANCHOR EVENT: '{extracted_event.anchor_event}'")
        print(f"  └─► CANDIDATE ENDPOINTS: {extracted_event.candidate_endpoints}")

        # ---------------------------------------------------------------
        # SECTION 3: AGENTIC TEMPORAL GROUNDING
        # ---------------------------------------------------------------
        print("\n  [RUNNING SECTION 3: GROUNDING ENGINE...]")
        grounded_context = grounder.verify_event(extracted_event)

        if not grounded_context or not grounded_context.release_date_found or not grounded_context.is_future_event:
            print(f"  └─► ❌ REJECTED (Section 3 Grounder): Release date missing or event in past.")
            continue

        print(f"  └─► ✅ GROUNDED RELEASE DATE: {grounded_context.exact_date_str}")
        print(f"  └─► REASONING: {grounded_context.temporal_reasoning}")

        # ---------------------------------------------------------------
        # SECTION 4: STREAMLINED MARKET MAKER & GATEKEEPER
        # ---------------------------------------------------------------
        print("\n  [RUNNING SECTION 4: MARKET MAKER & GATEKEEPER...]")
        contract = market_maker.create_market(extracted_event, grounded_context)

        if not contract:
            print(f"  └─► ❌ REJECTED (Section 4 Gatekeeper): Gate 2 Trackability check failed.")
            continue

        finalized_contracts.append(contract)

        print("\n  🔥 [SUCCESS - FANGRAM CONTRACT CREATED] 🔥")
        print(f"     Market ID:        {contract.market_id}")
        print(f"     Market Title:     {contract.market_title}")
        print(f"     Binary Question:  {contract.binary_question}")
        print(f"     Category:         {contract.category}")
        print(f"     Verified Date:    {contract.verified_release_date}")
        print(f"     Betting Lock:     {contract.betting_lock_utc}")
        print(f"     Resolution Date:  {contract.resolution_deadline_utc}")
        print(f"     Primary Oracle:   {contract.resolution_rules.primary_source}")

    # -------------------------------------------------------------------
    # FINAL SUMMARY & JSON DUMP
    # -------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"SUMMARY: Generated {len(finalized_contracts)} Production Market Contracts")
    print("=" * 75 + "\n")

    if finalized_contracts:
        print("--- SAMPLE GENERATED MARKET CONTRACT (JSON) ---")
        sample_json = finalized_contracts[0].model_dump_json(indent=2)
        print(sample_json)


if __name__ == "__main__":
    run_full_fangram_pipeline()


# import os
# import json
# import logging
# from openai import OpenAI

# from engine.ingester import IngestionEngine
# from engine.guardrails import Tier1SafetyGuardrail
# from engine.extractor import EventExtractorEngine
# from engine.grounding import GroundingEngine
# from engine.market_maker import MarketMakerEngine

# # Configure Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )
# logger = logging.getLogger("FanGram.PipelineTest")


# def run_full_fangram_pipeline():
#     print("=" * 75)
#     print("      FANGRAM END-TO-END PREDICTION MARKET PIPELINE (SECTIONS 1 - 4)      ")
#     print("=" * 75 + "\n")

#     # Load API Keys securely from environment variables
#     openai_key = os.getenv("OPENAI_API_KEY")
#     tavily_key = os.getenv("TAVILY_API_KEY")

#     if not openai_key or not tavily_key:
#         logger.error(
#             "Missing environment variables! Please set OPENAI_API_KEY and TAVILY_API_KEY."
#         )
#         return

#     openai_client = OpenAI(api_key=openai_key)

#     # Initialize Engine Components
#     ingester = IngestionEngine()
#     extractor = EventExtractorEngine(llm_client=openai_client)
#     grounder = GroundingEngine(tavily_api_key=tavily_key, client=openai_client)
#     market_maker = MarketMakerEngine(tavily_api_key=tavily_key, llm_client=openai_client)

#     # -------------------------------------------------------------------
#     # SECTION 1: INGESTION & SAFETY GUARDRAILS
#     # -------------------------------------------------------------------
#     print("--- SECTION 1: RSS INGESTION & TIER-1 GUARDRAILS ---")
#     raw_signals = ingester.fetch_rss_feed()
#     print(f"Total Raw RSS Signals Ingested: {len(raw_signals)}")

#     if not raw_signals:
#         print("No RSS signals fetched. Exiting pipeline.")
#         return

#     safe_signals = []
#     for sig in raw_signals:
#         is_safe, reason = Tier1SafetyGuardrail.evaluate(sig)
#         if is_safe:
#             safe_signals.append(sig)

#     print(f"Passed Tier-1 Guardrails: {len(safe_signals)} / {len(raw_signals)}\n")

#     # Limit batch to first 5 safe signals for the test run
#     test_batch = safe_signals[12:12 + 5]  # Adjust indices as needed for testing
#     finalized_contracts = []

#     # -------------------------------------------------------------------
#     # SECTIONS 2, 3 & 4: EXTRACTION, GROUNDING & TWO-PASS MARKET MAKER
#     # -------------------------------------------------------------------
#     print("--- SECTIONS 2, 3 & 4: PROCESSING HEADLINES TO MARKET CONTRACTS ---")

#     for idx, signal in enumerate(test_batch, start=1):
#         print("\n" + "=" * 75)
#         print(f"[{idx}/{len(test_batch)}] HEADLINE: {signal.title}")
#         print("=" * 75)

#         # ---------------------------------------------------------------
#         # SECTION 2: ANCHOR EXTRACTION & BINARY CANDIDATES
#         # ---------------------------------------------------------------
#         extracted_event = extractor.extract(signal)

#         if not extracted_event.has_anchor_event or not extracted_event.has_future_uncertainty:
#             print(f"  └─► ❌ REJECTED (Section 2 Extractor): {extracted_event.rejection_reason}")
#             continue

#         print(f"  └─► ✅ ANCHOR EVENT: '{extracted_event.anchor_event}'")
#         print(f"  └─► CANDIDATE ENDPOINTS: {extracted_event.candidate_endpoints}")

#         # ---------------------------------------------------------------
#         # SECTION 3: AGENTIC TEMPORAL GROUNDING
#         # ---------------------------------------------------------------
#         print("\n  [RUNNING SECTION 3: GROUNDING ENGINE...]")
#         grounded_context = grounder.verify_event(extracted_event)

#         if not grounded_context or not grounded_context.release_date_found or not grounded_context.is_future_event:
#             print(f"  └─► ❌ REJECTED (Section 3 Grounder): Release date missing or event in past.")
#             continue

#         print(f"  └─► ✅ GROUNDED RELEASE DATE: {grounded_context.exact_date_str}")
#         print(f"  └─► REASONING: {grounded_context.temporal_reasoning}")

#         # ---------------------------------------------------------------
#         # SECTION 4: TWO-PASS MARKET MAKER & PARALLEL DUAL-GATE
#         # ---------------------------------------------------------------
#         print("\n  [RUNNING SECTION 4: MARKET MAKER & PARALLEL DUAL-GATE...]")
#         contract = market_maker.create_market(extracted_event, grounded_context)

#         if not contract:
#             print(f"  └─► ❌ REJECTED (Section 4 Gatekeeper): Dual-Gate check failed.")
#             continue

#         finalized_contracts.append(contract)

#         print("\n  🔥 [SUCCESS - POLYMARKET CONTRACT CREATED] 🔥")
#         print(f"     Market ID:        {contract.market_id}")
#         print(f"     Market Title:     {contract.market_title}")
#         print(f"     Binary Question:  {contract.binary_question}")
#         print(f"     Category:         {contract.category}")
#         print(f"     Verified Date:    {contract.verified_release_date}")
#         print(f"     Betting Lock:     {contract.betting_lock_utc}")
#         print(f"     Resolution Date:  {contract.resolution_deadline_utc}")
#         print(f"     Primary Oracle:   {contract.resolution_rules.primary_source}")

#     # -------------------------------------------------------------------
#     # FINAL SUMMARY & JSON DUMP
#     # -------------------------------------------------------------------
#     print("\n" + "=" * 75)
#     print(f"SUMMARY: Generated {len(finalized_contracts)} Production Market Contracts")
#     print("=" * 75 + "\n")

#     if finalized_contracts:
#         print("--- SAMPLE GENERATED MARKET CONTRACT (JSON) ---")
#         sample_json = finalized_contracts[0].model_dump_json(indent=2)
#         print(sample_json)


# if __name__ == "__main__":
#     run_full_fangram_pipeline()





# # import os
# # import logging
# # import json
# # from openai import OpenAI

# # from engine.ingester import IngestionEngine
# # from engine.guardrails import Tier1SafetyGuardrail
# # from engine.extractor import EventExtractorEngine
# # from engine.grounding import GroundingEngine

# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# # )

# # def test_grounding_pipeline():
# #     print("==========================================================")
# #     print("   FANGRAM AGENTIC GROUNDING PIPELINE TEST (SECTIONS 1-3)  ")
# #     print("==========================================================\n")

# #     # OPENAI_KEY = "YOUR_OPENAI_API_KEY_HERE"
# #     # TAVILY_KEY = "YOUR_TAVILY_API_KEY_HERE"


# #     openai_client = OpenAI(api_key=OPENAI_KEY)

# #     # Initialize Section 1 to 3 Modules
# #     ingester = IngestionEngine()
# #     extractor = EventExtractorEngine(llm_client=openai_client)
# #     grounder = GroundingEngine(tavily_api_key=TAVILY_KEY, client=openai_client)

# #     # -------------------------------------------------------------------
# #     # SECTION 1: RSS INGESTION & TIER-1 GUARDRAILS
# #     # -------------------------------------------------------------------
# #     print("--- SECTION 1: INGESTION & GUARDRAILS ---")
# #     raw_signals = ingester.fetch_rss_feed()
# #     print(f"Total Raw Signals Ingested: {len(raw_signals)}")

# #     if not raw_signals:
# #         print("No RSS signals found. Exiting.")
# #         return

# #     safe_signals = []
# #     for sig in raw_signals:
# #         is_safe, reason = Tier1SafetyGuardrail.evaluate(sig)
# #         if is_safe:
# #             safe_signals.append(sig)

# #     print(f"Passed Tier-1 Guardrails: {len(safe_signals)} / {len(raw_signals)}")

# #     # -------------------------------------------------------------------
# #     # SECTION 2 & 3: EXTRACTION & AGENTIC GROUNDING
# #     # -------------------------------------------------------------------
# #     print("\n--- SECTIONS 2 & 3: ANCHOR EXTRACTION & AGENTIC GROUNDING ---")
    
# #     valid_grounded_events = []

# #     for signal in safe_signals[6:10]:  # Test top 5 safe headlines
# #         print("\n" + "="*70)
# #         print(f"[HEADLINE]: {signal.title}")
# #         print("="*70)

# #         # Section 2: Extract Anchor Event
# #         extracted_event = extractor.extract(signal)
        
# #         # Checking schema boolean attribute (has_future_uncertainty or is_valid_future_event)
# #         is_event_valid = getattr(
# #             extracted_event, "has_future_uncertainty", 
# #             getattr(extracted_event, "is_valid_future_event", False)
# #         )

# #         if not is_event_valid:
# #             print(f"  └─► REJECTED (Section 2 Extractor): {extracted_event.rejection_reason}")
# #             continue

# #         print(f"  └─► EXTRACTED ANCHOR EVENT: '{extracted_event.anchor_event}'")
# #         print(f"  └─► CANDIDATE ENDPOINTS:     {extracted_event.candidate_endpoints}")

# #         # Section 3: Agentic Tool-Calling Grounding
# #         print("\n  [RUNNING AGENTIC GROUNDING ENGINE...]")
# #         grounded_context = grounder.verify_event(extracted_event)

# #         if not grounded_context:
# #             print("  └─► REJECTED (Section 3 Grounder): Execution error or empty context.")
# #             continue

# #         if not grounded_context.release_date_found:
# #             print(f"  └─► REJECTED (Section 3 Grounder): No explicit release date found.")
# #             continue

# #         if not grounded_context.is_future_event:
# #             print(f"  └─► REJECTED (Section 3 Grounder): Event is in the PAST ({grounded_context.exact_date_str}).")
# #             print(f"      Reasoning: {grounded_context.temporal_reasoning}")
# #             continue

# #         # If passed all checks
# #         valid_grounded_events.append((extracted_event, grounded_context))
        
# #         print("\n  ✅ [GROUNDING PASSED - VALID FUTURE EVENT] ✅")
# #         print(f"     Anchor Event:     {extracted_event.anchor_event}")
# #         print(f"     Verified Date:    {grounded_context.exact_date_str}")
# #         print(f"     Is Future Event:  {grounded_context.is_future_event}")
# #         print(f"     CoT Reasoning:   {grounded_context.temporal_reasoning}")
# #         print(f"     Snippet Proof:    \"{grounded_context.grounding_snippet}\"")

# #     print("\n==========================================================")
# #     print(f"SUMMARY: Successfully Verified & Grounded {len(valid_grounded_events)} Events")
# #     print("==========================================================")

# # if __name__ == "__main__":
# #     test_grounding_pipeline()



# # import logging
# # import json
# # from openai import OpenAI

# # from engine.ingester import IngestionEngine
# # from engine.guardrails import Tier1SafetyGuardrail
# # from engine.extractor import EventExtractorEngine
# # from engine.grounding import GroundingEngine
# # from engine.market_maker import MarketMakerEngine

# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# # )

# # def run_full_pipeline():
# #     print("==========================================================")
# #     print("      FANGRAM AUTOMATED PREDICTION ENGINE (END-TO-END)     ")
# #     print("==========================================================\n")

# #     
# #     TAVILY_KEY = "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"

# #     openai_client = OpenAI(api_key=OPENAI_KEY)

# #     # Initialize Modules
# #     ingester = IngestionEngine()
# #     extractor = EventExtractorEngine(llm_client=openai_client)
# #     grounder = GroundingEngine(tavily_api_key=TAVILY_KEY, client=openai_client)
# #     market_maker = MarketMakerEngine(tavily_api_key=TAVILY_KEY, client=openai_client)

# #     # 1. RSS Ingestion
# #     print("\n--- SECTION 1: INGESTION ---")
# #     raw_signals = ingester.fetch_rss_feed()
# #     print(f"Ingested {len(raw_signals)} raw signals.")

# #     if not raw_signals:
# #         return

# #     # 2. Safety Guardrails
# #     print("\n--- SECTION 1.5: GUARDRAILS ---")
# #     safe_signals = [sig for sig in raw_signals if Tier1SafetyGuardrail.evaluate(sig)[0]]
# #     print(f"Passed Guardrails: {len(safe_signals)} / {len(raw_signals)}")

# #     # 3. Extraction, Grounding & Contract Generation
# #     print("\n--- SECTIONS 2, 3 & 4: EXTRACTION, GROUNDING & TWO-PASS MARKET MAKER ---")
# #     generated_contracts = []

# #     for signal in safe_signals[:3]:  # Process top 3 safe headlines
# #         print(f"\n" + "="*70)
# #         print(f"[HEADLINE]: {signal.title}")
# #         print("="*70)

# #         # Step A: Anchor Event Extraction
# #         event = extractor.extract(signal)
# #         if not event.has_future_uncertainty:
# #             print(f"  └─► REJECTED (Section 2): {event.rejection_reason}")
# #             continue

# #         print(f"  └─► ANCHOR EVENT: '{event.anchor_event}'")
# #         print(f"  └─► ENDPOINTS: {event.candidate_endpoints}")

# #         # Step B: Grounding Verification (Is it real & future?)
# #         grounded = grounder.verify_event(event)
# #         if not grounded or not grounded.is_future_event:
# #             print(f"  └─► REJECTED (Section 3): Event failed web search or is in the past.")
# #             continue

# #         print(f"  └─► GROUNDED DATE: {grounded.exact_date_str}")

# #         # Step C: Two-Pass Market Maker for EACH Candidate Endpoint
# #         for endpoint in event.candidate_endpoints:
# #             print(f"\n  [EVALUATING ENDPOINT]: '{endpoint}'")
# #             contract = market_maker.process_endpoint(
# #                 anchor_event=event.anchor_event, 
# #                 endpoint=endpoint
# #             )

# #             if contract and contract.is_resolvable:
# #                 generated_contracts.append(contract)
# #                 print("\n  🔥 POLYMARKET CONTRACT GENERATED 🔥")
# #                 print(f"    Title:        {contract.market_title}")
# #                 print(f"    Question:     {contract.final_question}")
# #                 print(f"    Lock (IST):   {contract.lock_timestamp_display}")
# #                 print(f"    Lock (UTC):   {contract.lock_timestamp_utc}")
# #                 print(f"    Resolve UTC:  {contract.resolution_timestamp_utc}")
# #                 print(f"    Source URL:   {contract.source_of_truth_url}")
# #                 print(f"    Rules:        {json.dumps(contract.resolution_rules)}")
# #                 print(f"    Void Clause:  {contract.void_clause}")
# #             else:
# #                 reason = contract.unresolvable_reason if contract else "Unknown Error"
# #                 print(f"    └─► DROPPED (Unresolvable): {reason}")

# #     print(f"\n🎉 PIPELINE COMPLETE! Total Valid Contracts Created: {len(generated_contracts)}")

# # if __name__ == "__main__":
# #     run_full_pipeline()






# # import logging
# # from engine.ingester import IngestionEngine
# # from engine.guardrails import Tier1SafetyGuardrail
# # from engine.extractor import EventExtractorEngine
# # from engine.grounding import GroundingEngine
# # from openai import OpenAI

# # # Configure logging output
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# # )

# # def run_test():
# #     print("\n--- STEP 1: INGESTION ---")
# #     ingester = IngestionEngine()
# #     # Fetch live RSS feed
# #     raw_signals = ingester.fetch_rss_feed()
# #     print(f"Total Raw Signals Ingested: {len(raw_signals)}")

# #     if not raw_signals:
# #         print("No signals retrieved. Check network connection.")
# #         return

# #     print("\n--- STEP 2: GUARDRAILS ---")
# #     safe_signals = []
# #     for signal in raw_signals:
# #         is_safe, reason = Tier1SafetyGuardrail.evaluate(signal)
# #         if is_safe:
# #             safe_signals.append(signal)

# #     print(f"Passed Guardrail: {len(safe_signals)} / {len(raw_signals)}")

# #     print("\n--- STEP 3: ANCHOR EVENT EXTRACTION ---\n\n")

# #     extractor = EventExtractorEngine(llm_client=client)  # Runs with OpenAI client
    
# #     extracted_events = []
# #     for signal in safe_signals[:2]:  # Test first 5 safe signals
# #         event = extractor.extract(signal)
# #         if event.has_future_uncertainty:
# #             extracted_events.append(event)
# #             print(f"\n[VALID EVENT EXTRACTED]")
# #             print(f"  Headline:     {signal.title}")
# #             # print(f"  Link:         {signal.link}")
# #             print(f"  Anchor Event: {event.anchor_event}")
# #             print(f"  Endpoints:    {event.candidate_endpoints}")

# #             # Run Grounding!
# #             grounder = GroundingEngine(
# #                 tavily_api_key="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",
# #                 client=client
# #             )
# #             grounded_context = grounder.verify_event(event)
            
# #             if grounded_context:
# #                 print("  --> GROUNDING PASSED!")
# #                 print(f"      Verified Date: {grounded_context.exact_date_str}")
# #                 print(f"      Proof: {grounded_context.grounding_snippet}\n")
# #             else:
# #                 print("  --> GROUNDING FAILED (Dropped from pipeline)\n")



# #         else:
# #             print(f"\n[REJECTED EVENT]")
# #             print(f"  Headline:     {signal.title}")
# #             # print(f"  Link:         {signal.link}")
# #             print(f"  Reason:       {event.rejection_reason}")

# #     print(f"\nTotal Valid Extracted Events: {len(extracted_events)}")



# # if __name__ == "__main__":
# #     run_test()