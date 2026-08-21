"""
Section 3 scaffolding: Metric Registry

Static mapping of question templates to their BH resolution source.
No LLM involved here — this is what the Section 3 prompt will reference
so the LLM picks from known-resolvable templates instead of inventing
search queries.
"""

METRIC_REGISTRY = [
    {
        "metric_id": "day1_threshold",
        "question_template": "Will '{movie}' gross over ₹{threshold} Cr India Net on Day 1?",
        "resolution_days_after_release": 1,
        "source_url_template": "https://www.bollywoodhungama.com/movie/{slug}/box-office/",
        "resolution_field": "Opening Day",
        "resolution_question_text": "What is the Day 1 Box Office Collection of {movie}?",
    },
    {
        "metric_id": "opening_weekend_threshold",
        "question_template": "Will '{movie}' gross over ₹{threshold} Cr India Net in its Opening Weekend?",
        "resolution_days_after_release": 3,
        "source_url_template": "https://www.bollywoodhungama.com/movie/{slug}/box-office/",
        "resolution_field": "End of Opening Weekend",
        "resolution_question_text": "What were the opening weekend collections of {movie}?",
    },
    {
        "metric_id": "week1_crore_club",
        "question_template": "Will '{movie}' enter the ₹{threshold} Cr Club within 7 days of release?",
        "resolution_days_after_release": 7,
        
        "source_url_template": "https://www.bollywoodhungama.com/movie/{slug}/box-office/",
        "resolution_field": "End of Week 1",
        "resolution_question_text": "What were the opening week collections of {movie}?",
    },
    {
        "metric_id": "verdict_hit",
        "question_template": "Will '{movie}' be declared a Hit or better by BollywoodHungama's official verdict?",
        "resolution_days_after_release": 30,
        "source_url_template": None,  # NOT YET CONFIRMED — only seen on the aggregate
                                       # box-office-collections table, not this movie page.
                                       # Needs verifying before this entry is usable.
        "resolution_question_text": None,
    },
]
