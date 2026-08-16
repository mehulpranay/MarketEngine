import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests
import http.client
import json
import sqlite3
from datetime import datetime
from openai import OpenAI
import time
from datetime import datetime, timezone
import os

# client = OpenAI(api_key="your_openai_key")
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)
RAPIDAPI_KEY = "e2a44a97d2mshb4c76758df3d610p1065ebjsnb731792f5363" 

import requests
import time
from datetime import datetime, timezone

def fetch_matches():
    url_upcoming = f"https://cricbuzz-cricket.p.rapidapi.com/matches/v1/upcoming?t={int(time.time())}"
    url_live = f"https://cricbuzz-cricket.p.rapidapi.com/matches/v1/live?t={int(time.time())}"
    
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com"
    }

    
    HIGH_ENGAGEMENT_TEAMS = [
        "India", "Australia", "England", "Pakistan", 
        "South Africa", "New Zealand", "West Indies", "Bangladesh", "TBC", "TBD"
    ]
    EXCLUDED_FORMATS = []
    
    seen_match_ids = set()
    target_matches = []

    

    for url in [url_live, url_upcoming]:

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for RapidAPI quota or error messages
            if isinstance(data, dict) and "message" in data:
                print(f"API Error Message: {data['message']}")
                continue
        
            
            for type_match in data.get("typeMatches", []):
                for series in type_match.get("seriesMatches", []):
                    wrapper = series.get("seriesAdWrapper")
                    matches_list = wrapper.get("matches", []) if wrapper else series.get("matches", [])
                        
                    for match in matches_list:
                        info = match.get("matchInfo")
                        if not info:
                            continue
                            
                        match_id = info["matchId"]
                        if match_id in seen_match_ids:
                            continue
                            
                        team1 = info.get("team1", {}).get("teamName", "TBC")
                        team2 = info.get("team2", {}).get("teamName", "TBC")
                        series_name = info.get("seriesName", "Unknown Series")
                        fmt = info.get("matchFormat", "")
                        state = info.get("state", "").title()
                        status = info.get("status", "")
                        
                        # Ensure match has not started yet
                        is_pre_match = state in ["Upcoming", "Preview", ""] or status.startswith("Match starts")
                        is_in_progress = state in ["In Progress", "Inning Break", "Complete"] or "opt to" in status.lower() or "need" in status.lower()
                        
                        if is_in_progress or not is_pre_match:
                            continue
                        
                        # EXACTLY TWO CONDITIONS: Format is HUN OR Team is High Engagement
                        is_hun = (fmt == "HUN")
                        is_high_engagement = (team1 in HIGH_ENGAGEMENT_TEAMS or team2 in HIGH_ENGAGEMENT_TEAMS)
                        
                        if fmt not in EXCLUDED_FORMATS and (is_hun or is_high_engagement):
                            start_time_sec = int(info["startDate"]) / 1000
                            lock_time_sec = start_time_sec - (15 * 60)
                            
                            start_iso = datetime.fromtimestamp(start_time_sec, tz=timezone.utc).isoformat()
                            lock_iso = datetime.fromtimestamp(lock_time_sec, tz=timezone.utc).isoformat()
                            
                            target_matches.append({
                                "matchId": match_id,
                                "seriesName": series_name,
                                "matchFormat": fmt,
                                "team1": team1,
                                "team2": team2,
                                "state": state if state else "Preview",
                                "status": status,
                                "startDate": info["startDate"],
                                "startTimeIso": start_iso,
                                "lockTimeIso": lock_iso,
                                "venue": info.get("venueInfo", {}).get("ground", "Unknown Ground")
                            })
                            seen_match_ids.add(match_id)
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error occurred for {url}: {err}")
        except Exception as e:
            print(f"Unexpected error for {url}: {e}")
                
    
    return target_matches

# if __name__ == "__main__":
#     target_matches = fetch_matches()
#     for m in target_matches:
#         print(m)

def fetch_scorecard(match_id):

    """
    Fetches the scorecard for a specific cricket match from Cricbuzz API.
    
    Input:
        - match_id : unique Cricbuzz match identifier (from fetch_matches output)
    
    Process:
        - Calls mcenter/v1/{match_id}/hscard endpoint
        - Adds cache-busting timestamp parameter to ensure fresh data every call
        - Extracts match completion status, result string, and innings details

    Note:
    - isMatchComplete flag is unreliable — Cricbuzz sometimes returns False even
      when match is complete. Use "won by" or "abandoned" in status string as
      primary completion check.

    """

    
    import time
    
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{match_id}/hscard?t={int(time.time())}"
    
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    return {
        "isMatchComplete": data.get("isMatchComplete", False),
        "status": data.get("status", ""),
        "scorecard": data.get("scorecard", [])
    }


# match_id = target_matches[0]['matchId']
# scorecard = fetch_scorecard(match_id)
# print(scorecard["isMatchComplete"])
# print(scorecard["status"])

def generate_question(match_info):
    prompt = f"""
You are a cricket prediction engine. Given pre-match information, generate ONE engaging yes/no prediction question. 
For V1, only generate match winner questions that can be resolved purely from the final match result status string. 

CONTEXT: This prediction is being generated BEFORE the match starts and BEFORE the toss.

Match details:
- Team 1: {match_info['team1']}
- Team 2: {match_info['team2']}
- Format: {match_info['matchFormat']}
- Series: {match_info['seriesName']}
- Venue:  {match_info['venue']}
- Start Time: {match_info['startTimeIso']}

Rules:
- Question must be yes/no only
- Must be resolvable from match result
- Must be engaging for fans
- No subjective questions

Good examples:
- "Will Sunrisers Leeds win this match?"
- "Will Southern Brave win this match?"

Bad examples:
- "Will it be an exciting match?" (subjective)
- "Will fans enjoy this?" (unresolvable)

Output strictly in this JSON format:
{{
    "question": "...",
    "probability": 0.XX,
    "probability_reasoning": "explain specifically what historical knowledge, team strength, venue history, or contextual factors informed your pre-match probability estimate. Be honest if you have limited information.",
    "resolution_rule": "...",
    "reasoning": "Explain why this question is compelling and significant for fans — what narrative, rivalry, or stakes make it worth predicting on."
}}
"""


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    
    if not content:
        print("Empty response from model")
        return None
    
    return json.loads(content)


# if __name__ == "__main__":
#     match = {'matchId': 163013, 'seriesName': 'India tour of Sri Lanka 2026', 'matchFormat': 'TEST', 'team1': 'India', 'team2': 'Sri Lanka', 'state': 'In Progress', 'status': 'Day 1: 2nd Session', 'startDate': '1786768200000', 'startTimeIso': '2026-08-15T04:30:00+00:00', 'lockTimeIso': '2026-08-15T04:15:00+00:00', 'venue': 'Galle International Stadium, Galle'}
#     # match = target_matches[0]
#     if match['state'] == 'Complete':
#         print(f"Skipping completed match: {match['seriesName']}")
#     else:
#         question_data = generate_question(match)
#         print(question_data)



def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matchId TEXT,
            team1 TEXT,
            team2 TEXT,
            seriesName TEXT,
            match_format TEXT,
            match_date TEXT,
            match_start_time TEXT,
            lock_time TEXT,
            market_status TEXT DEFAULT 'OPEN',
            venue TEXT,
            question TEXT,
            probability REAL,
            probability_reasoning TEXT,
            resolution_rule TEXT,
            reasoning TEXT,
            actual_outcome TEXT,
            is_correct INTEGER,
            created_at TEXT,
            brier_score REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(match_info, question_data):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions 
        (matchId, team1, team2, seriesName, match_format, match_date, 
        match_start_time, lock_time, market_status, venue,
        question, probability, probability_reasoning, resolution_rule, reasoning, 
        actual_outcome, is_correct, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
    ''', (
        match_info['matchId'],
        match_info['team1'],
        match_info['team2'],
        match_info['seriesName'],
        match_info['matchFormat'],
        match_info['startDate'],
        match_info['startTimeIso'],
        match_info['lockTimeIso'],
        match_info.get('venue', 'Unknown'),
        question_data['question'],
        question_data['probability'],
        question_data['probability_reasoning'],
        question_data['resolution_rule'],
        question_data['reasoning'],
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()

def resolve_prediction(prediction, status_string):
    prompt = f"""
You are resolving a cricket prediction based on match result.

Based on the match status, did this question resolve YES or NO?
If match was abandoned or result is unclear, say UNRESOLVED.

Output only one word: YES, NO, or UNRESOLVED

Examples:

Question: "Will India win the match?"
Resolution rule: "Resolves YES if India wins, NO if they lose or draw"
Match status: "India won by 6 wickets"
Answer: YES

Question: "Will Australia win the match?"
Resolution rule: "Resolves YES if Australia wins, NO if they lose or draw"
Match status: "England won by 45 runs"
Answer: NO

Question: "Will Sunrisers Leeds win the match?"
Resolution rule: "Resolves YES if Sunrisers Leeds wins, NO if they lose"
Match status: "Match abandoned due to rain"
Answer: UNRESOLVED

Question: {prediction['question']}
Resolution rule: {prediction['resolution_rule']}
Match status: {status_string}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":

    prediction = {'question':'Will India lose this match',
                'resolution_rule': 'Resolves YES if India wins, NO if they lose or draw'}
    status_string = 'India won by 6 wickets'
    result = resolve_prediction(prediction, status_string)
    print(result)


def check_and_lock_markets():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, question, lock_time 
        FROM predictions 
        WHERE market_status = 'OPEN'
    ''')
    open_markets = cursor.fetchall()
    
    now_utc = datetime.now(timezone.utc)
    
    for market in open_markets:
        market_id, question, lock_time_iso = market
        lock_dt = datetime.fromisoformat(lock_time_iso)
        
        if now_utc >= lock_dt:
            cursor.execute('''
                UPDATE predictions 
                SET market_status = 'LOCKED' 
                WHERE id = ?
            ''', (market_id,))
            print(f"🔒 Locked market #{market_id}: {question}")
            
    conn.commit()
    conn.close()


def check_and_resolve():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, matchId, question, resolution_rule, probability 
        FROM predictions 
        WHERE market_status = 'LOCKED'
    ''')
    locked_predictions = cursor.fetchall()
    
    for pred in locked_predictions:
        pred_id, match_id, question, resolution_rule, probability = pred
        
        scorecard = fetch_scorecard(match_id)
        status = scorecard['status']
        
        is_complete = (
            scorecard['isMatchComplete'] or 
            "won by" in status.lower() or 
            "abandoned" in status.lower() or
            "draw" in status.lower() or
            "no result" in status.lower()
        )
        
        if is_complete:
            outcome = resolve_prediction(
                {'question': question, 'resolution_rule': resolution_rule},
                status
            )
            
            if outcome in ['YES', 'NO']:
                actual_numeric = 1 if outcome == "YES" else 0
                is_correct = 1 if (
                    (outcome == 'YES' and probability >= 0.5) or
                    (outcome == 'NO' and probability < 0.5)
                ) else 0
                
                brier_score = (probability - actual_numeric) ** 2
                
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_outcome = ?, is_correct = ?, brier_score = ?, market_status = 'RESOLVED'
                    WHERE id = ?
                ''', (outcome, is_correct, brier_score, pred_id))
                
                print(f"✅ RESOLVED #{pred_id}: {question} → {outcome} | Brier: {brier_score:.4f}")
                
            elif outcome == 'UNRESOLVED':
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_outcome = 'VOID', market_status = 'VOID'
                    WHERE id = ?
                ''', (pred_id,))
                
                print(f"⚠️ VOIDED #{pred_id}: Match abandoned or no result.")
        else:
            print(f"⏳ Still in progress (#{pred_id}): {status}")
            
    conn.commit()
    conn.close()



def check_existing_prediction(match_id):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM predictions WHERE matchId = ?', (str(match_id),))
    result = cursor.fetchone()
    conn.close()
    return result

def run_pipeline():
    print(f"\n{'='*50}")
    print(f"Pipeline run: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*50}")
    
    # -------------------------------------------------------------
    # Pass 1: Fetch Upcoming Matches & Create Open Markets
    # -------------------------------------------------------------
    print("\n[Pass 1] Fetching & creating upcoming markets...")
    upcoming_matches = fetch_matches()
    print(f"Found {len(upcoming_matches)} target upcoming matches.")
    
    for match in upcoming_matches:
        existing = check_existing_prediction(match['matchId'])
        if not existing:
            print(f"Generating market for: {match['team1']} vs {match['team2']} ({match['seriesName']})")
            question_data = generate_question(match)
            if question_data:
                save_prediction(match, question_data)
                print(f"Created market #{match['matchId']}: {question_data['question']}")
        else:
            print(f"Market already exists for matchId: {match['matchId']}")
    
    # -------------------------------------------------------------
    # Pass 2: Lock Expired Open Markets (Time >= Lock Time)
    # -------------------------------------------------------------
    print("\n[Pass 2] Checking open markets for betting lock...")
    check_and_lock_markets()
    
    # -------------------------------------------------------------
    # Pass 3: Check & Resolve Locked Markets
    # -------------------------------------------------------------
    print("\n[Pass 3] Checking locked markets for resolution...")
    check_and_resolve()


# -------------------------------------------------------------
# Execution Entry Point (Loop every 15 minutes)
# -------------------------------------------------------------
init_db()  # Ensures SQLite schema is initialized with lock_time & market_status

while True:
    run_pipeline()
    print("\nSleeping for 15 minutes...")
    time.sleep(15 * 60)