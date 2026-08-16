import requests
from bs4 import BeautifulSoup

URL = "https://www.bollywoodhungama.com/box-office-collections/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def resolve_movie_metric(target_movie: str, metric_name: str = "Opening Day"):
    response = requests.get(URL, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    if not table:
        print("❌ No table found on page.")
        return None

    # Step 1: Dynamically map column headers to their positional index
    header_row = table.find("tr")
    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
    
    # Step 2: Find the exact column index for our desired metric
    metric_idx = None
    for idx, header in enumerate(headers):
        if metric_name.lower() in header.lower():
            metric_idx = idx
            break

    if metric_idx is None:
        print(f"❌ Metric '{metric_name}' not found in headers: {headers}")
        return None

    print(f"📌 Found '{metric_name}' column at Index {metric_idx} in table header.")

    # Step 3: Scan rows, match target movie, and extract value at metric_idx
    for tr in table.find_all("tr")[1:]:  # Skip header row
        row_text = tr.get_text()
        if target_movie.lower() in row_text.lower():
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            
            if len(cells) > metric_idx:
                opening_day_value = cells[metric_idx]
                print(f"🎯 Matched Row Data: {cells}")
                print(f"✅ Resolved '{metric_name}' for {target_movie}: ₹{opening_day_value} Cr")
                return opening_day_value

    print(f"❌ Movie '{target_movie}' not found in table.")
    return None

if __name__ == "__main__":
    # Test for Awarapan 2 Opening Day
    resolve_movie_metric(target_movie="Awarapan 2", metric_name="Opening Day")

# import requests
# from bs4 import BeautifulSoup

# URL = "https://www.bollywoodhungama.com/box-office-collections/"
# HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# def resolve_awarapan_collection():
#     response = requests.get(URL, headers=HEADERS, timeout=10)
#     soup = BeautifulSoup(response.text, "html.parser")

#     # Iterate through table rows to find 'Awarapan 2'
#     for tr in soup.find_all("tr"):
#         row_text = tr.get_text()
#         if "Awarapan 2" in row_text:
#             cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
#             print(f"🎯 Matched Row Data: {cells}")
            
#             # Extract opening day collection (e.g., '3.79')
#             for cell in cells:
#                 if "3.7" in cell:
#                     print(f"✅ Resolved Day 1 Collection: ₹{cell} Cr")
#                     return cell
                    
#     print("❌ Movie row or collection value not found.")
#     return None

# if __name__ == "__main__":
#     resolve_awarapan_collection()

# import os
# import requests
# from bs4 import BeautifulSoup
# from tavily import TavilyClient

# # ---------------------------------------------------------------------
# # CONFIGURATION
# # ---------------------------------------------------------------------
# TARGET_URL = "https://www.bollywoodhungama.com/box-office-collections/"
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY_HERE")

# def test_1_raw_requests():
#     """Test 1: Standard Python requests + BeautifulSoup (Static HTML)."""
#     print("\n" + "=" * 75)
#     print("TEST 1: RAW PYTHON REQUESTS + BEAUTIFULSOUP (STATIC HTML)")
#     print("=" * 75)

#     headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
#     try:
#         response = requests.get(TARGET_URL, headers=headers, timeout=10)
#         soup = BeautifulSoup(response.text, "html.parser")

#         # Search for table tags in static HTML
#         tables = soup.find_all("table")
#         print(f"Total HTML <table> tags found: {len(tables)}")

#         # Search if dynamic movie names exist in static source
#         has_awarapan = "Awarapan 2" in response.text
#         has_batwara = "Batwara 1947" in response.text

#         print(f"Contains 'Awarapan 2' in static HTML?   -> {has_awarapan}")
#         print(f"Contains 'Batwara 1947' in static HTML? -> {has_batwara}")

#         # Snippet preview
#         text_preview = soup.get_text()[:400].replace('\n', ' ')
#         print(f"\nStatic Content Preview:\n\"{text_preview}...\"")

#     except Exception as e:
#         print(f"Request failed: {e}")


# def test_2_tavily_basic_extract(client: TavilyClient):
#     """Test 2: Tavily Extract with basic depth."""
#     print("\n" + "=" * 75)
#     print("TEST 2: TAVILY EXTRACT (DEPTH = 'basic')")
#     print("=" * 75)

#     try:
#         response = client.extract(
#             urls=[TARGET_URL],
#             extract_depth="basic"
#         )
#         results = response.get("results", [])
#         if not results:
#             print("No extraction results returned.")
#             return

#         raw_content = results[0].get("raw_content", "")
#         print(f"Extracted Length: {len(raw_content)} characters")

#         has_awarapan = "Awarapan 2" in raw_content
#         has_batwara = "Batwara 1947" in raw_content
#         has_table_headers = "Opening Day" in raw_content or "INR cr." in raw_content

#         print(f"Contains Table Headers ('Opening Day')? -> {has_table_headers}")
#         print(f"Contains 'Awarapan 2'?                   -> {has_awarapan}")
#         print(f"Contains 'Batwara 1947'?                 -> {has_batwara}")

#         print(f"\nBasic Extract Preview:\n\"{raw_content[:500]}...\"")

#     except Exception as e:
#         print(f"Tavily Basic Extract failed: {e}")


# def test_3_tavily_advanced_extract(client: TavilyClient):
#     """Test 3: Tavily Extract with advanced depth (Handles JS rendering & tables)."""
#     print("\n" + "=" * 75)
#     print("TEST 3: TAVILY EXTRACT (DEPTH = 'advanced' - JS RENDERED)")
#     print("=" * 75)

#     try:
#         response = client.extract(
#             urls=[TARGET_URL],
#             extract_depth="advanced"
#         )
#         results = response.get("results", [])
#         if not results:
#             print("No extraction results returned.")
#             return

#         raw_content = results[0].get("raw_content", "")
#         print(f"Extracted Length: {len(raw_content)} characters")

#         has_awarapan = "Awarapan 2" in raw_content
#         has_batwara = "Batwara 1947" in raw_content
#         has_numbers = "1.15" in raw_content or "3.79" in raw_content

#         print(f"Contains 'Awarapan 2'?                   -> {has_awarapan}")
#         print(f"Contains 'Batwara 1947'?                 -> {has_batwara}")
#         print(f"Contains Table Numbers (3.79 / 1.15)?    -> {has_numbers}")

#         print("\n--- EXTRACTED TABLE / CONTENT SAMPLE ---")
#         # Print snippet surrounding any found match
#         if "Awarapan 2" in raw_content:
#             idx = raw_content.find("Awarapan 2")
#             snippet = raw_content[max(0, idx - 100): min(len(raw_content), idx + 300)]
#             print(f"Matched Snippet around 'Awarapan 2':\n\"{snippet}\"")
#         else:
#             print(f"Raw Snippet Preview:\n\"{raw_content[:600]}...\"")

#     except Exception as e:
#         print(f"Tavily Advanced Extract failed: {e}")


# if __name__ == "__main__":
#     if TAVILY_API_KEY == "YOUR_TAVILY_API_KEY_HERE":
#         print("Please export TAVILY_API_KEY in your environment or set it in the script.")
#     else:
#         client = TavilyClient(api_key=TAVILY_API_KEY)
        
#         # Run all three tests to inspect differences
#         test_1_raw_requests()
#         test_2_tavily_basic_extract(client)
#         test_3_tavily_advanced_extract(client)