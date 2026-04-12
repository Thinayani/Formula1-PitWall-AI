# Fetches race results, qualifying, and fastest lap data from the Ergast API
# and converts each race weekend into a readable text passage for embedding.
#
# Run: python ingest_ergast.py
# Output: data/passages/<season>_<round>_<race_name>.txt

import requests
import time
from pathlib import Path

# Config
SEASONS    = list(range(2010, 2024))
OUTPUT_DIR = Path("data/passages")
BASE_URL   = "https://ergast.com/api/f1"
DELAY      = 0.5  # seconds between requests — be polite to the free API

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch(endpoint: str) -> dict | None:
    url = f"{BASE_URL}/{endpoint}.json?limit=100"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [!] Request failed: {url}\n      {e}")
        return None


def get_race_results(season: int, round_num: int) -> list[dict]:
    data = fetch(f"{season}/{round_num}/results")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("Results", []) if races else []
