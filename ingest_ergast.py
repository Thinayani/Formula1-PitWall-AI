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

def get_qualifying(season: int, round_num: int) -> list[dict]:
    data = fetch(f"{season}/{round_num}/qualifying")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("QualifyingResults", []) if races else []


def get_fastest_laps(season: int, round_num: int) -> list[dict]:
    data = fetch(f"{season}/{round_num}/fastest/1/results")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("Results", []) if races else []


def get_pit_stops(season: int, round_num: int) -> list[dict]:
    data = fetch(f"{season}/{round_num}/pitstops")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("PitStops", []) if races else []

# Passage builder

def fmt_driver(result: dict) -> str:
    d = result.get("Driver", {})
    return f"{d.get('givenName', '')} {d.get('familyName', '')}".strip()


def fmt_team(result: dict) -> str:
    return result.get("Constructor", {}).get("name", "Unknown")


def build_race_passage(season: int, race_meta: dict) -> str:
    round_num = int(race_meta["round"])
    race_name = race_meta.get("raceName", "Unknown Race")
    circuit   = race_meta.get("Circuit", {}).get("circuitName", "Unknown Circuit")
    country   = race_meta.get("Circuit", {}).get("Location", {}).get("country", "")
    date      = race_meta.get("date", "")

    results  = get_race_results(season, round_num)
    quali    = get_qualifying(season, round_num)
    fastest  = get_fastest_laps(season, round_num)
    pitstops = get_pit_stops(season, round_num)

    time.sleep(DELAY)

    lines = []
    lines.append(f"Formula 1 — {race_name} ({season})")
    lines.append(f"Circuit: {circuit}, {country}. Race date: {date}. Round {round_num} of the {season} season.")
    lines.append("")

    # Podium & race winner
    if results:
        top3 = results[:3]
        podium = ", ".join(
            f"P{r['position']} {fmt_driver(r)} ({fmt_team(r)})" for r in top3
        )
        lines.append(f"Podium: {podium}.")

        winner = top3[0]
        w_name  = fmt_driver(winner)
        w_team  = fmt_team(winner)
        w_laps  = winner.get("laps", "?")
        w_time  = winner.get("Time", {}).get("time", "no time recorded")
        w_grid  = winner.get("grid", "?")
        lines.append(
            f"{w_name} driving for {w_team} won the race from grid position {w_grid}, "
            f"completing {w_laps} laps in {w_time}."
        )
