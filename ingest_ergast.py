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
        # Points
        points_lines = [
            f"P{r['position']} {fmt_driver(r)} ({fmt_team(r)}) — {r.get('points', 0)} pts"
            for r in results[:10]
        ]
        lines.append("Top 10 finishers: " + "; ".join(points_lines) + ".")

        # Retirements
        finished_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps"}
        retirements = [
            r for r in results
            if r.get("status", "") not in finished_statuses
            and not r.get("status", "").startswith("+")
        ]
        if retirements:
            ret_str = "; ".join(
                f"{fmt_driver(r)} — {r.get('status', '?')}" for r in retirements[:6]
            )
            lines.append(f"Retirements/incidents: {ret_str}.")
    lines.append("")

    # Qualifying
    if quali:
        pole      = quali[0]
        pole_name = fmt_driver(pole)
        pole_team = fmt_team(pole)
        pole_time = pole.get("Q3", pole.get("Q2", pole.get("Q1", "?")))
        lines.append(
            f"Qualifying: {pole_name} ({pole_team}) took pole position with a lap of {pole_time}."
        )
        top5 = "; ".join(
            f"P{q['position']} {fmt_driver(q)} ({q.get('Q3', q.get('Q2', q.get('Q1', '?')))})"
            for q in quali[:5]
        )
        lines.append(f"Top 5 qualifying times: {top5}.")

    lines.append("")

     # Fastest lap
    if fastest:
        fl        = fastest[0]
        fl_driver = fmt_driver(fl)
        fl_team   = fmt_team(fl)
        fl_time   = fl.get("FastestLap", {}).get("Time", {}).get("time", "?")
        fl_lap    = fl.get("FastestLap", {}).get("lap", "?")
        fl_speed  = fl.get("FastestLap", {}).get("AverageSpeed", {}).get("speed", "?")
        lines.append(
            f"Fastest lap: {fl_driver} ({fl_team}) on lap {fl_lap} "
            f"with a time of {fl_time} ({fl_speed} kph average)."
        )

    lines.append("")

    # Pit stop summary
    if pitstops:
        stop_counts: dict[str, int] = {}
        for stop in pitstops:
            drv = stop.get("driverId", "unknown")
            stop_counts[drv] = stop_counts.get(drv, 0) + 1

        one_stop  = [d for d, c in stop_counts.items() if c == 1]
        two_stop  = [d for d, c in stop_counts.items() if c == 2]
        three_plus = [d for d, c in stop_counts.items() if c >= 3]

        pit_parts = []
        if one_stop:
            pit_parts.append(f"{len(one_stop)} driver(s) on a one-stop strategy")
        if two_stop:
            pit_parts.append(f"{len(two_stop)} driver(s) on a two-stop strategy")
        if three_plus:
            pit_parts.append(f"{len(three_plus)} driver(s) on three or more stops")
        if pit_parts:
            lines.append("Pit stop strategies: " + "; ".join(pit_parts) + ".")

        # Quickest stop
        timed = [s for s in pitstops if s.get("duration")]
        if timed:
            quickest = min(timed, key=lambda s: float(s["duration"]))
            lines.append(
                f"Fastest pit stop: {quickest.get('driverId', '?')} on lap "
                f"{quickest.get('lap', '?')} in {quickest.get('duration', '?')}s."
            )

    return "\n".join(lines)


# Main
def ingest_season(season: int):
    print(f"\nSeason {season}...")
    data = fetch(f"{season}/races")
    if not data:
        print(f"  [!] Could not fetch race schedule")
        return

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    print(f"  {len(races)} races found")

    for race in races:
        round_num = race["round"]
        race_name = race.get("raceName", f"Round {round_num}")
        slug      = race_name.lower().replace(" ", "_").replace("/", "-")
        out_path  = OUTPUT_DIR / f"{season}_{int(round_num):02d}_{slug}.txt"

        if out_path.exists():
            print(f"  [skip] {out_path.name}")
            continue

        print(f"  R{round_num}: {race_name}...", end=" ", flush=True)
        passage = build_race_passage(season, race)
        out_path.write_text(passage, encoding="utf-8")
        print(f"done ({len(passage)} chars)")
        time.sleep(DELAY)


if __name__ == "__main__":
    print("PitWall AI — Ergast ingestion")
    print(f"Seasons: {SEASONS[0]}–{SEASONS[-1]}  |  Output: {OUTPUT_DIR.resolve()}")
    print("-" * 60)

    for season in SEASONS:
        ingest_season(season)

    files      = list(OUTPUT_DIR.glob("*.txt"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"\nComplete. {len(files)} passages saved (~{total_size // 1024} KB)")
    print("Next: run python ingest_fastf1.py")