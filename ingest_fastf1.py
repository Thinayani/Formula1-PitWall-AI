# Fetches lap-by-lap telemetry and tyre strategy data using the FastF1 library.
# Produces text passages describing stint structure, compound choices, and lap times.

import fastf1
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Config

# FastF1 only has reliable data from 2018 onward
SEASONS    = list(range(2018, 2024))
OUTPUT_DIR = Path("data/passages")
CACHE_DIR  = Path("data/fastf1_cache")

# Setup

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


# Passage builder

def safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def format_laptime(td) -> str:
    """Convert a pandas Timedelta to a human-readable lap time string."""
    try:
        total_seconds = td.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:06.3f}"
    except Exception:
        return "?"


def build_telemetry_passage(season: int, event) -> str | None:
    """
    Load a race session and build a passage describing:
    - Each driver's stint structure (which compounds, how many laps)
    - Fastest lap per driver
    - Safety car / VSC laps if detectable
    """
    race_name = event.get("EventName", "Unknown")
    round_num = event.get("RoundNumber", "?")

    try:
        session = fastf1.get_session(season, int(round_num), "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"    [!] Could not load session: {e}")
        return None

    laps = session.laps
    if laps is None or laps.empty:
        print(f"    [!] No lap data available")
        return None

    lines = []
    lines.append(f"F1 Telemetry & Strategy — {race_name} ({season}), Round {round_num}")
    lines.append("")

    # Per-driver stint analysis
    driver_summaries = []
    drivers = laps["Driver"].unique()

    for driver in sorted(drivers):
        drv_laps = laps[laps["Driver"] == driver].copy()
        if drv_laps.empty:
            continue

        # Build stint list: group consecutive laps on the same compound
        drv_laps = drv_laps.sort_values("LapNumber")
        stints = []
        current_compound = None
        stint_start      = None
        stint_lap_count  = 0

        for _, lap in drv_laps.iterrows():
            compound = lap.get("Compound", "UNKNOWN")
            if compound != current_compound:
                if current_compound is not None:
                    stints.append({
                        "compound":   current_compound,
                        "start_lap":  stint_start,
                        "laps":       stint_lap_count,
                    })
                current_compound = compound
                stint_start      = int(lap["LapNumber"])
                stint_lap_count  = 1
            else:
                stint_lap_count += 1

        if current_compound:
            stints.append({
                "compound":  current_compound,
                "start_lap": stint_start,
                "laps":      stint_lap_count,
            })

        # Fastest lap for this driver
        valid_laps     = drv_laps[drv_laps["IsPersonalBest"] == True]
        fastest_lap_td = drv_laps["LapTime"].min()
        fastest_lap_no = drv_laps.loc[drv_laps["LapTime"].idxmin(), "LapNumber"] \
                         if not drv_laps["LapTime"].isna().all() else "?"

        stint_str = " → ".join(
            f"{s['compound']} ({s['laps']} laps from lap {s['start_lap']})"
            for s in stints
            if s["compound"] not in (None, "UNKNOWN", "nan")
        )

        fl_str = format_laptime(fastest_lap_td) if pd.notna(fastest_lap_td) else "?"

        summary = (
            f"{driver}: {len(stints)} stint(s) — {stint_str}. "
            f"Fastest lap: {fl_str} on lap {fastest_lap_no}."
        )
        driver_summaries.append(summary)

    if driver_summaries:
        lines.append("Tyre strategies and fastest laps per driver:")
        lines.extend(driver_summaries)
        lines.append("")

    # Compound popularity
    compound_counts = (
        laps[laps["Compound"].notna()]
        .groupby("Compound")["LapNumber"]
        .count()
        .sort_values(ascending=False)
    )
    if not compound_counts.empty:
        compound_str = "; ".join(
            f"{comp}: {count} laps"
            for comp, count in compound_counts.items()
            if str(comp) not in ("nan", "UNKNOWN")
        )
        lines.append(f"Total laps per compound: {compound_str}.")
        lines.append("")

    # Approximate SC laps (unusually slow laps)
    try:
        median_lap = laps["LapTime"].median()
        if pd.notna(median_lap):
            slow_threshold = median_lap * 1.15  # 15% slower than median
            slow_laps      = laps[laps["LapTime"] > slow_threshold]["LapNumber"].unique()
            if len(slow_laps) > 0:
                slow_str = ", ".join(str(int(l)) for l in sorted(slow_laps)[:10])
                lines.append(
                    f"Laps significantly slower than median (possible safety car / VSC): {slow_str}."
                )
    except Exception:
        pass

    return "\n".join(lines)


# Main

def ingest_season(season: int):
    print(f"\nSeason {season}...")
    try:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
    except Exception as e:
        print(f"  [!] Could not fetch schedule: {e}")
        return

    # Filter to rounds that have already happened
    races = schedule[schedule["EventFormat"] != "testing"]

    for _, event in races.iterrows():
        round_num = int(event.get("RoundNumber", 0))
        race_name = event.get("EventName", f"Round {round_num}")
        slug      = race_name.lower().replace(" ", "_").replace("/", "-").replace("'", "")
        out_path  = OUTPUT_DIR / f"{season}_{round_num:02d}_{slug}_telemetry.txt"

        if out_path.exists():
            print(f"  [skip] {out_path.name}")
            continue

        print(f"  R{round_num}: {race_name}...", end=" ", flush=True)

        passage = build_telemetry_passage(season, event)
        if passage:
            out_path.write_text(passage, encoding="utf-8")
            print(f"done ({len(passage)} chars)")
        else:
            print("skipped (no data)")


if __name__ == "__main__":
    print("PitWall AI — FastF1 telemetry ingestion")
    print(f"Seasons: {SEASONS[0]}–{SEASONS[-1]}  |  Cache: {CACHE_DIR.resolve()}")
    print("Note: first run downloads ~50–200 MB per season. Be patient.")
    print("-" * 60)

    for season in SEASONS:
        ingest_season(season)

    files      = list(OUTPUT_DIR.glob("*_telemetry.txt"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"\nComplete. {len(files)} telemetry passages saved (~{total_size // 1024} KB)")
    print("Next: run python embed_and_index.py")
