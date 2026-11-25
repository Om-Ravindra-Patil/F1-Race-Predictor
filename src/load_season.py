"""
Load F1 race results for a given season using fastf1.
Saves consolidated results to data/raw/season_<year>_results.csv
"""

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import fastf1
import pandas as pd
import requests
from pathlib import Path
from typing import Optional
import time

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Enable fastf1's local cache — critical for not hammering the API
fastf1.Cache.enable_cache(str(CACHE_DIR))

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


def load_race_results_jolpica(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Fallback: pull race results from Jolpica (Ergast replacement) when fastf1 fails."""
    try:
        url = f"{JOLPICA_BASE}/{year}/{round_num}/results.json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None

        race = races[0]
        results = race.get("Results", [])
        if not results:
            return None

        rows = []
        for r in results:
            rows.append({
                "DriverNumber": r.get("number"),
                "Abbreviation": r["Driver"].get("code"),
                "DriverId": r["Driver"].get("driverId"),
                "FirstName": r["Driver"].get("givenName"),
                "LastName": r["Driver"].get("familyName"),
                "FullName": f"{r['Driver'].get('givenName', '')} {r['Driver'].get('familyName', '')}".strip(),
                "TeamName": r["Constructor"].get("name"),
                "TeamId": r["Constructor"].get("constructorId"),
                "Position": r.get("position"),
                "ClassifiedPosition": r.get("positionText"),
                "GridPosition": r.get("grid"),
                "Status": r.get("status"),
                "Points": r.get("points"),
                "Laps": r.get("laps"),
                "Time": r.get("Time", {}).get("time") if r.get("Time") else None,
                "Year": year,
                "Round": round_num,
                "EventName": race.get("raceName"),
                "EventDate": race.get("date"),
                "Circuit": race.get("Circuit", {}).get("Location", {}).get("locality"),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"    Jolpica fallback also failed: {e}")
        return None

def load_qualifying_results_jolpica(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Fallback: pull qualifying results from Jolpica when fastf1 fails."""
    try:
        url = f"{JOLPICA_BASE}/{year}/{round_num}/qualifying.json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None

        race = races[0]
        results = race.get("QualifyingResults", [])
        if not results:
            return None

        rows = []
        for r in results:
            rows.append({
                "DriverNumber": r.get("number"),
                "Abbreviation": r["Driver"].get("code"),
                "DriverId": r["Driver"].get("driverId"),
                "FullName": f"{r['Driver'].get('givenName', '')} {r['Driver'].get('familyName', '')}".strip(),
                "TeamName": r["Constructor"].get("name"),
                "Position": r.get("position"),  # qualifying position
                "Q1": r.get("Q1"),
                "Q2": r.get("Q2"),
                "Q3": r.get("Q3"),
                "Year": year,
                "Round": round_num,
                "EventName": race.get("raceName"),
                "EventDate": race.get("date"),
                "Circuit": race.get("Circuit", {}).get("Location", {}).get("locality"),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"    Jolpica qualifying fallback failed: {e}")
        return None


def load_qualifying_results(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Load qualifying results for a single round. Returns None if unavailable."""
    try:
        session = fastf1.get_session(year, round_num, "Q")  # 'Q' = Qualifying
        session.load(laps=False, telemetry=False, weather=False, messages=False)  # results only

        results = session.results.copy()
        # Add metadata columns
        results["Year"] = year
        results["Round"] = round_num
        results["EventName"] = session.event["EventName"]
        results["EventDate"] = session.event["EventDate"].strftime("%Y-%m-%d")
        results["Circuit"] = session.event["Location"]  # needed for races not yet run

        return results

    except Exception as e:
        print(f"  Round {round_num} qualifying failed: {e}")
        return None


def load_race_results(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Load race results for a single round. Returns None if unavailable."""
    try:
        session = fastf1.get_session(year, round_num, "R")  # 'R' = Race
        session.load(laps=False, telemetry=False, weather=False, messages=False)  # results only

        results = session.results.copy()
        # fastf1 sometimes returns only the finishing order (no grid/status/points);
        # treat that as a failure so the Jolpica fallback fills it in
        if results["GridPosition"].isna().all():
            print(f"Round {round_num}: fastf1 results incomplete (no grid positions)")
            return None

        # Add metadata columns
        results["Year"] = year
        results["Round"] = round_num
        results["EventName"] = session.event["EventName"]
        results["EventDate"] = session.event["EventDate"].strftime("%Y-%m-%d")
        results["Circuit"] = session.event["Location"]

        return results

    except Exception as e:
        print(f"Round {round_num} failed: {e}")
        return None


def load_season(year: int, max_rounds: int = 24, start_round: int = 1) -> pd.DataFrame:
    """Load available race results for a season, from start_round onwards."""
    all_results = []

    for round_num in range(start_round, max_rounds + 1):
        print(f"Loading {year} Round {round_num}...")
        df = load_race_results(year, round_num)

        if df is not None and len(df) > 0:
            all_results.append(df)
            print(f"  Loaded {len(df)} drivers (fastf1)")
        else:
            # Try Jolpica fallback
            print(f"  fastf1 returned no data, trying Jolpica...")
            df = load_race_results_jolpica(year, round_num)
            if df is not None and len(df) > 0:
                all_results.append(df)
                print(f"  Loaded {len(df)} drivers (Jolpica)")
            else:
                print(f"  No data from either source, stopping at round {round_num}")
                break

        time.sleep(0.5)

    print(f"\n  DEBUG: collected {len(all_results)} rounds before combining")

    if not all_results:
        raise RuntimeError(f"No race data loaded for {year}")

    combined = pd.concat(all_results, ignore_index=True)
    print(f"  DEBUG: combined DataFrame shape = {combined.shape}")
    return combined

def has_qualifying_times(df: Optional[pd.DataFrame]) -> bool:
    """Qualifying with no lap times at all is incomplete: sources sometimes publish the
    order before the times. Saved rounds are never re-fetched, so don't save it yet."""
    return df is not None and len(df) > 0 and df["Q1"].notna().any()


def load_qualifying_season(year: int, max_rounds: int = 24, start_round: int = 1) -> pd.DataFrame:
    """Load available qualifying results for a season, from start_round onwards."""
    all_results = []

    for round_num in range(start_round, max_rounds + 1):
        print(f"Loading {year} Round {round_num} qualifying...")
        df = load_qualifying_results(year, round_num)

        if has_qualifying_times(df):
            all_results.append(df)
            print(f"  Loaded {len(df)} drivers (fastf1)")
        else:
            print(f"  fastf1 returned no (or incomplete) qualifying data, trying Jolpica...")
            df = load_qualifying_results_jolpica(year, round_num)
            if has_qualifying_times(df):
                all_results.append(df)
                print(f"  Loaded {len(df)} drivers (Jolpica)")
            else:
                print(f"  No qualifying data, stopping at round {round_num}")
                break

        time.sleep(0.5)

    if not all_results:
        raise RuntimeError(f"No qualifying data loaded for {year}")

    combined = pd.concat(all_results, ignore_index=True)
    return combined

def save_new_rounds(year: int, load_fn, path: Path) -> None:
    """Fetch only rounds not already saved in path and append them.

    Saved rounds are never re-fetched, so their data stays stable even when a
    later fetch comes from a different source (e.g. GitHub runners can't reach
    F1 live timing and fastf1 falls back to Ergast). Delete the CSV to force a full reload.
    """
    # Read saved rows as exact text so they are written back byte-for-byte
    existing = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else None
    start = existing["Round"].astype(int).max() + 1 if existing is not None else 1
    try:
        new = load_fn(year, start_round=start)
    except RuntimeError:
        print(f"\nNo new rounds for {path.name} (have up to round {start - 1})")
        return
    df = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} rows to {path} (rounds {start}-{new['Round'].max()} new)")


def save_calendar(year: int) -> None:
    """Save the season calendar: round, event, venue, qualifying and race start (UTC).

    Lets the dashboard list races that haven't happened yet. Fetched once, like
    saved rounds; delete the file to re-fetch after a calendar change.
    """
    path = DATA_RAW_DIR / f"calendar_{year}.csv"
    if path.exists():
        return
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    def session_start(event, name):
        # Qualifying isn't always the same session number (sprint weekends differ)
        for i in range(1, 6):
            if event[f"Session{i}"] == name:
                return event[f"Session{i}DateUtc"]
        return None

    pd.DataFrame({
        "Round": schedule["RoundNumber"],
        "EventName": schedule["EventName"],
        "Location": schedule["Location"],
        "QualifyingUtc": schedule.apply(lambda e: session_start(e, "Qualifying"), axis=1),
        "RaceUtc": schedule.apply(lambda e: session_start(e, "Race"), axis=1),
        "Sprint": schedule.apply(lambda e: session_start(e, "Sprint") is not None, axis=1),
    }).to_csv(path, index=False)
    print(f"\nSaved {year} calendar ({len(schedule)} rounds) to {path}")


def save_standings(year: int) -> None:
    """Save the official drivers' and constructors' standings after the latest round.

    Taken from Jolpica rather than summed from our race results, because those
    leave out sprint points. Overwritten on every run (standings change each round).
    """
    for kind in ("driver", "constructor"):
        resp = requests.get(f"{JOLPICA_BASE}/{year}/{kind}standings.json?limit=100", timeout=15)
        resp.raise_for_status()
        lists = resp.json()["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            print(f"No {kind} standings for {year} yet")
            continue
        standings = lists[0]
        rows = []
        for s in standings[f"{kind.capitalize()}Standings"]:
            row = {
                "Round": int(standings["round"]),
                "Position": int(s.get("position", 0)),
                "Points": float(s["points"]),
                "Wins": int(s["wins"]),
            }
            if kind == "driver":
                driver = s["Driver"]
                row["Abbreviation"] = driver.get("code")
                row["FullName"] = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
                row["TeamName"] = s["Constructors"][-1]["name"]  # latest team if they switched
            else:
                row["TeamName"] = s["Constructor"]["name"]
            rows.append(row)
        path = DATA_RAW_DIR / f"standings_{year}_{kind}s.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {year} {kind} standings after round {standings['round']} to {path}")


if __name__ == "__main__":
    import sys

    # Allow CLI args: python3 src/load_season.py 2022 2023 2024 [--qualifying-only] [--races-only]
    args = sys.argv[1:]
    qualifying_only = "--qualifying-only" in args
    races_only = "--races-only" in args
    years = [int(a) for a in args if a.isdigit()]

    if not years:
        years = [2022, 2023, 2024, 2025, 2026]

    for year in years:
        save_calendar(year)
        try:
            save_standings(year)
        except Exception as e:  # standings are extra; never block the race-data update
            print(f"Could not update {year} standings: {e}")

        if not qualifying_only:
            print(f"\n{'='*50}\nLoading {year} race results\n{'='*50}")
            save_new_rounds(year, load_season, DATA_RAW_DIR / f"season_{year}_results.csv")

        if not races_only:
            print(f"\n{'='*50}\nLoading {year} qualifying results\n{'='*50}")
            save_new_rounds(year, load_qualifying_season, DATA_RAW_DIR / f"season_{year}_qualifying.csv")
