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
                "TeamName": r["Constructor"].get("name"),
                "Position": r.get("position"),  # qualifying position
                "Q1": r.get("Q1"),
                "Q2": r.get("Q2"),
                "Q3": r.get("Q3"),
                "Year": year,
                "Round": round_num,
                "EventName": race.get("raceName"),
                "EventDate": race.get("date"),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"    Jolpica qualifying fallback failed: {e}")
        return None


def load_qualifying_results(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Load qualifying results for a single round. Returns None if unavailable."""
    try:
        session = fastf1.get_session(year, round_num, "Q")  # 'Q' = Qualifying
        session.load(telemetry=False, weather=False, messages=False)

        results = session.results.copy()
        # Add metadata columns
        results["Year"] = year
        results["Round"] = round_num
        results["EventName"] = session.event["EventName"]
        results["EventDate"] = session.event["EventDate"]

        return results

    except Exception as e:
        print(f"  Round {round_num} qualifying failed: {e}")
        return None


def load_race_results(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Load race results for a single round. Returns None if unavailable."""
    try:
        session = fastf1.get_session(year, round_num, "R")  # 'R' = Race
        session.load(telemetry=False, weather=False, messages=False)

        results = session.results.copy()
        # Add metadata columns
        results["Year"] = year
        results["Round"] = round_num
        results["EventName"] = session.event["EventName"]
        results["EventDate"] = session.event["EventDate"]
        results["Circuit"] = session.event["Location"]

        return results

    except Exception as e:
        print(f"Round {round_num} failed: {e}")
        return None


def load_season(year: int, max_rounds: int = 24) -> pd.DataFrame:
    """Load all available race results for a season."""
    all_results = []

    for round_num in range(1, max_rounds + 1):
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

def load_qualifying_season(year: int, max_rounds: int = 24) -> pd.DataFrame:
    """Load all available qualifying results for a season."""
    all_results = []

    for round_num in range(1, max_rounds + 1):
        print(f"Loading {year} Round {round_num} qualifying...")
        df = load_qualifying_results(year, round_num)

        if df is not None and len(df) > 0:
            all_results.append(df)
            print(f"  Loaded {len(df)} drivers (fastf1)")
        else:
            print(f"  fastf1 returned no qualifying data, trying Jolpica...")
            df = load_qualifying_results_jolpica(year, round_num)
            if df is not None and len(df) > 0:
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

if __name__ == "__main__":
    import sys

    # Allow CLI args: python3 src/load_season.py 2022 2023 2024 [--qualifying-only] [--races-only]
    args = sys.argv[1:]
    qualifying_only = "--qualifying-only" in args
    races_only = "--races-only" in args
    years = [int(a) for a in args if a.isdigit()]

    if not years:
        years = [2024]

    for year in years:
        if not qualifying_only:
            print(f"\n{'='*50}\nLoading {year} race results\n{'='*50}")
            df = load_season(year)
            output_path = DATA_RAW_DIR / f"season_{year}_results.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved {len(df)} race rows to {output_path}")
            print(f"   Rounds: {df['Round'].nunique()}, Drivers: {df['Abbreviation'].nunique()}")

        if not races_only:
            print(f"\n{'='*50}\nLoading {year} qualifying results\n{'='*50}")
            df = load_qualifying_season(year)
            output_path = DATA_RAW_DIR / f"season_{year}_qualifying.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved {len(df)} qualifying rows to {output_path}")
            print(f"   Rounds: {df['Round'].nunique()}, Drivers: {df['Abbreviation'].nunique()}")
    
