"""
Load F1 race results for a given season using fastf1.
Saves consolidated results to data/raw/season_<year>_results.csv
"""

import fastf1
import pandas as pd
from pathlib import Path
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


from typing import Optional

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

        if df is not None:
            all_results.append(df)
            print(f"  ✓ Loaded {len(df)} drivers")
        else:
            # Likely a round that hasn't happened yet — stop the loop
            print(f"  → No data, stopping at round {round_num}")
            break

        # Be polite to the API even with cache (first run only)
        time.sleep(0.5)

    if not all_results:
        raise RuntimeError(f"No race data loaded for {year}")

    combined = pd.concat(all_results, ignore_index=True)
    return combined


if __name__ == "__main__":
    YEAR = 2024

    print(f"Loading {YEAR} season race results...\n")
    df = load_season(YEAR)

    output_path = DATA_RAW_DIR / f"season_{YEAR}_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"   Rounds: {df['Round'].nunique()}")
    print(f"   Drivers: {df['Abbreviation'].nunique()}")
    print(f"\nColumns: {list(df.columns)}")