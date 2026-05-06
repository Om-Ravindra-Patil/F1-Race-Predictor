"""
Feature engineering for F1 race winner prediction.

Loads race + qualifying data, computes features, saves modelling-ready dataset.
Run from project root: python3 src/features.py
"""

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def parse_qualifying_time(t) -> Optional[float]:
    """Convert qualifying time to seconds (float).

    Handles three formats:
    - fastf1 Timedelta string: '0 days 00:01:30.031000'
    - Jolpica string format: '1:30.031'
    - Already-numeric / NaN
    """
    if pd.isna(t) or t == "" or t is None:
        return None

    s = str(t).strip()

    # fastf1 format: "0 days 00:01:30.031000"
    if "days" in s:
        try:
            td = pd.to_timedelta(s)
            return td.total_seconds()
        except Exception:
            return None

    # Jolpica format: "1:30.031" or "1:30.5"
    if ":" in s:
        try:
            mins, secs = s.split(":")
            return int(mins) * 60 + float(secs)
        except Exception:
            return None

    # Already numeric
    try:
        return float(s)
    except Exception:
        return None


def best_qualifying_time(row) -> Optional[float]:
    """Return the driver's best qualifying time in seconds.

    Takes the minimum of Q1/Q2/Q3 lap times. F1 ranking uses each session's
    fastest lap as the qualifying-determining time, so we want the lowest
    valid value across whichever sessions the driver participated in.
    """
    times = [parse_qualifying_time(row.get(c)) for c in ["Q1", "Q2", "Q3"]]
    valid = [t for t in times if t is not None]
    return min(valid) if valid else None

def load_combined_data() -> pd.DataFrame:
    """Load all seasons' race + qualifying data and merge into one DataFrame."""
    race_frames = []
    qual_frames = []

    for year in [2022, 2023, 2024]:
        race_df = pd.read_csv(DATA_RAW / f"season_{year}_results.csv")
        qual_df = pd.read_csv(DATA_RAW / f"season_{year}_qualifying.csv")
        race_frames.append(race_df)
        qual_frames.append(qual_df)

    races = pd.concat(race_frames, ignore_index=True)
    quals = pd.concat(qual_frames, ignore_index=True)

    # Compute best qualifying time per driver per round
    quals["BestQualiTime"] = quals.apply(best_qualifying_time, axis=1)

    # Keep only the columns we need from qualifying — avoids column name clashes on merge
    quals_slim = quals[["Year", "Round", "Abbreviation", "BestQualiTime", "Position"]].copy()
    quals_slim = quals_slim.rename(columns={"Position": "QualifyingPosition"})

    # Merge race + qualifying on (Year, Round, Driver)
    merged = races.merge(quals_slim, on=["Year", "Round", "Abbreviation"], how="left")

    # Coerce numerics
    for col in ["Position", "GridPosition", "Points", "Laps", "QualifyingPosition"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["Points"] = merged["Points"].fillna(0)

    # Normalise team names (handles Jolpica vs fastf1 inconsistencies)
    team_name_map = {
        "Red Bull": "Red Bull Racing",
        "Alpine F1 Team": "Alpine",
        "RB F1 Team": "RB",
        "Haas F1 Team": "Haas",
    }
    merged["TeamName"] = merged["TeamName"].replace(team_name_map)

    # Sort chronologically — critical for rolling features
    merged["EventDate"] = pd.to_datetime(merged["EventDate"])
    merged = merged.sort_values(["EventDate", "Position"]).reset_index(drop=True)

    return merged


def add_qualifying_gap(df: pd.DataFrame, max_gap: float = 10.0) -> pd.DataFrame:
    """Add 'QualifyingGapToPole' — gap (seconds) between driver's qualifying time and pole-sitter's.

    Pole time is the actual P1 driver's lap. Outlier gaps (wet sessions, data corruption)
    are clipped at max_gap seconds — beyond that, magnitude provides no extra signal,
    just noise that distorts model training.
    """
    pole_rows = df[df["QualifyingPosition"] == 1][
        ["Year", "Round", "BestQualiTime"]
    ].rename(columns={"BestQualiTime": "PoleTime"})

    df = df.drop(columns=["PoleTime"], errors="ignore")
    df = df.merge(pole_rows, on=["Year", "Round"], how="left")

    raw_gap = df["BestQualiTime"] - df["PoleTime"]
    df["QualifyingGapToPole"] = raw_gap.clip(lower=0, upper=max_gap)

    return df


def add_rolling_form(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add rolling average finish position over last N races for driver and team.

    Critical: uses .shift(1) to ensure we only look at PAST races, not the current one.
    Without shift, we'd be leaking the current race's result into its own feature.
    """
    df = df.sort_values(["Abbreviation", "EventDate"]).reset_index(drop=True)

    # Driver form: rolling mean of finish position, excluding current race
    df["DriverFormLast3"] = (
        df.groupby("Abbreviation")["Position"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Team form: same logic but at constructor level
    df = df.sort_values(["TeamName", "EventDate"]).reset_index(drop=True)
    df["TeamFormLast3"] = (
        df.groupby("TeamName")["Position"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    return df

def add_driver_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver's historical average finish position at this specific circuit.

    For each (driver, circuit) pair, computes the mean of all PRIOR finishes
    at that circuit. Uses .shift(1) and expanding mean to avoid leakage.

    Captures circuit-specialist effects that aren't visible in general form
    features (e.g., VER at Suzuka, ALO at Monaco, HAM at Silverstone).
    """
    df = df.sort_values(["Abbreviation", "Circuit", "EventDate"]).reset_index(drop=True)

    df["DriverCircuitAvg"] = (
        df.groupby(["Abbreviation", "Circuit"])["Position"]
          .transform(lambda x: x.shift(1).expanding().mean())
    )

    return df

def add_team_momentum(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add team's recent performance trend (slope of last N race finishes).

    A negative slope = team is improving (finishing positions getting lower).
    Positive slope = team is declining.

    Captures things like McLaren's 2024 rise or Williams' 2023 fall — which
    rolling averages would miss because they only show the level, not direction.
    """
    df = df.sort_values(["TeamName", "EventDate"]).reset_index(drop=True)

    def slope_of_window(series):
        """Simple linear regression slope. Returns NaN if too few points."""
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        # Standard least-squares slope formula
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return ((x - x_mean) * (y - y_mean)).sum() / denom

    df["TeamMomentum"] = (
        df.groupby("TeamName")["Position"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=2).apply(slope_of_window, raw=False))
    )

    return df

def add_quali_gap_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Add the driver's qualifying gap to pole, standardised within their race.

    A 0.3s gap is huge at Monaco (tight quali) but small at Spa (long lap).
    Z-scoring within race makes "1.0 = one std-dev slower than average for this race"
    — which is more meaningful than raw seconds.
    """
    grouped = df.groupby(["Year", "Round"])["QualifyingGapToPole"]
    df["QualiGapZScore"] = (
        df["QualifyingGapToPole"] - grouped.transform("mean")
    ) / grouped.transform("std").replace(0, np.nan)

    return df


def add_dnf_rate(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling DNF rate over last N races per driver."""
    df = df.sort_values(["Abbreviation", "EventDate"]).reset_index(drop=True)
    df["IsDNF"] = df["Position"].isna().astype(int)

    df["DriverDNFRateLast5"] = (
        df.groupby("Abbreviation")["IsDNF"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    return df


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tag street circuits — they have very different race dynamics (low overtaking, safety cars)."""
    street_circuits = {
        "Monte Carlo", "Monaco",
        "Singapore",
        "Baku",
        "Las Vegas",
        "Jeddah",
        "Miami",
    }
    df["IsStreetCircuit"] = df["Circuit"].isin(street_circuits).astype(int)
    return df


def build_feature_dataset() -> pd.DataFrame:
    """End-to-end feature build."""
    print("Loading combined race + qualifying data...")
    df = load_combined_data()
    print(f"  Loaded {len(df)} rows")

    print("Computing qualifying gap to pole...")
    df = add_qualifying_gap(df)

    print("Computing rolling driver and team form...")
    df = add_rolling_form(df)

    print("Computing driver-circuit history...")
    df = add_driver_circuit_history(df)

    print("Computing team momentum...")
    df = add_team_momentum(df)

    print("Computing standardised qualifying gap...")
    df = add_quali_gap_zscore(df)

    print("Computing DNF rates...")
    df = add_dnf_rate(df)

    print("Tagging circuit features...")
    df = add_circuit_features(df)

    feature_cols = [
        "Year", "Round", "EventName", "EventDate", "Circuit",
        "Abbreviation", "FullName", "TeamName",
        "GridPosition", "QualifyingPosition", "Position", "Points", "Status",
        "BestQualiTime", "PoleTime", "QualifyingGapToPole", "QualiGapZScore",
        "DriverFormLast3", "TeamFormLast3", "DriverDNFRateLast5",
        "DriverCircuitAvg", "TeamMomentum",
        "IsStreetCircuit",
    ]
    df = df[feature_cols].copy()

    return df


if __name__ == "__main__":
    df = build_feature_dataset()

    output_path = DATA_PROCESSED / "features_2022_2024.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"   Seasons: {sorted(df['Year'].unique())}")
    print(f"   Drivers: {df['Abbreviation'].nunique()}")
    print(f"   Features: {[c for c in df.columns if c not in ['Year','Round','EventName','EventDate','Circuit','Abbreviation','FullName','TeamName','Position','Points','Status']]}")