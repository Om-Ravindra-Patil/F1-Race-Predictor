"""
Track outlines for the dashboard, drawn from fastf1 car-position data.

For every circuit in the feature dataset, takes the fastest qualifying lap at its
most recent event, rotates it the way F1 broadcasts show it, simplifies it and
saves it to data/circuits.json. Circuits already saved are skipped.

Run locally from project root: python3 src/circuits.py
(GitHub's runners can't reach F1 live timing, so this can't run in the Action;
a new circuit needs one local run.)
"""

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import json
import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from fastf1 import _api

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CIRCUITS_FILE = PROJECT_ROOT / "data" / "circuits.json"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
POINTS = 160  # enough for a smooth outline, small enough to inline in the page


def lap_outline(year: int, round_number: int) -> list:
    """Fastest qualifying lap as [[x, y], ...] in 0-1 page coordinates (y down)."""
    session = fastf1.get_session(year, round_number, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    lap = session.laps.pick_fastest()

    # Position data straight from the API: fastf1 3.7 can't parse 2026 car
    # telemetry, which would otherwise fail the whole telemetry load
    pos = _api.position_data(session.api_path)[str(lap["DriverNumber"])]
    in_lap = (pos["Time"] >= lap["LapStartTime"]) & (pos["Time"] <= lap["Time"]) & (pos["Status"] == "OnTrack")
    xy = pos.loc[in_lap, ["X", "Y"]].to_numpy(float)
    if len(xy) < 50:
        raise ValueError(f"only {len(xy)} position samples in the lap")

    # Rotate to the broadcast orientation (unknown for brand-new circuits: leave as is)
    try:
        rotation = session.get_circuit_info().rotation
    except Exception:
        rotation = 0
    angle = np.radians(rotation)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = xy @ rot.T

    # Resample evenly along the track so the outline has POINTS points
    dist = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    even = np.linspace(0, dist[-1], POINTS)
    xy = np.column_stack([np.interp(even, dist, xy[:, 0]), np.interp(even, dist, xy[:, 1])])

    # Fit into a unit box keeping the aspect ratio; flip y for SVG
    xy -= xy.min(axis=0)
    xy /= xy.max()
    xy[:, 1] = xy[:, 1].max() - xy[:, 1]
    return xy.round(4).tolist()


def build_circuit_outlines() -> None:
    outlines = json.loads(CIRCUITS_FILE.read_text()) if CIRCUITS_FILE.exists() else {}

    features = pd.read_csv(FEATURES_FILE)
    # Most recent event per circuit (newest track layout)
    latest = features.sort_values(["Year", "Round"]).groupby("Circuit")[["Year", "Round"]].last()

    for circuit, (year, round_number) in latest.iterrows():
        if circuit in outlines:
            continue
        try:
            outlines[circuit] = lap_outline(int(year), int(round_number))
            print(f"  {circuit}: {year} R{round_number} ok")
        except Exception as e:
            print(f"  {circuit}: {year} R{round_number} failed ({e})")

    CIRCUITS_FILE.write_text(json.dumps(outlines, separators=(",", ":"), sort_keys=True))
    print(f"\nSaved {len(outlines)} circuit outlines to {CIRCUITS_FILE}")


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    fastf1.Cache.enable_cache(str(PROJECT_ROOT / "cache"))
    build_circuit_outlines()
