"""
Model training and prediction logic for the F1 Race Predictor.

This module is the bridge between the notebook-based exploration and the
Streamlit app. It encapsulates:
  - Data loading and cleaning
  - Model training on a configurable training window
  - Per-race prediction with actual results for comparison

The 6-feature linear regression is the final model selected after Day 4-7's
iterative experimentation (see notebooks/04_baseline_models.ipynb and
notebooks/05_validation_2025.ipynb for the selection rationale).
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# Project structure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = DATA_PROCESSED / "features_2022_2025.csv"

# The 6-feature set selected as the final model
FEATURES = [
    "GridPosition",
    "QualifyingPosition",
    "QualifyingGapToPole",
    "DriverFormLast3",
    "TeamFormLast3",
    "IsStreetCircuit",
]


def load_features() -> pd.DataFrame:
    """Load and clean the engineered feature dataset.

    Drops rows with missing target (DNFs/withdrawals) and missing critical
    features (early-season races without rolling form features).
    """
    df = pd.read_csv(FEATURES_FILE)
    df = df.dropna(subset=["Position"]).copy()
    df = df.dropna(subset=FEATURES).copy()
    return df


def train_model(
    train_years: List[int]
) -> Tuple[LinearRegression, StandardScaler]:
    """Train the 6-feature linear regression on the specified seasons.

    Returns the fitted model and scaler. Both are needed for prediction:
    the scaler standardises features so coefficients are interpretable.
    """
    df = load_features()
    train_df = df[df["Year"].isin(train_years)].copy()

    X = train_df[FEATURES]
    y = train_df["Position"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler


def predict_race(year: int, round_number: int) -> pd.DataFrame:
    """Predict finish positions for every driver in a specific race.

    The model is trained on all seasons EXCEPT the season being predicted.
    For example, predicting 2025 races trains on 2022-2024.

    Returns a DataFrame sorted by predicted position with columns:
      Abbreviation, FullName, TeamName, QualifyingPosition, GridPosition,
      ActualPosition, PredictedPosition, PositionDelta
    """
    # Train on all years except the one being predicted (true holdout style)
    df = load_features()
    available_years = sorted(df["Year"].unique())
    train_years = [y for y in available_years if y != year]

    model, scaler = train_model(train_years)

    # Get the specific race
    race_df = df[(df["Year"] == year) & (df["Round"] == round_number)].copy()
    if race_df.empty:
        raise ValueError(f"No data found for {year} Round {round_number}")

    # Predict
    X_race = race_df[FEATURES]
    X_race_scaled = scaler.transform(X_race)
    race_df["PredictedPosition"] = model.predict(X_race_scaled)

    # Rank predictions (lowest predicted = predicted P1)
    race_df["PredictedRank"] = race_df["PredictedPosition"].rank(method="min").astype(int)

    # Compute prediction confidence per driver
    # Confidence is based on the gap between this driver's prediction and the
    # nearest neighbours' predictions. Larger gap = more isolated = more confident.
    sorted_preds = race_df["PredictedPosition"].sort_values().values
    pred_to_confidence = {}
    for i, pred_value in enumerate(sorted_preds):
        # Gap to the prediction immediately above (lower position)
        gap_above = pred_value - sorted_preds[i - 1] if i > 0 else float("inf")
        # Gap to the prediction immediately below (higher position)
        gap_below = sorted_preds[i + 1] - pred_value if i < len(sorted_preds) - 1 else float("inf")
        # Use the smaller of the two gaps — that's the closest competitor
        nearest_gap = min(gap_above, gap_below)
        pred_to_confidence[pred_value] = nearest_gap

    race_df["NearestGap"] = race_df["PredictedPosition"].map(pred_to_confidence)

    # Normalise to 0-1 confidence score, capped at gap of 2.0 (meaning 2 places clear)
    # Then bucket into 1-5 stars for visualisation
    race_df["ConfidenceScore"] = (race_df["NearestGap"] / 2.0).clip(0, 1)
    race_df["ConfidenceLevel"] = (race_df["ConfidenceScore"] * 5).round().astype(int).clip(1, 5)

    # Build output frame
    output_cols = [
        "Abbreviation", "FullName", "TeamName",
        "QualifyingPosition", "GridPosition",
        "Position", "PredictedPosition", "PredictedRank",
        "ConfidenceScore", "ConfidenceLevel",
    ]
    output = race_df[output_cols].copy()
    output = output.rename(columns={"Position": "ActualPosition"})
    output["PositionDelta"] = output["ActualPosition"] - output["PredictedRank"]

    return output.sort_values("PredictedRank").reset_index(drop=True)


def get_race_metadata(year: int, round_number: int) -> dict:
    """Return display metadata for a specific race (event name, circuit, date)."""
    df = load_features()
    race = df[(df["Year"] == year) & (df["Round"] == round_number)]
    if race.empty:
        raise ValueError(f"No data found for {year} Round {round_number}")
    first_row = race.iloc[0]
    return {
        "year": year,
        "round": round_number,
        "event_name": first_row["EventName"],
        "circuit": first_row["Circuit"],
        "event_date": first_row["EventDate"],
    }


def get_available_races(year: int) -> pd.DataFrame:
    """Return all races for a given year as a small DataFrame."""
    df = load_features()
    races = df[df["Year"] == year][
        ["Round", "EventName", "Circuit", "EventDate"]
    ].drop_duplicates().sort_values("Round").reset_index(drop=True)
    return races