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
FEATURES_FILE = DATA_PROCESSED / "features.csv"

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
    """Load the engineered feature dataset — every row, including races not yet
    run (no finish position) and drivers with missing features."""
    return pd.read_csv(FEATURES_FILE)


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the model can learn from: a finish position and all features present.

    Drops non-starters (no position) and rows without form history. Retirements
    keep their classified position, so the model treats them as back-of-field finishes.
    """
    return df.dropna(subset=["Position", *FEATURES])


def train_model(
    train_years: List[int]
) -> Tuple[LinearRegression, StandardScaler]:
    """Train the 6-feature linear regression on the specified seasons.

    Returns the fitted model and scaler. Both are needed for prediction:
    the scaler standardises features so coefficients are interpretable.
    """
    df = training_rows(load_features())
    train_df = df[df["Year"].isin(train_years)]

    X = train_df[FEATURES]
    y = train_df["Position"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler


def predict_race(year: int, round_number: int) -> pd.DataFrame:
    """Predict finish positions for every driver in a specific race.

    The model is trained only on seasons BEFORE the one being predicted, so it
    never sees that season or any later one. For example, predicting 2024 races
    trains on 2022-2023. The earliest season has no training data and raises.

    Works for races not yet run (qualifying done): ActualPosition is then NaN
    for every driver.

    Returns a DataFrame sorted by predicted position with columns:
      Abbreviation, FullName, TeamName, QualifyingPosition, GridPosition,
      ActualPosition, PredictedPosition, PositionDelta, FormEstimated
    """
    # Train only on earlier seasons (true holdout: no future data)
    df = load_features()
    train_years = [y for y in df["Year"].unique() if y < year]
    if not train_years:
        raise ValueError(f"No seasons before {year} to train on")

    model, scaler = train_model(train_years)

    # Get the specific race
    race_df = df[(df["Year"] == year) & (df["Round"] == round_number)].copy()
    if race_df.empty:
        raise ValueError(f"No data found for {year} Round {round_number}")

    # Completed race: leave out non-starters. Upcoming race: every row has no result yet.
    if race_df["Position"].notna().any():
        race_df = race_df.dropna(subset=["Position"])

    # Every driver gets a prediction, even with missing features (debuts, new teams,
    # no qualifying time): driver form falls back to team form, anything else to the
    # race median.
    # ponytail: median is a neutral guess — a missing qualifying position lands mid-grid
    race_df["FormEstimated"] = race_df[["DriverFormLast3", "TeamFormLast3"]].isna().any(axis=1)
    race_df["DriverFormLast3"] = race_df["DriverFormLast3"].fillna(race_df["TeamFormLast3"])
    race_df[FEATURES] = race_df[FEATURES].fillna(race_df[FEATURES].median())

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
        "ConfidenceScore", "ConfidenceLevel", "FormEstimated",
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

def season_metrics(year: int) -> dict:
    """Model vs pole baseline over a season's completed races.

    Uses the same setup as the dashboard: trained only on earlier seasons.
    Returns races, rmse, pole_rmse, winner_top1 and winner_top3 (race counts).
    """
    df = training_rows(load_features())
    test = df[df["Year"] == year]
    if test.empty:
        return {"races": 0}

    model, scaler = train_model([y for y in df["Year"].unique() if y < year])
    test = test.assign(Predicted=model.predict(scaler.transform(test[FEATURES])))

    def rmse(pred):
        return float(np.sqrt(((pred - test["Position"]) ** 2).mean()))

    # Where did each race's actual winner sit in the predicted order?
    predicted_rank = test.groupby("Round")["Predicted"].rank(method="min")
    winner_rank = predicted_rank[test["Position"] == test.groupby("Round")["Position"].transform("min")]
    winner_rank = winner_rank.groupby(test["Round"]).min()

    return {
        "races": int(test["Round"].nunique()),
        "rmse": rmse(test["Predicted"]),
        "pole_rmse": rmse(test["QualifyingPosition"]),
        "winner_top1": int((winner_rank == 1).sum()),
        "winner_top3": int((winner_rank <= 3).sum()),
    }
