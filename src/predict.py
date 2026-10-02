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

# Win/podium chances: simulate each race this many times
N_SIMULATIONS = 10_000
# Resampled training errors are scaled by this before simulating. Unscaled, the chances
# were too flat (favourites won more often than predicted). 0.5 was chosen on 2024
# alone and checked on 2025-26: drivers given ~40% won 44% of the time, ~21% won 19%,
# and win Brier improved from 0.032 to 0.029 (notebooks/08_win_probability_calibration).
NOISE_SCALE = 0.5

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
    # Kept for simulating races (see NOISE_SCALE)
    model.residuals_ = (y - model.predict(X_scaled)).to_numpy()

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

    # Why: how far each feature pushes this driver's prediction away from the field
    # average, in places (negative = towards P1). Exact for a linear model: the pushes
    # add up to PredictedPosition - FieldAverage.
    push = pd.DataFrame(
        (X_race_scaled - X_race_scaled.mean(axis=0)) * model.coef_,
        columns=FEATURES, index=race_df.index,
    )
    race_df["FieldAverage"] = race_df["PredictedPosition"].mean()
    # Grid slot, qualifying position and gap to pole all measure one thing and move
    # together, so the model's split of credit between them is arbitrary (gap to pole
    # alone even gets a counter-intuitive sign): shown as one qualifying push
    race_df["WhyQualifying"] = push[["GridPosition", "QualifyingPosition", "QualifyingGapToPole"]].sum(axis=1)
    race_df["WhyDriverForm"] = push["DriverFormLast3"]
    race_df["WhyTeamForm"] = push["TeamFormLast3"]
    # IsStreetCircuit is the same for the whole field, so its push is always 0

    # Rank predictions (lowest predicted = predicted P1)
    race_df["PredictedRank"] = race_df["PredictedPosition"].rank(method="min").astype(int)

    # Win and podium chances: re-run the race N_SIMULATIONS times, each time adding
    # errors resampled from the model's own training residuals (which keeps the long
    # tail of retirements), and count how often each driver finishes 1st / top 3.
    rng = np.random.default_rng(0)  # fixed seed: the same race always shows the same numbers
    noise = NOISE_SCALE * rng.choice(model.residuals_, (N_SIMULATIONS, len(race_df)))
    simulated = race_df["PredictedPosition"].to_numpy() + noise
    finish = simulated.argsort(axis=1).argsort(axis=1) + 1  # finishing position in each run
    race_df["WinChance"] = (finish == 1).mean(axis=0)
    race_df["PodiumChance"] = (finish <= 3).mean(axis=0)

    # Build output frame
    output_cols = [
        "Abbreviation", "FullName", "TeamName",
        "QualifyingPosition", "GridPosition",
        "Position", "PredictedPosition", "PredictedRank",
        "WinChance", "PodiumChance", "FormEstimated",
        "QualifyingGapToPole", "DriverFormLast3", "TeamFormLast3",
        "FieldAverage", "WhyQualifying", "WhyDriverForm", "WhyTeamForm",
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


def get_calendar(year: int) -> pd.DataFrame:
    """The season calendar saved by load_season.py, or an empty frame if there isn't one.

    Columns: Round, EventName, Location, QualifyingUtc, RaceUtc (UTC timestamps).
    """
    path = PROJECT_ROOT / "data" / "raw" / f"calendar_{year}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Round", "EventName", "Location", "QualifyingUtc", "RaceUtc"])
    return pd.read_csv(path, parse_dates=["QualifyingUtc", "RaceUtc"])
