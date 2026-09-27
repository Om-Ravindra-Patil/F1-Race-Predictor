import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_circuit_features,
    add_qualifying_gap,
    add_rolling_form,
    best_qualifying_time,
    parse_qualifying_time,
)
from src.predict import FEATURES_FILE, predict_race


# --- Qualifying time parsing ---

@pytest.mark.parametrize("raw, expected", [
    ("0 days 00:01:30.031000", 90.031),  # fastf1
    ("1:30.031", 90.031),                # Jolpica
    ("90.5", 90.5),                      # numeric string
    (90.5, 90.5),
])
def test_parse_qualifying_time_formats(raw, expected):
    assert parse_qualifying_time(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, np.nan, "", "garbage", "1:xx"])
def test_parse_qualifying_time_invalid(raw):
    assert parse_qualifying_time(raw) is None


def test_best_qualifying_time_takes_fastest_session():
    row = {"Q1": "1:31.000", "Q2": "1:30.000", "Q3": np.nan}
    assert best_qualifying_time(row) == pytest.approx(90.0)


def test_best_qualifying_time_no_valid_times():
    assert best_qualifying_time({"Q1": np.nan, "Q2": "", "Q3": None}) is None


# --- Qualifying gap to pole ---

def test_add_qualifying_gap():
    df = pd.DataFrame({
        "Year": [2025] * 3,
        "Round": [1] * 3,
        "QualifyingPosition": [1, 2, 3],
        "BestQualiTime": [90.0, 90.5, 105.0],
    })
    out = add_qualifying_gap(df)
    assert len(out) == 3  # pole merge must not duplicate rows
    assert out["QualifyingGapToPole"].tolist() == pytest.approx([0.0, 0.5, 10.0])  # clipped at 10s


# --- Rolling form (leak prevention) ---

def test_driver_form_uses_only_past_races():
    df = pd.DataFrame({
        "Abbreviation": ["AAA"] * 5,
        "TeamName": ["Ferrari"] * 5,
        "EventDate": pd.date_range("2025-03-01", periods=5, freq="7D"),
        "Position": [1, 3, 5, np.nan, 7],  # race 4 is a DNF
    })
    out = add_rolling_form(df).sort_values("EventDate")
    # Race 1 has no history; each later race averages up to 3 prior finishes, DNFs skipped
    assert out["DriverFormLast3"].tolist() == pytest.approx([np.nan, 1, 2, 3, 4], nan_ok=True)


def test_team_form_carries_across_rebrand():
    df = pd.DataFrame({
        "Abbreviation": ["AAA", "AAA"],
        "TeamName": ["AlphaTauri", "RB"],
        "EventDate": pd.to_datetime(["2023-11-26", "2024-03-02"]),
        "Position": [10, 12],
    })
    out = add_rolling_form(df).sort_values("EventDate")
    assert out["TeamFormLast3"].iloc[1] == 10


# --- Street circuits ---

def test_street_circuit_flag():
    df = pd.DataFrame({"Circuit": ["Marina Bay", "Monaco", "Miami", "Monza", "Silverstone"]})
    assert add_circuit_features(df)["IsStreetCircuit"].tolist() == [1, 1, 1, 0, 0]


# --- Committed feature dataset (guards against regressions of past data bugs) ---

@pytest.fixture(scope="module")
def features():
    return pd.read_csv(FEATURES_FILE)


def test_dataset_has_no_pit_lane_grid_zero(features):
    assert (features["GridPosition"] != 0).all()


def test_dataset_street_circuits(features):
    street = set(features.loc[features["IsStreetCircuit"] == 1, "Circuit"])
    assert street == {"Monaco", "Marina Bay", "Baku", "Las Vegas", "Jeddah", "Miami"}
    assert "Miami Gardens" not in set(features["Circuit"])


def test_dataset_one_row_per_driver_per_race(features):
    assert not features.duplicated(["Year", "Round", "Abbreviation"]).any()


# --- Race prediction ---

def test_predict_race_output():
    out = predict_race(2025, 1)
    assert out["PredictedRank"].is_monotonic_increasing
    assert out["PredictedRank"].iloc[0] == 1
    assert out["ConfidenceLevel"].between(1, 5).all()
    assert out["Abbreviation"].is_unique


def test_predict_race_unknown_round():
    with pytest.raises(ValueError):
        predict_race(2025, 99)
