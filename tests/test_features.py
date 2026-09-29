import numpy as np
import pandas as pd
import pytest

from src.features import (
    CIRCUIT_ALIASES,
    add_circuit_features,
    add_dnf_rate,
    add_qualifying_gap,
    add_rolling_form,
    best_qualifying_time,
    parse_qualifying_time,
)
from src.predict import FEATURES_FILE, predict_race, season_metrics


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


def test_team_form_excludes_teammate_same_race():
    df = pd.DataFrame({
        "Abbreviation": ["AAA", "BBB", "AAA", "BBB"],
        "TeamName": ["Ferrari"] * 4,
        "EventDate": pd.to_datetime(["2025-03-01", "2025-03-01", "2025-03-08", "2025-03-08"]),
        "Position": [1, 20, 3, 4],
    })
    out = add_rolling_form(df).sort_values(["EventDate", "Abbreviation"])
    # Race 1: no history for either driver; race 2: both see race 1's team mean (1+20)/2
    assert out["TeamFormLast3"].tolist() == pytest.approx([np.nan, np.nan, 10.5, 10.5], nan_ok=True)


def test_team_form_carries_across_rebrand():
    df = pd.DataFrame({
        "Abbreviation": ["AAA", "AAA"],
        "TeamName": ["AlphaTauri", "RB"],
        "EventDate": pd.to_datetime(["2023-11-26", "2024-03-02"]),
        "Position": [10, 12],
    })
    out = add_rolling_form(df).sort_values("EventDate")
    assert out["TeamFormLast3"].iloc[1] == 10


# --- DNF rate ---

def test_dnf_rate_counts_retirements_with_a_position():
    df = pd.DataFrame({
        "Abbreviation": ["AAA"] * 4,
        "EventDate": pd.date_range("2025-03-01", periods=4, freq="7D"),
        "ClassifiedPosition": ["1", "R", "W", np.nan],  # last race not yet run
    })
    out = add_dnf_rate(df).sort_values("EventDate")
    assert out["DriverDNFRateLast5"].tolist() == pytest.approx([np.nan, 0, 0.5, 2 / 3], nan_ok=True)


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
    assert street == {"Monaco", "Marina Bay", "Baku", "Las Vegas", "Jeddah", "Miami", "Madrid"}
    assert not set(CIRCUIT_ALIASES) & set(features["Circuit"])  # every alias normalised


def test_dataset_every_race_has_grid_positions(features):
    # A whole race with no grid data means a partial fastf1 load slipped through
    missing = features["GridPosition"].isna().groupby([features["Year"], features["Round"]]).mean()
    assert (missing < 0.5).all(), missing[missing >= 0.5]


def test_dataset_teammates_share_team_form(features):
    per_team_race = features.groupby(["Year", "Round", "TeamName"])["TeamFormLast3"].nunique()
    assert (per_team_race <= 1).all()


def test_dataset_one_row_per_driver_per_race(features):
    assert not features.duplicated(["Year", "Round", "Abbreviation"]).any()


# --- Race prediction ---

def test_predict_race_output():
    out = predict_race(2025, 1)
    assert out["PredictedRank"].is_monotonic_increasing
    assert out["PredictedRank"].iloc[0] == 1
    assert out["ConfidenceLevel"].between(1, 5).all()
    assert out["Abbreviation"].is_unique


def test_predict_race_trains_only_on_earlier_seasons(monkeypatch):
    import src.predict as predict
    seen = []
    real_train = predict.train_model

    def spy(years):
        seen.extend(years)
        return real_train(years)

    monkeypatch.setattr(predict, "train_model", spy)
    predict.predict_race(2023, 1)
    assert seen == [2022]


def test_predict_race_first_season_has_no_training_data():
    with pytest.raises(ValueError):
        predict_race(2022, 5)


def test_predict_upcoming_race(monkeypatch):
    # Simulate a race that has qualified but not run: no finish positions yet
    import src.predict as predict
    df = pd.read_csv(FEATURES_FILE)
    upcoming = (df["Year"] == 2025) & (df["Round"] == 24)
    df.loc[upcoming, "Position"] = np.nan
    monkeypatch.setattr(predict, "load_features", lambda: df.copy())

    out = predict.predict_race(2025, 24)
    assert len(out) == upcoming.sum()  # whole grid predicted
    assert out["ActualPosition"].isna().all()
    assert sorted(out["PredictedRank"]) == list(range(1, len(out) + 1))


def test_predict_race_includes_drivers_without_form_history():
    # 2026 R1: Cadillac's debut and a rookie have no form yet but must still be predicted
    out = predict_race(2026, 1)
    assert len(out) == 22
    assert out[["PredictedPosition"]].notna().all().all()
    assert {"BOT", "PER", "LIN"} <= set(out.loc[out["FormEstimated"], "Abbreviation"])


def test_season_metrics_match_published_2025_holdout():
    # The README headline: RMSE 4.25 vs 4.69 pole, winner in top 3 in 22/24, exact in 12/24
    m = season_metrics(2025)
    assert m["races"] == 24
    assert m["rmse"] == pytest.approx(4.248, abs=1e-3)
    assert m["pole_rmse"] == pytest.approx(4.686, abs=1e-3)
    assert (m["winner_top1"], m["winner_top3"]) == (12, 22)


def test_prediction_breakdown_adds_up():
    # The "why" pushes must explain each prediction exactly
    out = predict_race(2026, 15)
    pushes = out[["WhyQualifying", "WhyDriverForm", "WhyTeamForm"]].sum(axis=1)
    assert (out["FieldAverage"] + pushes).tolist() == pytest.approx(out["PredictedPosition"].tolist())


def test_predict_race_unknown_round():
    with pytest.raises(ValueError):
        predict_race(2025, 99)


# --- Incremental data loading ---

def test_save_new_rounds_appends_without_touching_saved_rows(tmp_path):
    from src.load_season import save_new_rounds
    path = tmp_path / "season.csv"
    saved = "Round,Abbreviation,HeadshotUrl\n" + "".join(f"{r},AAA,None\n" for r in range(1, 11))
    path.write_text(saved)

    calls = []
    def fake_load(year, start_round):
        calls.append(start_round)
        return pd.DataFrame({"Round": [start_round], "Abbreviation": ["AAA"], "HeadshotUrl": ["x"]})

    save_new_rounds(2026, fake_load, path)
    assert calls == [11]  # numeric max, not text ("9" > "10")
    assert path.read_text() == saved + "11,AAA,x\n"  # saved rows byte-identical


def test_save_new_rounds_nothing_new(tmp_path):
    from src.load_season import save_new_rounds
    path = tmp_path / "season.csv"
    path.write_text("Round,Abbreviation\n1,AAA\n")

    def no_data(year, start_round):
        raise RuntimeError("No race data loaded")

    save_new_rounds(2026, no_data, path)
    assert path.read_text() == "Round,Abbreviation\n1,AAA\n"
