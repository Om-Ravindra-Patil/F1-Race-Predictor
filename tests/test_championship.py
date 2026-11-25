import pandas as pd
import pytest

from src.championship import load_standings, points_available, simulate_championship, title_status


def events(races, sprints):
    return pd.DataFrame({"Sprint": [True] * sprints + [False] * (races - sprints)})


def test_points_available_includes_sprints():
    assert points_available(events(2, 0), "drivers") == 50
    assert points_available(events(2, 1), "drivers") == 58  # + 8 for a sprint win
    assert points_available(events(2, 1), "constructors") == 86 + 15  # P1+P2 in both races and the sprint


def test_title_status():
    standings = pd.DataFrame({"Points": [402, 304, 267]})
    assert title_status(standings, 50).tolist() == ["Champion", "Out", "Out"]
    # A lead equal to the points left isn't decided (countback could still beat it)
    assert title_status(pd.DataFrame({"Points": [100, 50]}), 50).tolist() == ["In contention", "In contention"]
    assert title_status(pd.DataFrame({"Points": [100, 60, 40]}), 50).tolist() == ["In contention", "In contention", "Out"]


def test_standings_files_are_consistent():
    for year in range(2022, 2027):
        for kind in ("drivers", "constructors"):
            df = load_standings(year, kind)
            assert not df.empty, (year, kind)
            assert df["Points"].is_monotonic_decreasing, (year, kind)
            assert df["Round"].nunique() == 1


def test_simulation_probabilities():
    result = simulate_championship(2026, n=2000)
    for kind in ("drivers", "constructors"):
        df = result[kind]
        positions = list(range(1, len(df) + 1))
        assert df[positions].sum(axis=1).tolist() == pytest.approx([1] * len(df))  # each entrant ends somewhere
        assert df[positions].sum(axis=0).tolist() == pytest.approx([1] * len(df))  # each position filled once
        champions = result[f"{kind}_status"] == "Champion"
        assert (df.loc[champions, 1] == 1).all()  # a clinched title is certain in every simulation
