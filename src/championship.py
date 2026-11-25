"""
Drivers' and constructors' championship: official standings, exact title maths,
and a simulation of the remaining races for each final position's chances.

Standings come from data/raw/standings_<year>_*.csv (saved by load_season.py from
Jolpica, so they include sprint points our race results don't have).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features import CIRCUIT_ALIASES, TEAM_NAMES, add_circuit_features
from src.predict import (
    N_SIMULATIONS, NOISE_SCALE, get_calendar, load_features, predict_race,
    train_model, training_rows,
)

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

RACE_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1])
FORM_FEATURES = ["DriverFormLast3", "TeamFormLast3", "IsStreetCircuit"]


def load_standings(year: int, kind: str) -> pd.DataFrame:
    """Official standings ("drivers" or "constructors"), or an empty frame if none saved."""
    path = DATA_RAW / f"standings_{year}_{kind}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["TeamName"] = df["TeamName"].replace(TEAM_NAMES)
    return df


def remaining_events(year: int, after_round: int) -> pd.DataFrame:
    """Calendar rounds still to run after the standings' round."""
    calendar = get_calendar(year)
    if "Sprint" not in calendar:
        calendar["Sprint"] = False
    return calendar[calendar["Round"] > after_round].reset_index(drop=True)


def points_available(events: pd.DataFrame, kind: str) -> float:
    """Most points one driver (or one team, with both cars) can still score."""
    races, sprints = len(events), int(events["Sprint"].sum())
    if kind == "drivers":
        return races * RACE_POINTS[0] + sprints * SPRINT_POINTS[0]
    return races * RACE_POINTS[:2].sum() + sprints * SPRINT_POINTS[:2].sum()


def title_status(standings: pd.DataFrame, available: float) -> pd.Series:
    """'Champion' once the leader can't be caught, 'In contention' or 'Out' for the rest.

    A tie on points is decided by countback, so a lead equal to the points left
    is not treated as decided.
    """
    leader = standings["Points"].max()
    second = standings["Points"].nlargest(2).iloc[-1]
    if leader - second > available:
        return pd.Series(np.where(standings["Points"] == leader, "Champion", "Out"), index=standings.index)
    return pd.Series(
        np.where(standings["Points"] + available >= leader, "In contention", "Out"), index=standings.index
    )


def _expected_positions(year: int, event: pd.Series, drivers: list, features: pd.DataFrame):
    """Expected finishing position per driver for one remaining event, plus the
    errors to resample when simulating it.

    With qualifying done, the race model is used. Further ahead there is no
    qualifying, so a weaker form-only model stands in (unscaled errors, so its
    chances stay wide).
    """
    race_rows = features[(features["Year"] == year) & (features["Round"] == event["Round"])]
    if not race_rows.empty:  # qualifying is in: full model
        pred = predict_race(year, int(event["Round"])).set_index("Abbreviation")["PredictedPosition"]
        model, _ = train_model([y for y in features["Year"].unique() if y < year])
        return pred.reindex(drivers), NOISE_SCALE * model.residuals_

    # Form-only model, trained on earlier seasons
    rows = training_rows(features)
    rows = rows[rows["Year"] < year].dropna(subset=FORM_FEATURES)
    model = LinearRegression().fit(rows[FORM_FEATURES], rows["Position"])
    residuals = (rows["Position"] - model.predict(rows[FORM_FEATURES])).to_numpy()

    # Current form: average finish over each driver's / team's last 3 races
    done = features[(features["Year"] == year) & features["Position"].notna()].sort_values("Round")
    driver_form = done.groupby("Abbreviation")["Position"].apply(lambda s: s.tail(3).mean())
    team_form = (done.groupby(["TeamName", "Round"])["Position"].mean()
                 .groupby(level="TeamName").apply(lambda s: s.tail(3).mean()))
    latest_team = done.groupby("Abbreviation")["TeamName"].last()
    street = add_circuit_features(pd.DataFrame({"Circuit": [CIRCUIT_ALIASES.get(event["Location"], event["Location"])]}))
    X = pd.DataFrame({
        "DriverFormLast3": driver_form.reindex(drivers).to_numpy(),
        "TeamFormLast3": team_form.reindex(latest_team.reindex(drivers)).to_numpy(),
        "IsStreetCircuit": street["IsStreetCircuit"].iloc[0],
    })
    X = X.fillna(X.median())  # ponytail: a debut driver gets the field median
    return pd.Series(model.predict(X), index=drivers), residuals


def simulate_championship(year: int, n: int = N_SIMULATIONS, seed: int = 0) -> dict:
    """Chance of each final championship position, for drivers and constructors.

    Returns {"drivers": DataFrame, "constructors": DataFrame}, one row per entrant
    (in current standings order) with columns 1..N giving the probability of
    finishing in that position, plus the round the standings are from and the
    events still to run. Empty dict if there are no standings.
    """
    drivers_now = load_standings(year, "drivers")
    teams_now = load_standings(year, "constructors")
    if drivers_now.empty or teams_now.empty:
        return {}
    after = int(drivers_now["Round"].iloc[0])
    events = remaining_events(year, after)
    rng = np.random.default_rng(seed)

    # Current roster: drivers who started the latest race (others' points are final)
    features = load_features()
    latest = features[(features["Year"] == year) & features["Position"].notna()]
    latest = latest[latest["Round"] == latest["Round"].max()]
    roster = [d for d in drivers_now["Abbreviation"] if d in set(latest["Abbreviation"])]
    team_of = latest.set_index("Abbreviation")["TeamName"]

    driver_pts = np.tile(drivers_now["Points"].to_numpy(float), (n, 1))
    roster_cols = [drivers_now.index[drivers_now["Abbreviation"] == d][0] for d in roster]
    for _, event in events.iterrows():
        expected, residuals = _expected_positions(year, event, roster, features)
        sessions = [RACE_POINTS] + ([SPRINT_POINTS] if event["Sprint"] else [])
        for table in sessions:
            noisy = expected.to_numpy() + rng.choice(residuals, (n, len(roster)))
            finish = noisy.argsort(axis=1).argsort(axis=1)  # 0 = winner
            scored = np.where(finish < len(table), table[np.minimum(finish, len(table) - 1)], 0)
            driver_pts[:, roster_cols] += scored

    # Constructors: current points plus what their roster drivers add
    added = driver_pts - drivers_now["Points"].to_numpy(float)
    team_pts = np.tile(teams_now["Points"].to_numpy(float), (n, 1))
    for j, team in enumerate(teams_now["TeamName"]):
        cols = [c for d, c in zip(roster, roster_cols) if team_of.get(d) == team]
        team_pts[:, j] += added[:, cols].sum(axis=1)

    def position_chances(points: np.ndarray, standings: pd.DataFrame) -> pd.DataFrame:
        # Ties broken by current position (stands in for countback)
        order = points - standings["Position"].to_numpy() * 1e-6
        final = (-order).argsort(axis=1).argsort(axis=1) + 1
        k = points.shape[1]
        probs = np.stack([(final == p).mean(axis=0) for p in range(1, k + 1)], axis=1)
        out = standings.copy()
        out[list(range(1, k + 1))] = probs
        out["ExpectedPoints"] = points.mean(axis=0)
        return out

    return {
        "after_round": after,
        "events": events,
        "drivers": position_chances(driver_pts, drivers_now),
        "constructors": position_chances(team_pts, teams_now),
        "drivers_status": title_status(drivers_now, points_available(events, "drivers")),
        "constructors_status": title_status(teams_now, points_available(events, "constructors")),
    }


if __name__ == "__main__":
    result = simulate_championship(2026)
    print(f"After round {result['after_round']}, remaining: {result['events']['EventName'].tolist()}")
    for kind in ("drivers", "constructors"):
        df = result[kind]
        name = "Abbreviation" if kind == "drivers" else "TeamName"
        df["Status"] = result[f"{kind}_status"]
        print(df[[name, "Points", "Status", "ExpectedPoints", 1, 2, 3, 4, 5]].head(8).round(3).to_string(index=False))
