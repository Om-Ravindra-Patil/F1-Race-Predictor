from streamlit.testing.v1 import AppTest


def selected(at):
    season, race = at.sidebar.selectbox
    return season.value, race.value[:3], dict(at.query_params)


def test_shareable_race_links():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.query_params.update(season="2024", round="11")
    at.run()
    assert not at.exception
    assert selected(at) == (2024, "R11", {"season": ["2024"], "round": ["11"]})  # link opens the race

    at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[4]).run()
    assert selected(at) == (2024, "R05", {"season": ["2024"], "round": ["5"]})  # picking a race updates the URL

    at.sidebar.selectbox[0].set_value(2025).run()
    assert selected(at)[:2] == (2025, "R24")  # new season opens its latest race

    at.query_params.update(season="2023", round="3")  # a new link pasted into the same tab
    at.run()
    assert selected(at)[:2] == (2023, "R03")


def test_bad_link_falls_back_to_latest_race():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.query_params.update(season="1999", round="abc")
    at.run()
    assert not at.exception
    season, race, _ = selected(at)
    assert season == int(at.sidebar.selectbox[0].options[0])  # newest season
    # Latest race with data, not a future one (future rounds are labelled with their date)
    races_with_data = [o for o in at.sidebar.selectbox[1].options if " · " not in o]
    assert race == races_with_data[-1][:3]


def test_future_race_shows_when_prediction_appears():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.query_params.update(season="2026", round="23")  # Abu Dhabi: on the calendar, not yet run
    at.run()
    assert not at.exception
    assert selected(at)[:2] == (2026, "R23")
    page = " ".join(m.value for m in at.markdown)
    assert "Prediction not available yet" in page
    assert "after qualifying on" in page
