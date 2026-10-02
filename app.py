"""
F1 Race Winner Predictor — Streamlit Dashboard

F1 broadcast-style interactive dashboard showcasing the validated prediction model.
Trained on earlier seasons only; beats the pole baseline on RMSE on the 2025 holdout.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from src import predict
from src.team_colors import get_team_color

# Data only changes on redeploy (a new push restarts the app), so results can be
# cached for the life of the process: switching races doesn't retrain the model.
predict_race = st.cache_data(show_spinner=False)(predict.predict_race)
get_race_metadata = st.cache_data(show_spinner=False)(predict.get_race_metadata)
get_available_races = st.cache_data(show_spinner=False)(predict.get_available_races)
get_calendar = st.cache_data(show_spinner=False)(predict.get_calendar)
season_metrics = st.cache_data(show_spinner=False)(predict.season_metrics)


# F1 timing-screen language for prediction accuracy: purple = exactly right,
# green = within 2 places, yellow = further off. Validated on the dark surface
# (colour-blind separation 12.3); always shown with the number, never colour alone.
TIMING_EXACT, TIMING_CLOSE, TIMING_OFF = "#A855F7", "#10B981", "#FBBF24"


def timing_colour(delta: float) -> str:
    """Colour for how far the actual finish was from the predicted rank."""
    if abs(delta) == 0:
        return TIMING_EXACT
    return TIMING_CLOSE if abs(delta) <= 2 else TIMING_OFF


def chance(p: float) -> str:
    """Simulated chance as a percentage; tiny non-zero chances show as <1%."""
    return "<1%" if 0 < p < 0.005 else f"{p:.0%}"


@st.cache_data(show_spinner=False)
def get_seasons() -> list:
    """Seasons the dashboard can show, newest first. The first season in the
    data has no earlier season to train on, so it's left out."""
    return sorted(predict.load_features()["Year"].unique().tolist(), reverse=True)[:-1]


# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Custom CSS — F1 broadcast aesthetic
# ──────────────────────────────────────────────────────────────────────
F1_RED = "#E10600"
F1_DARK = "#15151E"
F1_DARK_2 = "#1F1F2C"
F1_LIGHT = "#FFFFFF"
F1_GREY = "#949498"

st.markdown(f"""
<style>
    /* Import F1-style font */
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

    /* Global background */
    .stApp {{
        background-color: {F1_DARK};
        color: {F1_LIGHT};
        font-family: 'Titillium Web', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
        overflow-x: hidden;
    }}

    /* Top accent bar — broadcast convention */
    .f1-top-bar {{
        height: 4px;
        background: linear-gradient(90deg, {F1_RED} 0%, {F1_RED} 60%, {F1_DARK_2} 60%, {F1_DARK_2} 100%);
        margin: -1rem -1rem 0 -1rem;
        position: sticky;
        top: 0;
        z-index: 50;
        pointer-events: none;
    }}

    /* Validation banner — chyron-style */
    .f1-validation-banner {{
        background: linear-gradient(90deg, rgba(225, 6, 0, 0.12) 0%, rgba(225, 6, 0, 0.02) 100%);
        border-left: 3px solid {F1_RED};
        padding: 0.65rem 1.25rem;
        margin: 0.5rem 0 0 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }}
    .f1-validation-banner-left {{ color: {F1_LIGHT}; }}
    .f1-validation-banner-right {{ color: #00D26A; }}
    .f1-validation-banner .pulse {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00D26A;
        border-radius: 50%;
        margin-right: 0.5rem;
        vertical-align: middle;
        box-shadow: 0 0 0 0 rgba(0, 210, 106, 0.7);
        animation: pulse-glow 2s infinite;
    }}
    @keyframes pulse-glow {{
        0% {{ box-shadow: 0 0 0 0 rgba(0, 210, 106, 0.6); }}
        70% {{ box-shadow: 0 0 0 6px rgba(0, 210, 106, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0, 210, 106, 0); }}
    }}

    /* Hero header */
    .f1-hero {{
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #2A2A38;
        margin-bottom: 2rem;
    }}
    .f1-hero-eyebrow {{
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: {F1_RED};
        margin-bottom: 0.5rem;
    }}
    .f1-hero-title {{
        font-size: 2.75rem;
        font-weight: 900;
        line-height: 1.05;
        color: {F1_LIGHT};
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.01em;
    }}
    .f1-hero-subtitle {{
        font-size: 1rem;
        color: {F1_GREY};
        font-weight: 400;
    }}

    /* Custom metric cards */
    .f1-metrics-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }}
    .f1-metric-card {{
        background: {F1_DARK_2};
        border: 1px solid #2A2A38;
        border-left: 3px solid {F1_RED};
        padding: 1.25rem 1.5rem;
        border-radius: 4px;
    }}
    .f1-metric-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {F1_GREY};
        margin-bottom: 0.5rem;
    }}
    .f1-metric-value {{
        font-size: 2rem;
        font-weight: 900;
        color: {F1_LIGHT};
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }}
    .f1-metric-value-correct {{ color: #00D26A; }}
    .f1-metric-value-missed {{ color: {F1_RED}; }}
    .f1-metric-context {{
        font-size: 0.85rem;
        color: {F1_GREY};
        margin-top: 0.4rem;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {F1_DARK_2};
        border-right: 1px solid #2A2A38;
    }}
    section[data-testid="stSidebar"] h1 {{
        font-weight: 900;
        font-size: 1.5rem;
        color: {F1_LIGHT};
        margin-bottom: 0.25rem;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p {{
        color: {F1_GREY} !important;
        opacity: 1 !important;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 700;
    }}
    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p strong {{
        color: {F1_LIGHT} !important;
        opacity: 1 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.8rem;
        font-weight: 700;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        opacity: 1 !important;
    }}
    .stSelectbox label {{
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: {F1_GREY} !important;
    }}

    /* Hide Streamlit branding — but PRESERVE the header so the sidebar toggle works */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* Keep the header bar present and ABOVE content, just transparent */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        z-index: 999990 !important;
    }}
    /* Hide only the toolbar (Deploy/Share), NOT the whole header */
    header[data-testid="stHeader"] [data-testid="stToolbar"] {{
        display: none !important;
    }}


    header[data-testid="stHeader"] {{
        background: transparent !important;
        z-index: 999990 !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Force sidebar to always show, even after a collapse attempt */
    section[data-testid="stSidebar"] {{
        background-color: {F1_DARK_2};
        border-right: 1px solid #2A2A38;
        min-width: 244px !important;
        max-width: 244px !important;
        transform: none !important;
        visibility: visible !important;
        margin-left: 0px !important;
    }}

    section[data-testid="stSidebar"][aria-expanded="false"] {{
        transform: none !important;
        margin-left: 0px !important;
        visibility: visible !important;
    }}

    /* Hide the collapse button so it can't be triggered */
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="stBaseButton-headerNoPadding"] {{
        display: none !important;
    }}

    /* Make selectboxes non-typeable — click to open, select only (no search box confusion) */
    .stSelectbox [data-baseweb="select"] input {{
        caret-color: transparent !important;
        pointer-events: none !important;
    }}

    /* Section headers */
    .f1-section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {F1_LIGHT};
        margin: 2.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2A2A38;
    }}
    .f1-section-title::before {{
        content: '';
        display: inline-block;
        width: 4px;
        height: 1rem;
        background: {F1_RED};
        margin-right: 0.75rem;
        vertical-align: middle;
    }}

    /* Basic mobile responsiveness — stack grids on narrow screens */
    @media (max-width: 640px) {{
        .f1-metrics-grid {{
            grid-template-columns: 1fr !important;
        }}
        .f1-hero-title {{
            font-size: 1.8rem !important;
        }}
        .f1-pred-header, .f1-pred-row {{
            grid-template-columns: 40px 1.5fr 1fr 50px !important;
            font-size: 0.7rem !important;
        }}
        /* Hide less-critical columns on mobile to prevent squishing */
        .f1-pred-header > div:nth-child(5),
        .f1-pred-header > div:nth-child(6),
        .f1-pred-header > div:nth-child(7),
        .f1-pred-row > .f1-pred-cell:nth-child(5),
        .f1-pred-row > .f1-pred-cell:nth-child(6),
        .f1-pred-row > .f1-pred-cell:nth-child(7) {{
            display: none !important;
        }}
    }}
</style>

<div class="f1-top-bar"></div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# F1 PREDICTOR")
    st.caption("ML race forecasting")

    st.markdown("---")

    # Shareable links: ?season=2026&round=16 opens that race. The dropdowns own the
    # values and write them to the URL; the URL is applied on first load, or when it
    # no longer matches what we last wrote (a new link opened in an existing tab,
    # since Streamlit keeps the session across reloads).
    params = st.query_params
    seasons = get_seasons()
    url_season = int(params["season"]) if params.get("season", "").isdigit() else None
    url_round = int(params["round"]) if params.get("round", "").isdigit() else None
    apply_url = st.session_state.get("url_written") != (url_season, url_round)

    if apply_url and url_season in seasons:
        st.session_state.season = url_season
    elif "season" not in st.session_state:
        st.session_state.season = seasons[0]
    year = st.selectbox("Season", options=seasons, key="season")

    races = get_available_races(year)
    race_options = {
        f"R{row['Round']:02d} — {row['EventName']}": row["Round"]
        for _, row in races.iterrows()
    }
    latest_label = list(race_options)[-1]  # latest race with data: the upcoming one once qualifying is in
    # Rounds on the calendar with no data yet (qualifying not run): listed with their date
    calendar = get_calendar(year)
    future = calendar[~calendar["Round"].isin(races["Round"])]
    for _, row in future.iterrows():
        race_options[f"R{row['Round']:02d} — {row['EventName']} · {row['RaceUtc']:%d %b}"] = row["Round"]
    race_options = dict(sorted(race_options.items(), key=lambda item: item[1]))
    labels = list(race_options)
    race_key = f"race_{year}"  # one per season, so each season remembers its race
    from_url = [label for label, rnd in race_options.items() if url_season == year and rnd == url_round]
    if apply_url and from_url:
        st.session_state[race_key] = from_url[0]
    elif race_key not in st.session_state:
        st.session_state[race_key] = latest_label
    selected_race_label = st.selectbox("Race", options=labels, key=race_key)
    selected_round = race_options[selected_race_label]

    st.query_params.update(season=str(year), round=str(selected_round))
    st.session_state.url_written = (year, int(selected_round))

    st.markdown("---")
    st.markdown("**Model performance**")
    st.caption("Trained only on earlier seasons · vs predicting the qualifying order")

    # The two most recent seasons with completed races, computed from the data
    metric_blocks = []
    for season in seasons:
        m = season_metrics(season)
        if m["races"] == 0:
            continue
        metric_blocks.append(
            f"<strong style='color: {F1_LIGHT};'>{season} · {m['races']} races</strong><br>"
            f"Winner in top 3: <strong style='color: {F1_LIGHT};'>{m['winner_top3']}/{m['races']}</strong><br>"
            f"Exact winner: <strong style='color: {F1_LIGHT};'>{m['winner_top1']}/{m['races']}</strong><br>"
            f"RMSE: <strong style='color: {F1_LIGHT};'>{m['rmse']:.2f}</strong> "
            f"<span style='color: {F1_GREY};'>(pole {m['pole_rmse']:.2f})</span>"
        )
        if len(metric_blocks) == 2:
            break
    st.markdown(
        f"<div style='font-size: 0.85rem; color: {F1_GREY}; line-height: 1.6;'>"
        + "<br><br>".join(metric_blocks)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style='font-size: 0.8rem; color: {F1_GREY};'>
        Built by <a href='https://www.linkedin.com/in/om-patil-nu' style='color: {F1_RED}; text-decoration: none;'>Om Patil</a><br>
        <a href='https://github.com/Om-Ravindra-Patil/F1-Race-Predictor' style='color: {F1_RED}; text-decoration: none;'>GitHub</a><br><br>
        <span style='font-size: 0.7rem;'>Unofficial fan project. Not associated with Formula 1, the FIA or any team.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Next race banner — what's coming up in the newest season
# ──────────────────────────────────────────────────────────────────────
def countdown(when: pd.Timestamp, now: pd.Timestamp) -> str:
    left = when - now
    days, hours = left.days, left.seconds // 3600
    return f"{days}d {hours:02d}h" if days else f"{hours}h {left.seconds % 3600 // 60:02d}m"


now = pd.Timestamp.now(tz="UTC").tz_localize(None)
season_calendar = get_calendar(seasons[0])
upcoming_events = season_calendar[season_calendar["RaceUtc"] > now]
if not upcoming_events.empty:
    nxt = upcoming_events.iloc[0]
    if nxt["QualifyingUtc"] > now:
        status = (f"Qualifying in <strong style='color:{F1_LIGHT};'>{countdown(nxt['QualifyingUtc'], now)}</strong> "
                  f"· {nxt['QualifyingUtc']:%a %d %b %H:%M} UTC · prediction appears after")
    else:
        status = (f"Race in <strong style='color:{F1_LIGHT};'>{countdown(nxt['RaceUtc'], now)}</strong> "
                  f"· {nxt['RaceUtc']:%a %d %b %H:%M} UTC · "
                  f"<a href='?season={seasons[0]}&round={nxt['Round']}' target='_self' "
                  f"style='color:{F1_RED};text-decoration:none;font-weight:700;'>see the prediction →</a>")
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:0.4rem 1rem;align-items:baseline;"
        f"background:{F1_DARK_2};border:1px solid #2A2A38;border-left:4px solid {F1_RED};"
        f"border-radius:4px;padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:{F1_GREY};'>"
        f"<span style='font-size:0.7rem;font-weight:700;letter-spacing:0.15em;color:{F1_RED};'>NEXT RACE</span>"
        f"<span style='color:{F1_LIGHT};font-weight:700;'>R{nxt['Round']} · {nxt['EventName']}</span>"
        f"<span>{status}</span></div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Future race — on the calendar, but qualifying hasn't run yet
# ──────────────────────────────────────────────────────────────────────
if selected_round in future["Round"].values:
    event = future[future["Round"] == selected_round].iloc[0]
    st.markdown(f"""
<div class="f1-hero">
    <div class="f1-hero-eyebrow">Round {event['Round']} · Season {year}</div>
    <h1 class="f1-hero-title">{event['EventName']}</h1>
    <div class="f1-hero-subtitle">{event['Location']} · {event['RaceUtc']:%A %d %B %Y}</div>
</div>
<div style="background:{F1_DARK_2};border:1px solid #2A2A38;border-left:4px solid #FFC107;border-radius:4px;padding:1.5rem;">
    <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:#FFC107;margin-bottom:0.75rem;">Prediction not available yet</div>
    <div style="color:{F1_LIGHT};font-size:1.05rem;margin-bottom:0.75rem;">
        The prediction appears after qualifying on <strong>{event['QualifyingUtc']:%A %d %B, %H:%M} UTC</strong>.
    </div>
    <div style="color:{F1_GREY};font-size:0.85rem;line-height:1.6;">
        Race: {event['RaceUtc']:%A %d %B, %H:%M} UTC.<br>
        The model needs the qualifying result: from recent form alone it names the winner about a third as often.
        <a href="?season={year}&round={race_options[latest_label]}" target="_self" style="color:{F1_RED};text-decoration:none;">See the latest prediction →</a>
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────────────
# Main panel — fetch data (with safety net)
# ──────────────────────────────────────────────────────────────────────
try:
    metadata = get_race_metadata(year, selected_round)
    predictions = predict_race(year, selected_round)

    # Upcoming race: qualifying done, no results yet — predictions only
    upcoming = predictions["ActualPosition"].isna().all()
    predicted_winner = predictions[predictions["PredictedRank"] == 1].iloc[0]

    if not upcoming and not (predictions["ActualPosition"] == 1).any():
        st.warning(
            f"Results for {metadata['event_name']} ({year}) are incomplete — no race winner recorded."
        )
        st.stop()

except Exception as e:
    st.error(
        f"Unable to load predictions for this race. "
        f"This usually means the race data is incomplete or has a known issue. "
        f"Error details: `{type(e).__name__}: {str(e)[:200]}`"
    )
    st.info(
        "Try selecting a different race, or check the GitHub repository for known data issues."
    )
    st.stop()


if not upcoming:
    actual_winner = predictions[predictions["ActualPosition"] == 1].iloc[0]
    winner_correct = actual_winner["Abbreviation"] == predicted_winner["Abbreviation"]

    predicted_top3 = predictions[predictions["PredictedRank"] <= 3]["Abbreviation"].tolist()
    actual_top3 = predictions[predictions["ActualPosition"] <= 3]["Abbreviation"].tolist()
    top3_overlap = len(set(predicted_top3) & set(actual_top3))

    valid = predictions.dropna(subset=["ActualPosition"])
    race_mae = (valid["ActualPosition"] - valid["PredictedRank"]).abs().mean()

    # Biggest climber: driver who gained the most positions from quali to finish
    climbers = valid.dropna(subset=["QualifyingPosition"]).copy()
    climbers["PositionsGained"] = climbers["QualifyingPosition"] - climbers["ActualPosition"]
    if len(climbers) > 0 and climbers["PositionsGained"].max() > 0:
        top_climber = climbers.loc[climbers["PositionsGained"].idxmax()]
        climber_code = top_climber["Abbreviation"]
        climber_team = top_climber["TeamName"]
        climber_gain = int(top_climber["PositionsGained"])
        climber_from = int(top_climber["QualifyingPosition"])
        climber_to = int(top_climber["ActualPosition"])
    else:
        climber_code = "—"
        climber_team = "No gains"
        climber_gain = 0
        climber_from = None
        climber_to = None


# ──────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="f1-hero">
    <div class="f1-hero-eyebrow">Round {metadata['round']} · Season {year}</div>
    <h1 class="f1-hero-title">{metadata['event_name']}</h1>
    <div class="f1-hero-subtitle">{metadata['circuit']} · {metadata['event_date']}</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Top metrics — custom cards (broadcast style)
# ──────────────────────────────────────────────────────────────────────
if upcoming:
    pole = predictions.loc[predictions["QualifyingPosition"].idxmin()]
    podium_codes = " · ".join(predictions.nsmallest(3, "PredictedRank")["Abbreviation"])
    cards = [
        ("Predicted Winner", f'<span class="f1-metric-value-correct">{predicted_winner["Abbreviation"]}</span>', f'{predicted_winner["TeamName"]} · {chance(predicted_winner["WinChance"])} to win'),
        ("Pole Position", pole["Abbreviation"], pole["TeamName"]),
        ("Predicted Podium", f'<span style="font-size: 1.3rem;">{podium_codes}</span>', "Top 3 by predicted finish"),
        ("Race Status", '<span style="color: #FFC107;">UPCOMING</span>', "Grid = qualifying order (penalties not yet known)"),
    ]
else:
    winner_color_class = "f1-metric-value-correct" if winner_correct else "f1-metric-value-missed"
    winner_status = "PREDICTED" if winner_correct else "MISSED"
    top3_color_class = "f1-metric-value-correct" if top3_overlap == 3 else ""

    if climber_gain > 0:
        climber_value_html = f'<span style="color: #00D26A;">+{climber_gain}</span> <span style="font-size: 1.3rem; color: {F1_LIGHT};">{climber_code}</span>'
        climber_context = f"P{climber_from} → P{climber_to} · {climber_team}"
    else:
        climber_value_html = '<span style="color: #949498;">—</span>'
        climber_context = "No driver gained positions"

    cards = [
        ("Actual Winner", actual_winner["Abbreviation"], actual_winner["TeamName"]),
        ("Predicted Winner", f'<span class="{winner_color_class}">{predicted_winner["Abbreviation"]}</span>', f'{winner_status} · {predicted_winner["TeamName"]}'),
        ("Top-3 Overlap", f'<span class="{top3_color_class}">{top3_overlap}/3</span>', "Drivers in correct podium zone"),
        ("Biggest Climber", climber_value_html, climber_context),
    ]

cards_html = "".join(
    f'<div class="f1-metric-card"><div class="f1-metric-label">{label}</div>'
    f'<div class="f1-metric-value">{value}</div><div class="f1-metric-context">{context}</div></div>'
    for label, value, context in cards
)
st.markdown(f'<div class="f1-metrics-grid">{cards_html}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Placeholder sections for next layers
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="f1-section-title">PODIUM</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Podium helper — renders one podium (predicted or actual)
# ──────────────────────────────────────────────────────────────────────
def render_podium_box(row, position: int, height_px: int) -> str:
    """Render a single podium box as a flat HTML string."""
    team = row["TeamName"]
    team_color = get_team_color(team)
    driver_code = row["Abbreviation"]
    full_name = row["FullName"]

    position_colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
    pos_color = position_colors.get(position, F1_LIGHT)

    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;flex:1;margin:0 0.4rem;">'
        f'<div style="text-align:center;margin-bottom:0.75rem;min-height:60px;">'
        f'<div style="font-size:1.6rem;font-weight:900;color:{F1_LIGHT};line-height:1;">{driver_code}</div>'
        f'<div style="font-size:0.7rem;color:{F1_GREY};margin-top:0.3rem;line-height:1.3;">{full_name}</div>'
        f'<div style="font-size:0.65rem;color:{team_color};margin-top:0.2rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">{team}</div>'
        f'</div>'
        f'<div style="width:100%;height:{height_px}px;background:linear-gradient(180deg,{F1_DARK_2} 0%,#0E0E16 100%);border-top:4px solid {team_color};border-radius:4px 4px 0 0;display:flex;align-items:flex-start;justify-content:center;padding-top:1rem;box-shadow:0 -2px 8px rgba(0,0,0,0.3);">'
        f'<div style="font-size:2.5rem;font-weight:900;color:{pos_color};line-height:1;text-shadow:0 2px 4px rgba(0,0,0,0.5);">P{position}</div>'
        f'</div>'
        f'</div>'
    )


def render_podium(p1_row, p2_row, p3_row, title: str) -> str:
    """Render a complete podium panel."""
    box_p2 = render_podium_box(p2_row, 2, 100)
    box_p1 = render_podium_box(p1_row, 1, 140)
    box_p3 = render_podium_box(p3_row, 3, 70)

    return (
        f'<div style="background:{F1_DARK_2};border:1px solid #2A2A38;border-radius:4px;padding:1.5rem;margin-bottom:1rem;">'
        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:{F1_RED};margin-bottom:1.5rem;text-align:center;">{title}</div>'
        f'<div style="display:flex;align-items:flex-end;justify-content:center;min-height:220px;padding:0 1rem;">'
        f'{box_p2}{box_p1}{box_p3}'
        f'</div>'
        f'</div>'
    )


# Get the actual top 3 (sorted by ActualPosition)
actual_top3_df = (
    predictions.dropna(subset=["ActualPosition"])
    .nsmallest(3, "ActualPosition")
    .sort_values("ActualPosition")
    .reset_index(drop=True)
)

# Get the predicted top 3 (sorted by PredictedRank)
predicted_top3_df = (
    predictions.nsmallest(3, "PredictedRank")
    .sort_values("PredictedRank")
    .reset_index(drop=True)
)

# Render both podiums side-by-side
col_pred, col_actual = st.columns(2)

with col_pred:
    st.markdown(
        render_podium(
            predicted_top3_df.iloc[0],
            predicted_top3_df.iloc[1],
            predicted_top3_df.iloc[2],
            "Predicted Podium",
        ),
        unsafe_allow_html=True,
    )

with col_actual:
    if upcoming:
        st.markdown(
            f'<div style="background:{F1_DARK_2};border:1px solid #2A2A38;border-radius:4px;padding:1.5rem;'
            f'min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">'
            f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:{F1_RED};margin-bottom:1rem;">Actual Podium</div>'
            f'<div style="color:{F1_GREY};">Race on {metadata["event_date"]} — results appear here after the race.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            render_podium(
                actual_top3_df.iloc[0],
                actual_top3_df.iloc[1],
                actual_top3_df.iloc[2],
                "Actual Podium",
            ),
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────
# Why this prediction — exact per-factor breakdown of the linear model
# ──────────────────────────────────────────────────────────────────────
WHY_GAIN = "#3B82F6"  # pushes towards P1 (diverging pair validated on F1_DARK_2)
WHY_LOSE = F1_RED     # pushes towards the back

st.markdown('<div class="f1-section-title">WHY THIS PREDICTION</div>', unsafe_allow_html=True)
st.markdown(f"""
<style>
.why-card {{ background:{F1_DARK_2}; border:1px solid #2A2A38; border-radius:4px; padding:1.25rem 1.5rem; }}
.why-head {{ color:{F1_LIGHT}; font-size:1.05rem; font-weight:700; }}
.why-sub {{ color:{F1_GREY}; font-size:0.8rem; margin:0.25rem 0 1.25rem 0; }}
.why-row {{ display:grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr) 5.5rem; gap:1rem; align-items:center; padding:0.55rem 0; }}
.why-label {{ color:{F1_LIGHT}; font-size:0.85rem; font-weight:600; }}
.why-context {{ color:{F1_GREY}; font-size:0.75rem; margin-top:0.15rem; }}
.why-track {{ position:relative; height:14px; }}
.why-zero {{ position:absolute; left:50%; top:-4px; bottom:-4px; width:1px; background:#4A4A58; }}
.why-bar {{ position:absolute; top:0; height:14px; }}
.why-value {{ color:{F1_LIGHT}; font-size:0.85rem; font-variant-numeric:tabular-nums; text-align:right; }}
.why-legend {{ display:flex; justify-content:space-between; color:{F1_GREY}; font-size:0.72rem; margin-top:0.75rem; }}
.why-swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 0.35rem; vertical-align:-1px; }}
@media (max-width: 640px) {{
    .why-row {{ grid-template-columns: 1fr 4.5rem; }}
    .why-track {{ grid-column: 1 / -1; grid-row: 2; }}
}}
</style>
""", unsafe_allow_html=True)

WHY_FACTORS = [("WhyQualifying", "Qualifying"), ("WhyDriverForm", "Driver form"), ("WhyTeamForm", "Team form")]
# One scale for the whole race, so bars are comparable when switching drivers
why_scale = max(predictions[[col for col, _ in WHY_FACTORS]].abs().max().max(), 1e-9)

driver_labels = [f"P{int(r['PredictedRank'])} · {r['Abbreviation']} — {r['FullName']}" for _, r in predictions.iterrows()]
why_label = st.selectbox("Driver", driver_labels, key=f"why_{year}_{selected_round}")
why = predictions.iloc[driver_labels.index(why_label)]


def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")  # typographic minus


def why_context(col: str) -> str:
    if col == "WhyQualifying":
        text = "On pole" if why["QualifyingPosition"] == 1 else (
            f"Qualified P{int(why['QualifyingPosition'])}, {why['QualifyingGapToPole']:.2f}s off pole")
        if why["GridPosition"] != why["QualifyingPosition"]:
            text += f" · starts P{int(why['GridPosition'])}"
        return text
    if why["FormEstimated"]:
        return "No recent results — estimated"
    if col == "WhyDriverForm":
        return f"Averaged P{why['DriverFormLast3']:.1f} over the last 3 races"
    return f"Team averaged P{why['TeamFormLast3']:.1f} over the last 3 races"


def why_row(col: str, label: str) -> str:
    value = why[col]
    width = abs(value) / why_scale * 50  # % of the track; each side of zero is half
    # Bars grow from the zero line; the 4px rounded end is the data end
    if value < 0:
        bar = f"right:50%;width:{width:.2f}%;background:{WHY_GAIN};border-radius:4px 0 0 4px;"
        verdict = f"gains {abs(value):.1f} places"
    else:
        bar = f"left:50%;width:{width:.2f}%;background:{WHY_LOSE};border-radius:0 4px 4px 0;"
        verdict = f"loses {abs(value):.1f} places"
    return (
        f'<div class="why-row" title="{label}: {verdict} vs the field average">'
        f'<div><div class="why-label">{label}</div><div class="why-context">{why_context(col)}</div></div>'
        f'<div class="why-track"><div class="why-zero"></div><div class="why-bar" style="{bar}"></div></div>'
        f'<div class="why-value">{signed(value)}</div>'
        f'</div>'
    )


st.markdown(
    f'<div class="why-card">'
    f'<div class="why-head">{why["Abbreviation"]} is predicted P{int(why["PredictedRank"])} · {chance(why["WinChance"])} to win, {chance(why["PodiumChance"])} podium</div>'
    f'<div class="why-sub">Predicted finish {why["PredictedPosition"]:.1f} vs a field average of {why["FieldAverage"]:.1f}. '
    f'The factors below add up exactly to the difference ({signed(why["PredictedPosition"] - why["FieldAverage"])} places).</div>'
    + "".join(why_row(col, label) for col, label in WHY_FACTORS)
    + f'<div class="why-legend"><span><span class="why-swatch" style="background:{WHY_GAIN};"></span>Gains places (towards P1)</span>'
    f'<span>Loses places (towards the back)<span class="why-swatch" style="background:{WHY_LOSE};"></span></span></div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# Race story — qualifying → predicted → result, one line per driver
# ──────────────────────────────────────────────────────────────────────
def race_story_svg(preds: pd.DataFrame, highlight: str) -> str:
    """Bump chart: every driver a muted line, the highlighted one in team colour."""
    stages = [("Qualifying", "QualifyingPosition"), ("Predicted", "PredictedRank")]
    if not upcoming:
        stages.append(("Result", "ActualPosition"))

    width, label_w, top, row_h = 720, 64, 34, 18
    max_pos = int(preds[[col for _, col in stages]].max().max())
    height = top + max_pos * row_h + 6
    xs = [label_w + i * (width - 2 * label_w) / (len(stages) - 1) for i in range(len(stages))]

    def y(pos: float) -> float:
        return top + (pos - 0.5) * row_h

    parts = [
        f'<text x="{x:.1f}" y="14" text-anchor="middle" fill="{F1_GREY}" font-size="11" '
        f'font-weight="700" letter-spacing="0.08em">{name.upper()}</text>'
        for x, (name, _) in zip(xs, stages)
    ]
    parts += [
        f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 4}" stroke="#2A2A38" stroke-width="1"/>'
        for x in xs
    ]

    # Highlighted driver drawn last so it sits on top
    rows = sorted(preds.to_dict("records"), key=lambda r: r["Abbreviation"] == highlight)
    for r in rows:
        points = [(x, r[col]) for x, (_, col) in zip(xs, stages) if not pd.isna(r[col])]
        if len(points) < 2:
            continue
        is_hl = r["Abbreviation"] == highlight
        colour = get_team_color(r["TeamName"]) if is_hl else "#4A4A58"
        path = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in points)
        tip = f"{r['Abbreviation']} · " + " → ".join(
            f"{name} P{int(r[col])}" for name, col in stages if not pd.isna(r[col]))
        parts.append(
            f'<g><title>{tip}</title>'
            f'<polyline points="{path}" fill="none" stroke="transparent" stroke-width="10"/>'  # hover target
            f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="{3 if is_hl else 1.5}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{1 if is_hl else 0.9}"/>'
            + ("".join(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="4.5" fill="{colour}" stroke="{F1_DARK_2}" stroke-width="2"/>'
                       for x, v in points) if is_hl else "")
            + '</g>'
        )
        # Driver codes at both ends, in text colours (bold white for the highlight)
        ink, weight = (F1_LIGHT, 700) if is_hl else (F1_GREY, 400)
        (x0, v0), (x1, v1) = points[0], points[-1]
        parts.append(f'<text x="{x0 - 10:.1f}" y="{y(v0) + 4:.1f}" text-anchor="end" fill="{ink}" font-size="11" font-weight="{weight}">{r["Abbreviation"]}</text>')
        parts.append(f'<text x="{x1 + 10:.1f}" y="{y(v1) + 4:.1f}" text-anchor="start" fill="{ink}" font-size="11" font-weight="{weight}">{r["Abbreviation"]}</text>')

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Qualifying, predicted and actual positions for every driver; {highlight} highlighted" '
        f'style="display:block;font-family:inherit;">' + "".join(parts) + "</svg>"
    )


st.markdown('<div class="f1-section-title">RACE STORY</div>', unsafe_allow_html=True)
st.caption(
    f"Every driver from qualifying to {'the predicted finish' if upcoming else 'predicted finish to the actual result'}. "
    f"{why['Abbreviation']} is highlighted — pick another driver above. Hover a line for exact positions."
)
st.markdown(
    f'<div class="why-card">{race_story_svg(predictions, why["Abbreviation"])}</div>',
    unsafe_allow_html=True,
)


st.markdown('<div class="f1-section-title">PREDICTIONS</div>', unsafe_allow_html=True)
st.caption(
    "Win · Podium: how often each driver won or finished top 3 across 10,000 simulated "
    "versions of this race, using the model's own past errors. Checked on 2025–26: "
    "drivers given ~40% won 44% of the time."
)
if not upcoming:
    st.markdown(
        f'<div style="font-size:0.8rem;color:{F1_GREY};margin:-0.25rem 0 0.75rem 0;">'
        f'Δ = actual finish vs predicted, in F1 timing colours: '
        f'<span style="color:{TIMING_EXACT};font-weight:700;">● exact</span> · '
        f'<span style="color:{TIMING_CLOSE};font-weight:700;">● within 2 places</span> · '
        f'<span style="color:{TIMING_OFF};font-weight:700;">● further off</span></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Custom team-coloured predictions table
# ──────────────────────────────────────────────────────────────────────
def render_prediction_row(row) -> str:
    """Render one driver's row in the predictions table."""
    team = row["TeamName"]
    team_color = get_team_color(team)
    driver_code = row["Abbreviation"]
    full_name = row["FullName"]
    pred_rank = int(row["PredictedRank"])
    quali = int(row["QualifyingPosition"]) if not pd.isna(row["QualifyingPosition"]) else "—"
    actual = int(row["ActualPosition"]) if not pd.isna(row["ActualPosition"]) else ("—" if upcoming else "DNF")
    delta = row["PositionDelta"]

    # Delta styling
    if pd.isna(delta):
        delta_text = "—"
        delta_color = F1_GREY
    else:
        delta_int = int(delta)
        delta_text = "exact" if delta_int == 0 else f"{delta_int:+d}".replace("-", "−")
        delta_color = timing_colour(delta_int)

    chance_html = (
        f'<span style="color:{F1_LIGHT};font-variant-numeric:tabular-nums;">{chance(row["WinChance"])}</span>'
        f'<span style="color:{F1_GREY};font-variant-numeric:tabular-nums;"> · {chance(row["PodiumChance"])}</span>'
    )

    return (
        f'<div class="f1-pred-row" style="border-left:4px solid {team_color};">'
        f'<div class="f1-pred-cell f1-pred-rank">{pred_rank}</div>'
        f'<div class="f1-pred-cell f1-pred-driver">'
        f'<div class="f1-pred-code">{driver_code}</div>'
        f'<div class="f1-pred-name">{full_name}{" · est. form" if row["FormEstimated"] else ""}</div>'
        f'</div>'
        f'<div class="f1-pred-cell f1-pred-team" style="color:{team_color};">{team}</div>'
        f'<div class="f1-pred-cell">{chance_html}</div>'
        f'<div class="f1-pred-cell f1-pred-num">{quali}</div>'
        f'<div class="f1-pred-cell f1-pred-num">{actual}</div>'
        f'<div class="f1-pred-cell f1-pred-num" style="color:{delta_color};">{delta_text}</div>'
        f'</div>'
    )   



# Inject the table-specific CSS once
st.markdown(f"""
<style>
.f1-pred-table {{
    background: {F1_DARK_2};
    border: 1px solid #2A2A38;
    border-radius: 4px;
    overflow: hidden;
    margin: 0;
}}
.f1-pred-header {{
    display: grid;
    grid-template-columns: 60px 2fr 2fr 80px 1fr 1fr 1fr;
    align-items: center;
    padding: 0.85rem 1rem 0.85rem 0.85rem;
    background: #0E0E16;
    border-bottom: 1px solid #2A2A38;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {F1_GREY};
}}
.f1-pred-row {{
    display: grid;
    grid-template-columns: 60px 2fr 2fr 80px 1fr 1fr 1fr;
    align-items: center;
    padding: 0.7rem 1rem 0.7rem 0.85rem;
    border-bottom: 1px solid #2A2A38;
    transition: background 0.15s ease;
}}
.f1-pred-row:hover {{
    background: rgba(225, 6, 0, 0.04);
}}
.f1-pred-row:last-child {{
    border-bottom: none;
}}
.f1-pred-cell {{
    font-size: 0.9rem;
    color: {F1_LIGHT};
}}
.f1-pred-rank {{
    font-size: 1.3rem;
    font-weight: 900;
    color: {F1_LIGHT};
    font-variant-numeric: tabular-nums;
}}
.f1-pred-code {{
    font-size: 1rem;
    font-weight: 900;
    color: {F1_LIGHT};
    line-height: 1.1;
}}
.f1-pred-name {{
    font-size: 0.75rem;
    color: {F1_GREY};
    margin-top: 0.15rem;
}}
.f1-pred-team {{
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.f1-pred-num {{
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)


# Build the header
header_html = (
    '<div class="f1-pred-header">'
    '<div>Pred</div>'
    '<div>Driver</div>'
    '<div>Team</div>'
    '<div title="From 10,000 simulated races">Win · Podium</div>'
    '<div style="text-align:center;">Quali</div>'
    '<div style="text-align:center;">Actual</div>'
    '<div style="text-align:center;">Δ</div>'
    '</div>'
)

# Build all rows
rows_html = "".join(render_prediction_row(row) for _, row in predictions.iterrows())

# Render the complete table
st.markdown(
    f'<div class="f1-pred-table">{header_html}{rows_html}</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# Footer — context for first-time visitors
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="f1-section-title">ABOUT</div>', unsafe_allow_html=True)

footer_html = f"""
<div style="
    background: {F1_DARK_2};
    border: 1px solid #2A2A38;
    border-radius: 4px;
    padding: 1.75rem;
    margin-bottom: 2rem;
    line-height: 1.65;
">
    <div style="
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2.5rem;
    ">
        <div>
            <div style="
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: {F1_RED};
                margin-bottom: 0.75rem;
            ">How it works</div>
            <div style="color: {F1_LIGHT}; font-size: 0.9rem;">
                A 6-feature linear regression predicts each driver's finish position from
                qualifying performance, recent form, and circuit type. The model trains on
                every season before the one shown, so each race is predicted by a model
                that has never seen that season or any later one.
            </div>
        </div>
        <div>
            <div style="
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: {F1_RED};
                margin-bottom: 0.75rem;
            ">What it captures · what it misses</div>
            <div style="color: {F1_LIGHT}; font-size: 0.9rem;">
                <strong>Captures:</strong> qualifying-driven race outcomes, team-level pace,
                driver form trajectories.<br>
                <strong>Misses:</strong> in-race chaos — wet weather, safety cars, mechanical
                DNFs, strategic upsets.
            </div>
        </div>
    </div>
    <div style="
        margin-top: 1.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid #2A2A38;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        color: {F1_GREY};
    ">
        <div>
            Trained on 2022–2024 · Validated on 2025 (true holdout · RMSE 4.25 vs 4.69 pole baseline)
        </div>
        <div>
            <a href="https://github.com/Om-Ravindra-Patil/F1-Race-Predictor" style="color: {F1_RED}; text-decoration: none; font-weight: 600;">View source on GitHub →</a>
        </div>
    </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
