"""
F1 Race Winner Predictor — Streamlit Dashboard

F1 broadcast-style interactive dashboard showcasing the validated prediction model.
Trained on 2022-2024, achieves 100% top-3 accuracy on 2025 holdout.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from src.predict import predict_race, get_race_metadata, get_available_races
from src.team_colors import get_team_color


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
    }}

    /* Top accent bar — broadcast convention */
    .f1-top-bar {{
        height: 4px;
        background: linear-gradient(90deg, {F1_RED} 0%, {F1_RED} 60%, {F1_DARK_2} 60%, {F1_DARK_2} 100%);
        margin: -1rem -1rem 0 -1rem;
        position: sticky;
        top: 0;
        z-index: 100;
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

    .f1-metric-value-correct {{
        color: #00D26A;
    }}

    .f1-metric-value-missed {{
        color: {F1_RED};
    }}

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
    }}

    section[data-testid="stSidebar"] .stCaption {{
        color: {F1_GREY};
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 700;
    }}

    /* Sidebar selectbox styling */
    .stSelectbox label {{
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: {F1_GREY} !important;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

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
</style>

<div class="f1-top-bar"></div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# F1 PREDICTOR")
    st.caption("ML race forecasting · 2022-2025")

    st.markdown("---")

    year = st.selectbox(
        "Season",
        options=[2025, 2024, 2023, 2022],
        index=0,
    )

    races = get_available_races(year)
    race_options = {
        f"R{row['Round']:02d} — {row['EventName']}": row["Round"]
        for _, row in races.iterrows()
    }
    selected_race_label = st.selectbox(
        "Race",
        options=list(race_options.keys()),
        index=0,
    )
    selected_round = race_options[selected_race_label]

    st.markdown("---")
    st.markdown("**Model performance**")
    st.markdown(
        f"""
        <div style='font-size: 0.85rem; color: {F1_GREY}; line-height: 1.6;'>
        <strong style='color: {F1_LIGHT};'>2025 Holdout</strong><br>
        Top-1: <strong style='color: {F1_LIGHT};'>58.3%</strong><br>
        Top-3: <strong style='color: #00D26A;'>100%</strong><br>
        RMSE: <strong style='color: {F1_LIGHT};'>4.22</strong>
        <br><br>
        <strong style='color: {F1_LIGHT};'>2024 Test</strong><br>
        Top-3: <strong style='color: {F1_LIGHT};'>78.3%</strong><br>
        RMSE: <strong style='color: {F1_LIGHT};'>3.74</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style='font-size: 0.8rem; color: {F1_GREY};'>
        Built by <a href='https://www.linkedin.com/in/om-patil-nu' style='color: {F1_RED}; text-decoration: none;'>Om Patil</a><br>
        <a href='https://github.com/Om-Ravindra-Patil/F1-Race-Predictor' style='color: {F1_RED}; text-decoration: none;'>GitHub repo</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Main panel — fetch data
# ──────────────────────────────────────────────────────────────────────
metadata = get_race_metadata(year, selected_round)
predictions = predict_race(year, selected_round)


# Identify winners and compute metrics
actual_winner = predictions[predictions["ActualPosition"] == 1].iloc[0]
predicted_winner = predictions[predictions["PredictedRank"] == 1].iloc[0]
winner_correct = actual_winner["Abbreviation"] == predicted_winner["Abbreviation"]

predicted_top3 = predictions[predictions["PredictedRank"] <= 3]["Abbreviation"].tolist()
actual_top3 = predictions[predictions["ActualPosition"] <= 3]["Abbreviation"].tolist()
top3_overlap = len(set(predicted_top3) & set(actual_top3))

valid = predictions.dropna(subset=["ActualPosition"])
race_mae = (valid["ActualPosition"] - valid["PredictedRank"]).abs().mean()


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
winner_color_class = "f1-metric-value-correct" if winner_correct else "f1-metric-value-missed"
winner_status = "PREDICTED" if winner_correct else "MISSED"
top3_color_class = "f1-metric-value-correct" if top3_overlap == 3 else ""

st.markdown(f"""
<div class="f1-metrics-grid">
    <div class="f1-metric-card">
        <div class="f1-metric-label">Actual Winner</div>
        <div class="f1-metric-value">{actual_winner['Abbreviation']}</div>
        <div class="f1-metric-context">{actual_winner['TeamName']}</div>
    </div>
    <div class="f1-metric-card">
        <div class="f1-metric-label">Predicted Winner</div>
        <div class="f1-metric-value {winner_color_class}">{predicted_winner['Abbreviation']}</div>
        <div class="f1-metric-context">{winner_status} · {predicted_winner['TeamName']}</div>
    </div>
    <div class="f1-metric-card">
        <div class="f1-metric-label">Top-3 Overlap</div>
        <div class="f1-metric-value {top3_color_class}">{top3_overlap}/3</div>
        <div class="f1-metric-context">Drivers in correct podium zone</div>
    </div>
    <div class="f1-metric-card">
        <div class="f1-metric-label">Race MAE</div>
        <div class="f1-metric-value">{race_mae:.2f}</div>
        <div class="f1-metric-context">Mean prediction error (positions)</div>
    </div>
</div>
""", unsafe_allow_html=True)


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
    st.markdown(
        render_podium(
            actual_top3_df.iloc[0],
            actual_top3_df.iloc[1],
            actual_top3_df.iloc[2],
            "Actual Podium",
        ),
        unsafe_allow_html=True,
    )


st.markdown('<div class="f1-section-title">PREDICTIONS</div>', unsafe_allow_html=True)


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
    actual = int(row["ActualPosition"]) if not pd.isna(row["ActualPosition"]) else "DNF"
    delta = row["PositionDelta"]

    # Delta styling — green when model was correct/optimistic, red when too pessimistic
    if pd.isna(delta):
        delta_text = "—"
        delta_color = F1_GREY
    else:
        delta_int = int(delta)
        if delta_int == 0:
            delta_text = "—"
            delta_color = F1_GREY
        elif delta_int < 0:
            delta_text = f"{delta_int}"
            delta_color = "#00D26A"
        else:
            delta_text = f"+{delta_int}"
            delta_color = F1_RED

    return (
        f'<div class="f1-pred-row" style="border-left:4px solid {team_color};">'
        f'<div class="f1-pred-cell f1-pred-rank">{pred_rank}</div>'
        f'<div class="f1-pred-cell f1-pred-driver">'
        f'<div class="f1-pred-code">{driver_code}</div>'
        f'<div class="f1-pred-name">{full_name}</div>'
        f'</div>'
        f'<div class="f1-pred-cell f1-pred-team" style="color:{team_color};">{team}</div>'
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
    grid-template-columns: 60px 2fr 2fr 1fr 1fr 1fr;
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
    grid-template-columns: 60px 2fr 2fr 1fr 1fr 1fr;
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