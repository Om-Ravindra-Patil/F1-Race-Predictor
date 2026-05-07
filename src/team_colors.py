"""
F1 team colours and branding utilities.

Single source of truth for team colours used across the project — charts,
dashboard, podium graphics, and any future visualisations.

Colours are taken from F1's official 2024-2025 team palette where possible,
with adjustments for visual contrast on dark backgrounds.

Note: F1 teams occasionally change their primary colour mid-season or rebrand
year-to-year. Where a team's colour shifted between 2022-2025, we use the
most recognisable variant.
"""

from typing import Optional


# Primary team colours — used for table accents, chart bars, podium graphics
TEAM_COLORS = {
    # 2024-2025 era teams
    "Red Bull Racing": "#3671C6",      # Red Bull blue
    "McLaren": "#FF8000",               # McLaren papaya
    "Ferrari": "#E80020",               # Ferrari red
    "Mercedes": "#27F4D2",              # Mercedes silver-teal
    "Aston Martin": "#229971",          # Aston racing green
    "Alpine": "#0093CC",                # Alpine blue
    "Williams": "#64C4FF",              # Williams light blue
    "Kick Sauber": "#52E252",           # Sauber green (2024-25)
    "Racing Bulls": "#6692FF",          # RB / VCARB blue
    "Haas": "#B6BABD",                  # Haas grey

    # Historical 2022-2023 names (same teams, prior naming)
    "Alfa Romeo": "#900000",            # Alfa dark red (became Sauber)
    "AlphaTauri": "#5E8FAA",            # AlphaTauri navy (became Racing Bulls)
}

# Secondary/dark variant — for backgrounds, subtle accents
TEAM_COLORS_DARK = {
    "Red Bull Racing": "#1E3A8A",
    "McLaren": "#B85D00",
    "Ferrari": "#A50018",
    "Mercedes": "#00A99D",
    "Aston Martin": "#1A6B50",
    "Alpine": "#006B96",
    "Williams": "#4A8FBE",
    "Kick Sauber": "#388E3C",
    "Racing Bulls": "#4768B8",
    "Haas": "#7E8084",
    "Alfa Romeo": "#660000",
    "AlphaTauri": "#3F6680",
}

# Fallback for unknown / new teams
DEFAULT_COLOR = "#888888"
DEFAULT_COLOR_DARK = "#555555"


def get_team_color(team_name: str, dark: bool = False) -> str:
    """Get the primary hex colour for a team.

    Args:
        team_name: Team name as it appears in the dataset
        dark: If True, return the darker secondary variant

    Returns:
        Hex colour string (e.g. '#E80020'). Returns default grey for
        unknown teams.

    Example:
        >>> get_team_color('McLaren')
        '#FF8000'
        >>> get_team_color('Ferrari', dark=True)
        '#A50018'
    """
    if dark:
        return TEAM_COLORS_DARK.get(team_name, DEFAULT_COLOR_DARK)
    return TEAM_COLORS.get(team_name, DEFAULT_COLOR)


def get_text_color_for_team(team_name: str) -> str:
    """Choose readable text colour (black or white) for a given team's background.

    Uses simple luminance check — light colours get black text,
    dark colours get white.
    """
    light_text_teams = {
        "McLaren", "Mercedes", "Williams", "Haas",
        "Kick Sauber", "AlphaTauri",
    }
    return "#000000" if team_name in light_text_teams else "#FFFFFF"
    

def all_teams() -> list[str]:
    """Return a sorted list of all known team names."""
    return sorted(TEAM_COLORS.keys())