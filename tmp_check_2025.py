import pandas as pd

df = pd.read_csv("data/processed/features_2022_2025.csv")
print(f"Total rows: {len(df)}")
print(f"By year: {dict(df['Year'].value_counts().sort_index())}")

print(f"\n=== 2025 race winners ===")
winners_2025 = df[(df["Year"] == 2025) & (df["Position"] == 1)][
    ["Round", "EventName", "Abbreviation", "TeamName", "QualifyingPosition", "GridPosition"]
].sort_values("Round")
print(winners_2025.to_string(index=False))

print(f"\n=== 2025 unique teams ({df[df['Year'] == 2025]['TeamName'].nunique()}) ===")
print(sorted(df[df['Year'] == 2025]['TeamName'].unique()))

print(f"\n=== 2025 unique drivers ({df[df['Year'] == 2025]['Abbreviation'].nunique()}) ===")
print(sorted(df[df['Year'] == 2025]['Abbreviation'].unique()))

print(f"\n=== Win count by driver in 2025 ===")
print(df[(df["Year"] == 2025) & (df["Position"] == 1)]["Abbreviation"].value_counts())

print(f"\n=== 2025 pole→win conversion rate (sanity check vs 2024) ===")
for year in [2024, 2025]:
    year_df = df[df["Year"] == year].copy()
    pole_won = ((year_df["QualifyingPosition"] == 1) & (year_df["Position"] == 1)).sum()
    total_races = year_df["Round"].nunique()
    print(f"  {year}: {pole_won}/{total_races} races won by pole sitter ({pole_won/total_races*100:.1f}%)")

print(f"\n=== NaN counts in 2025 features (checking for data quality issues) ===")
features = [
    "GridPosition", "QualifyingPosition", "QualifyingGapToPole",
    "DriverFormLast3", "TeamFormLast3", "QualiGapZScore",
    "PoleToP2Gap", "RaceVsQualiPace", "HasGridPenalty"
]
for col in features:
    nan_2025 = df[df["Year"] == 2025][col].isna().sum()
    print(f"  {col}: {nan_2025} NaN")