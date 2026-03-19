"""Coach tournament experience features.

Computes per-coach NCAA tournament history (wins, appearances) and maps
to the current team-season via MTeamCoaches.
"""
import pandas as pd
import numpy as np

from src.data import load


def compute_coach_tourney_features() -> pd.DataFrame:
    """Compute coach tournament experience for each team-season.

    For each (Season, TeamID), looks up the coach and counts their
    prior NCAA tournament appearances and wins (excluding the current season).

    Returns:
        DataFrame with Season, TeamID, CoachTourneyWins, CoachTourneyApps.
    """
    coaches = load.load_coaches()
    tourney = load.load_tourney_compact()

    # Build coach-to-team mapping per season.
    # A coach can have multiple stints in a season (mid-season firing);
    # take the coach active at season end (max LastDayNum).
    coach_team = (
        coaches.sort_values("LastDayNum")
        .groupby(["Season", "TeamID"])
        .last()
        .reset_index()[["Season", "TeamID", "CoachName"]]
    )

    # Build per-game coach mapping by merging coach_team with tourney results.
    # For each tournament game, determine the winning and losing coach.
    tourney_w = tourney[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    tourney_l = tourney[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})

    # All tournament appearances (one row per team per game)
    tourney_apps = pd.concat([tourney_w, tourney_l], ignore_index=True)
    tourney_apps = tourney_apps.merge(coach_team, on=["Season", "TeamID"], how="left")

    # Tournament wins
    tourney_wins = tourney_w.merge(coach_team, on=["Season", "TeamID"], how="left")

    # Count per-coach cumulative appearances and wins (seasons prior to current)
    # First, count per coach per season
    apps_per_season = (
        tourney_apps.groupby(["CoachName", "Season"])
        .size()
        .reset_index(name="SeasonApps")
    )
    wins_per_season = (
        tourney_wins.groupby(["CoachName", "Season"])
        .size()
        .reset_index(name="SeasonWins")
    )

    # Merge
    coach_season = apps_per_season.merge(
        wins_per_season, on=["CoachName", "Season"], how="left"
    )
    coach_season["SeasonWins"] = coach_season["SeasonWins"].fillna(0).astype(int)

    # For each coach, compute cumulative stats from PRIOR seasons only
    coach_season = coach_season.sort_values(["CoachName", "Season"])

    coach_season["CumApps"] = coach_season.groupby("CoachName")["SeasonApps"].cumsum()
    coach_season["CumWins"] = coach_season.groupby("CoachName")["SeasonWins"].cumsum()

    # Shift to exclude current season (use only prior seasons)
    coach_season["CoachTourneyApps"] = (
        coach_season.groupby("CoachName")["CumApps"].shift(1).fillna(0).astype(int)
    )
    coach_season["CoachTourneyWins"] = (
        coach_season.groupby("CoachName")["CumWins"].shift(1).fillna(0).astype(int)
    )

    # Build lookup: for each coach, their cumulative stats by season
    coach_history = coach_season[["CoachName", "Season", "CoachTourneyApps", "CoachTourneyWins"]]

    # Now map back to all team-seasons via coach_team
    result = coach_team.merge(coach_history, on=["CoachName", "Season"], how="left")
    result["CoachTourneyApps"] = result["CoachTourneyApps"].fillna(0).astype(int)
    result["CoachTourneyWins"] = result["CoachTourneyWins"].fillna(0).astype(int)

    return result[["Season", "TeamID", "CoachTourneyWins", "CoachTourneyApps"]]
