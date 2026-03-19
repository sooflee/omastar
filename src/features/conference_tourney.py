"""Conference tournament performance features.

Computes how each team performed in their conference tournament:
wins, losses, and whether they won it.
"""
import pandas as pd

from src.data import load


def compute_conf_tourney_features() -> pd.DataFrame:
    """Compute conference tournament performance for each team-season.

    Returns:
        DataFrame with Season, TeamID, ConfTourneyWins, WonConfTourney.
    """
    ct = load.load_conference_tourney_games()

    # Count wins per team-season
    wins = (
        ct.groupby(["Season", "WTeamID"])
        .size()
        .reset_index(name="ConfTourneyWins")
        .rename(columns={"WTeamID": "TeamID"})
    )

    # Identify conference tournament champions:
    # the team that won the last game in each conference tournament
    last_game = ct.sort_values("DayNum").groupby(["Season", "ConfAbbrev"]).last().reset_index()
    champs = last_game[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    champs["WonConfTourney"] = 1

    # Merge wins and championship flag
    # Start with all teams that appeared in conference tourney (winners or losers)
    all_w = ct[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    all_l = ct[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})
    all_teams = pd.concat([all_w, all_l]).drop_duplicates()

    result = all_teams.merge(wins, on=["Season", "TeamID"], how="left")
    result["ConfTourneyWins"] = result["ConfTourneyWins"].fillna(0).astype(int)

    result = result.merge(champs, on=["Season", "TeamID"], how="left")
    result["WonConfTourney"] = result["WonConfTourney"].fillna(0).astype(int)

    return result[["Season", "TeamID", "ConfTourneyWins", "WonConfTourney"]]
