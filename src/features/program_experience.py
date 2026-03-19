"""Program (school) tournament experience features.

Unlike coach experience (which follows the coach), this tracks the program's
recent tournament history. Programs with deep runs in recent years tend to
have institutional advantages: recruiting, fan support, player expectations.
"""
import pandas as pd
import numpy as np

from src.data import load


def compute_program_tourney_features(lookback: int = 5) -> pd.DataFrame:
    """Compute program-level tournament experience features.

    For each team-season, counts tournament appearances and wins in the
    prior N seasons (no data leakage — excludes current season).

    Args:
        lookback: Number of prior seasons to consider.

    Returns:
        DataFrame with Season, TeamID, and:
        - ProgramTourneyApps: number of tournament appearances in last N years
        - ProgramTourneyWins: total tournament wins in last N years
        - ProgramDeepRuns: number of Sweet 16+ appearances in last N years
    """
    tourney = load.load_tourney_compact()
    seeds = load.load_tourney_seeds()

    # All tournament appearances by season (having a seed = made tournament)
    appearances = seeds[["Season", "TeamID"]].drop_duplicates()

    # Wins per team per season
    wins = tourney.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins")
    wins = wins.rename(columns={"WTeamID": "TeamID"})

    # Deep runs: 3+ wins in a tournament = Sweet 16 or better
    wins["DeepRun"] = (wins["Wins"] >= 3).astype(int)

    # Get all team-seasons we need features for
    all_seasons = sorted(appearances["Season"].unique())
    results = []

    for season in all_seasons:
        # Look at prior seasons only
        prior_start = season - lookback
        prior_mask = (appearances["Season"] >= prior_start) & (appearances["Season"] < season)

        # Count appearances in lookback window
        prior_apps = appearances[prior_mask].groupby("TeamID").size().reset_index(name="ProgramTourneyApps")

        # Count wins in lookback window
        prior_wins_mask = (wins["Season"] >= prior_start) & (wins["Season"] < season)
        prior_wins = wins[prior_wins_mask].groupby("TeamID").agg(
            ProgramTourneyWins=("Wins", "sum"),
            ProgramDeepRuns=("DeepRun", "sum"),
        ).reset_index()

        # Teams in this season's tournament
        season_teams = appearances[appearances["Season"] == season][["TeamID"]].copy()
        season_teams["Season"] = season

        season_teams = season_teams.merge(prior_apps, on="TeamID", how="left")
        season_teams = season_teams.merge(prior_wins, on="TeamID", how="left")

        season_teams["ProgramTourneyApps"] = season_teams["ProgramTourneyApps"].fillna(0).astype(int)
        season_teams["ProgramTourneyWins"] = season_teams["ProgramTourneyWins"].fillna(0).astype(int)
        season_teams["ProgramDeepRuns"] = season_teams["ProgramDeepRuns"].fillna(0).astype(int)

        results.append(season_teams)

    return pd.concat(results, ignore_index=True)[
        ["Season", "TeamID", "ProgramTourneyApps", "ProgramTourneyWins", "ProgramDeepRuns"]
    ]
