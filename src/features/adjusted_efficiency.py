"""KenPom-style iterative opponent-adjusted efficiency metrics.

Computes adjusted offensive/defensive efficiency, efficiency margin, and tempo
by iteratively accounting for opponent quality. A team that scores 1.1 PPP
against a defense allowing 0.9 PPP is rated higher than one scoring 1.1 PPP
against a defense allowing 1.2 PPP.
"""

import numpy as np
import pandas as pd

from config import FIRST_DETAILED_SEASON, POSSESSION_FTA_FACTOR


def _estimate_game_possessions(row_team: pd.Series, row_opp: pd.Series) -> float:
    """Estimate possessions for one team in a single game."""
    return row_team - row_opp  # placeholder; actual logic uses full formula


def compute_adjusted_efficiency(
    games: pd.DataFrame,
    n_iterations: int = 20,
) -> pd.DataFrame:
    """Compute KenPom-style adjusted efficiency metrics via iterative adjustment.

    Args:
        games: Normalized detailed game results (from clean.normalize_detailed_results).
              Must contain: Season, DayNum, TeamID, OppID, Score, OppScore,
              FGA, FTA, OR, TO, OppFGA, OppFTA, OppOR, OppTO.
        n_iterations: Number of adjustment iterations (default 20).

    Returns:
        DataFrame with columns: Season, TeamID, AdjO, AdjD, AdjEM, AdjT.
    """
    # Filter to regular season and seasons with detailed stats
    df = games[
        (games["DayNum"] <= 132) & (games["Season"] >= FIRST_DETAILED_SEASON)
    ].copy()

    # Estimate possessions per game (Dean Oliver formula, Basketball on Paper 2004)
    df["Poss"] = df["FGA"] - df["OR"] + df["TO"] + POSSESSION_FTA_FACTOR * df["FTA"]
    df["OppPoss"] = (df["OppFGA"] - df["OppOR"] + df["OppTO"]
                     + POSSESSION_FTA_FACTOR * df["OppFTA"])

    # Average possessions per game (use mean of both teams' estimates)
    df["GamePoss"] = (df["Poss"] + df["OppPoss"]) / 2.0

    # Clamp possessions to a minimum to avoid division by zero
    df["GamePoss"] = df["GamePoss"].clip(lower=1.0)

    # Per-game raw efficiency (points per 100 possessions)
    df["GameOffEff"] = df["Score"] / df["GamePoss"] * 100.0
    df["GameDefEff"] = df["OppScore"] / df["GamePoss"] * 100.0

    # Compute season-level averages as baselines
    season_avg = df.groupby("Season").agg(
        AvgOffEff=("GameOffEff", "mean"),
        AvgDefEff=("GameDefEff", "mean"),
        AvgTempo=("GamePoss", "mean"),
    ).reset_index()

    # Merge season averages onto game rows
    df = df.merge(season_avg, on="Season", how="left")

    # Initialize per-team adjusted values as raw season averages
    team_raw = df.groupby(["Season", "TeamID"]).agg(
        RawAdjO=("GameOffEff", "mean"),
        RawAdjD=("GameDefEff", "mean"),
        RawTempo=("GamePoss", "mean"),
    ).reset_index()

    adj = team_raw.rename(columns={
        "RawAdjO": "AdjO",
        "RawAdjD": "AdjD",
        "RawTempo": "AdjT",
    }).copy()

    # Build a lookup key for fast vectorized merging
    # We need: for each game row, the opponent's current AdjO, AdjD, AdjT
    # Keep game-level data with Season, TeamID, OppID, GameOffEff, GameDefEff, GamePoss, AvgOffEff, AvgDefEff, AvgTempo
    game_data = df[["Season", "TeamID", "OppID", "GameOffEff", "GameDefEff",
                     "GamePoss", "AvgOffEff", "AvgDefEff", "AvgTempo"]].copy()

    for _ in range(n_iterations):
        # Merge opponent's current adjusted ratings onto each game
        opp_adj = adj.rename(columns={
            "TeamID": "OppID",
            "AdjO": "OppAdjO",
            "AdjD": "OppAdjD",
            "AdjT": "OppAdjT",
        })

        gm = game_data.merge(opp_adj, on=["Season", "OppID"], how="left")

        # Adjustment factors:
        #   AdjO for team = mean over games of: GameOffEff * (AvgDefEff / OppAdjD)
        #   This boosts teams that scored well against tough defenses
        #   and penalizes teams that scored well only against weak defenses.
        #
        #   AdjD for team = mean over games of: GameDefEff * (AvgOffEff / OppAdjO)
        #   This rewards teams that held good offenses to low scoring.

        # Protect against division by zero
        opp_adj_d_safe = gm["OppAdjD"].clip(lower=1.0)
        opp_adj_o_safe = gm["OppAdjO"].clip(lower=1.0)
        opp_adj_t_safe = gm["OppAdjT"].clip(lower=1.0)

        gm["AdjGameOffEff"] = gm["GameOffEff"] * (gm["AvgDefEff"] / opp_adj_d_safe)
        gm["AdjGameDefEff"] = gm["GameDefEff"] * (gm["AvgOffEff"] / opp_adj_o_safe)
        gm["AdjGameTempo"] = gm["GamePoss"] * (gm["AvgTempo"] / opp_adj_t_safe)

        # Aggregate to new team-season adjusted values
        new_adj = gm.groupby(["Season", "TeamID"]).agg(
            AdjO=("AdjGameOffEff", "mean"),
            AdjD=("AdjGameDefEff", "mean"),
            AdjT=("AdjGameTempo", "mean"),
        ).reset_index()

        adj = new_adj

    # Compute efficiency margin
    adj["AdjEM"] = adj["AdjO"] - adj["AdjD"]

    return adj[["Season", "TeamID", "AdjO", "AdjD", "AdjEM", "AdjT"]]
