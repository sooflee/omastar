import pandas as pd
import numpy as np


# Features to compute as TeamA - TeamB differences
DIFF_FEATURES = [
    "SeedNum",
    "WinPct",
    "NetEff",
    "OffEff",
    "DefEff",
    "Tempo",
    "eFGPct",
    "TORate",
    "ORRate",
    "FTRate",
    "OppeFGPct",
    "OppTORate",
    "OppORRate",
    "OppFTRate",
    "FG3Pct",
    "OppFG3Pct",
    "ScoreMargin",
    "AstRate",
    "SOS",
    "MasseyMean",
    "MasseyMedian",
    "Rank_POM",
    "Rank_SAG",
    "Rank_MOR",
    "Rank_DOL",
    "Rank_COL",
    "Rank_AP",
    "Rank_USA",
    "Rank_WLK",
    "Rank_RTH",
    "RecentWinPct",
    "RecentMargin",
    "AdjO",
    "AdjD",
    "AdjEM",
    "AdjT",
    # External data (real KenPom, 538)
    "RealKAdjO",
    "RealKAdjD",
    "RealKAdjEM",
    "RealKAdjT",
    "BAdjEM",
    "BAdjO",
    "BAdjD",
    "PowerRating538",
    # EvanMiya
    "EvanMiyaRating",
    "RosterRank",
    "KillshotsMargin",
    # Resume / quality metrics
    "Q1Wins",
    "Q2Wins",
    "Elo",
    "WABRank",
    # Coach tournament experience
    "CoachTourneyWins",
    "CoachTourneyApps",
    # Conference tournament performance
    "ConfTourneyWins",
    "WonConfTourney",
    # Advanced: variance/consistency
    "ScoreStd",
    "MarginStd",
    "FG3PctStd",
    "WorstMargin",
    "BestMargin",
    # Advanced: clutch/close-game performance
    "CloseWinPct",
    "CloseGamePct",
    "OTWinPct",
    # Advanced: playing style
    "ThreePtDependence",
    "TwoPtPct",
    "BlkRate",
    "StlRate",
    "OppBlkRate",
    "OppStlRate",
    "AstTORatio",
    "DRRate",
    "OppThreePtDependence",
    # Conference strength
    "ConfStrength",
    "ConfDepth",
    "ConfTopHeavy",
    # Season trajectory
    "MarginTrend",
    "EffTrend",
    "WinTrendLate",
    # Program tournament experience
    "ProgramTourneyApps",
    "ProgramTourneyWins",
    "ProgramDeepRuns",
    # Barttorvik away/neutral-site
    "AwayNeutralAdjEM",
    "AwayNeutralAdjO",
    "AwayNeutralAdjD",
    "Talent",
    "Experience",
    "AvgHeight",
    "EffHeight",
    "EliteSOS",
    # Preseason-to-current improvement
    "PreseasonAdjEM",
    "AdjEMImprovement",
    "PreseasonRank",
    "RankImprovement",
    # Injury / availability
    "InjuryRank",
    # Pre-tournament AP Poll
    "APFinalRank",
    "APFinalVotes",
    # RPPF independent ratings
    "RPPFRating",
    "RPPFAdjEM",
    "RPPFSOS",
    # Shooting splits (shot quality)
    "DunksShare",
    "DunksFGPct",
    "CloseTwosShare",
    "CloseTwosFGPct",
    "ThreesShare",
    "DunksDShare",
    "CloseTwosDFGPct",
    "ThreesDFGPct",
]


def build_matchup_features(
    team_features: pd.DataFrame,
    matchups: pd.DataFrame,
) -> pd.DataFrame:
    """Build matchup-level feature vectors from team-season features.

    For each matchup, computes TeamA - TeamB differences for all features.
    TeamA is always the lower TeamID (Kaggle convention).

    Args:
        team_features: Team-season features with Season, TeamID, and feature columns.
        matchups: DataFrame with Season, TeamA, TeamB columns representing games.

    Returns:
        DataFrame with Season, TeamA, TeamB, and difference features.
    """
    # Merge TeamA features
    merged = matchups.merge(
        team_features,
        left_on=["Season", "TeamA"],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"], errors="ignore")

    # Merge TeamB features
    merged = merged.merge(
        team_features,
        left_on=["Season", "TeamB"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("_A", "_B"),
    ).drop(columns=["TeamID"], errors="ignore")

    # Compute difference features — build all at once to avoid DataFrame fragmentation
    diff_cols = {}
    for feat in DIFF_FEATURES:
        col_a = f"{feat}_A"
        col_b = f"{feat}_B"
        if col_a in merged.columns and col_b in merged.columns:
            diff_cols[f"{feat}_diff"] = merged[col_a].values - merged[col_b].values

    result = merged[["Season", "TeamA", "TeamB"]].copy()
    result = pd.concat([result, pd.DataFrame(diff_cols, index=result.index)], axis=1)

    return result


def build_training_data(
    team_features: pd.DataFrame,
    tourney_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training dataset from historical tournament results.

    Converts tournament results into matchup features with binary target.
    TeamA is always lower TeamID; target is 1 if TeamA won.

    Args:
        team_features: Team-season features.
        tourney_results: Tournament compact results with Season, WTeamID, LTeamID.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    # Create canonical matchup ordering (lower ID = TeamA)
    tr = tourney_results.reset_index(drop=True)
    matchups = pd.DataFrame({
        "Season": tr["Season"],
        "TeamA": np.minimum(tr["WTeamID"], tr["LTeamID"]),
        "TeamB": np.maximum(tr["WTeamID"], tr["LTeamID"]),
    })

    # Target: 1 if TeamA (lower ID) won
    target = (tr["WTeamID"] == matchups["TeamA"]).astype(int)

    features = build_matchup_features(team_features, matchups)

    return features, target
