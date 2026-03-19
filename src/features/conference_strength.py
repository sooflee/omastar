"""Conference strength features.

Computes the average quality of teams in each conference, then assigns
that as a feature for each team. Mid-major teams with high seeds often
underperform because their conference strength is weak.
"""
import pandas as pd
import numpy as np

from src.data import load


def compute_conference_strength(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute conference-level strength metrics for each team-season.

    Uses team-level AdjEM (or NetEff fallback) to compute the average
    quality of each conference, then assigns that to each team.

    Args:
        team_stats: Team-season features with Season, TeamID, and
            at least one of AdjEM or NetEff.

    Returns:
        DataFrame with Season, TeamID, ConfStrength, ConfDepth, ConfTopHeavy.
    """
    conferences = load.load_conferences()

    # Merge conference assignments onto team stats
    merged = team_stats[["Season", "TeamID"]].merge(
        conferences, on=["Season", "TeamID"], how="left",
    )

    # Use AdjEM if available, fall back to NetEff
    if "AdjEM" in team_stats.columns:
        quality_col = "AdjEM"
    elif "NetEff" in team_stats.columns:
        quality_col = "NetEff"
    else:
        # Can't compute without a quality metric
        result = team_stats[["Season", "TeamID"]].copy()
        result["ConfStrength"] = 0.0
        result["ConfDepth"] = 0.0
        result["ConfTopHeavy"] = 0.0
        return result

    merged = merged.merge(
        team_stats[["Season", "TeamID", quality_col]],
        on=["Season", "TeamID"],
        how="left",
    )

    # Drop teams without conference or quality data
    valid = merged.dropna(subset=["ConfAbbrev", quality_col])

    # Conference-level aggregates
    conf_agg = valid.groupby(["Season", "ConfAbbrev"]).agg(
        ConfMeanQuality=(quality_col, "mean"),
        ConfMedianQuality=(quality_col, "median"),
        ConfMaxQuality=(quality_col, "max"),
        ConfStdQuality=(quality_col, "std"),
        ConfSize=(quality_col, "count"),
    ).reset_index()
    conf_agg["ConfStdQuality"] = conf_agg["ConfStdQuality"].fillna(0)

    # ConfStrength: mean quality of conference
    # ConfDepth: number of above-average teams (positive AdjEM) in conference
    above_avg = valid[valid[quality_col] > 0].groupby(
        ["Season", "ConfAbbrev"]
    ).size().reset_index(name="ConfAboveAvgTeams")
    conf_agg = conf_agg.merge(above_avg, on=["Season", "ConfAbbrev"], how="left")
    conf_agg["ConfAboveAvgTeams"] = conf_agg["ConfAboveAvgTeams"].fillna(0)

    # Map back to teams
    result = merged[["Season", "TeamID", "ConfAbbrev"]].merge(
        conf_agg, on=["Season", "ConfAbbrev"], how="left",
    )

    # ConfStrength: mean conference quality
    result["ConfStrength"] = result["ConfMeanQuality"].fillna(0)

    # ConfDepth: count of quality teams (above-average) in the conference
    result["ConfDepth"] = result["ConfAboveAvgTeams"].fillna(0)

    # ConfTopHeavy: max - median (conferences where one team is way better)
    result["ConfTopHeavy"] = (
        result["ConfMaxQuality"] - result["ConfMedianQuality"]
    ).fillna(0)

    return result[["Season", "TeamID", "ConfStrength", "ConfDepth", "ConfTopHeavy"]]
