"""Season trajectory features.

Captures whether a team is improving or declining over the course of the
season. A team peaking in March is different from one that peaked in December.
"""
import pandas as pd
import numpy as np

from config import POSSESSION_FTA_FACTOR


def compute_trajectory_features(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute season trajectory (trend) features per team-season.

    Fits a linear slope to rolling efficiency/margin over the regular season
    to capture whether a team is improving or declining.

    Args:
        games: Normalized game results (team-perspective) with DayNum, Score,
            OppScore, and box score stats. Should be regular season only.
        window: Rolling window size for smoothing before slope calculation.

    Returns:
        DataFrame with Season, TeamID, and:
        - MarginTrend: slope of scoring margin over the season
        - EffTrend: slope of per-game efficiency over the season
        - WinTrendLate: win rate in last 1/3 of season minus first 1/3
    """
    g = games.copy()
    g["Margin"] = g["Score"] - g["OppScore"]

    # Per-game efficiency (points per 100 possessions)
    poss = (g["FGA"] - g["OR"] + g["TO"] + POSSESSION_FTA_FACTOR * g["FTA"])
    poss = np.maximum(poss, 1)
    g["GameEff"] = (g["Score"] / poss) * 100

    # Sort by game order within each team-season
    g = g.sort_values(["Season", "TeamID", "DayNum"])

    results = []

    for (season, team_id), team_games in g.groupby(["Season", "TeamID"]):
        n = len(team_games)
        if n < 5:
            results.append({
                "Season": season,
                "TeamID": team_id,
                "MarginTrend": 0.0,
                "EffTrend": 0.0,
                "WinTrendLate": 0.0,
            })
            continue

        # Normalize game index to [0, 1] for comparable slopes
        x = np.arange(n, dtype=float) / max(n - 1, 1)
        margins = team_games["Margin"].values.astype(float)
        effs = team_games["GameEff"].values.astype(float)
        wins = team_games["Win"].values.astype(float)

        # Linear regression slope (normalized)
        margin_slope = _slope(x, margins)
        eff_slope = _slope(x, effs)

        # Late vs early win rate (last third minus first third)
        third = max(n // 3, 1)
        early_wr = wins[:third].mean()
        late_wr = wins[-third:].mean()
        win_trend = late_wr - early_wr

        results.append({
            "Season": season,
            "TeamID": team_id,
            "MarginTrend": round(margin_slope, 4),
            "EffTrend": round(eff_slope, 4),
            "WinTrendLate": round(win_trend, 4),
        })

    return pd.DataFrame(results)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute OLS slope of y on x."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    denom = (x * x).sum() - n * x_mean * x_mean
    if abs(denom) < 1e-10:
        return 0.0
    return float(((x * y).sum() - n * x_mean * y_mean) / denom)
