"""Advanced features derived from game-level box scores.

These capture aspects of team performance that seed and aggregate stats miss:
- Variance/consistency (upset potential)
- Clutch/close-game performance
- Playing style composition (3PT dependence, defensive style)
"""
import pandas as pd
import numpy as np

from config import POSSESSION_FTA_FACTOR


def compute_variance_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute game-to-game variance features per team-season.

    High-variance teams are more upset-prone (and more likely to pull upsets).

    Args:
        games: Normalized game results with Score, OppScore, FGM3, FGA3, FGA per game.

    Returns:
        DataFrame with Season, TeamID, and variance features:
        - ScoreStd: std dev of points scored
        - MarginStd: std dev of scoring margin (consistency)
        - FG3PctStd: std dev of per-game 3PT% (shooting reliability)
        - WorstMargin: worst single-game margin (floor)
        - BestMargin: best single-game margin (ceiling)
    """
    g = games.copy()
    g["Margin"] = g["Score"] - g["OppScore"]
    g["GameFG3Pct"] = np.where(g["FGA3"] > 0, g["FGM3"] / g["FGA3"], np.nan)

    agg = g.groupby(["Season", "TeamID"]).agg(
        ScoreStd=("Score", "std"),
        MarginStd=("Margin", "std"),
        FG3PctStd=("GameFG3Pct", "std"),
        WorstMargin=("Margin", "min"),
        BestMargin=("Margin", "max"),
    ).reset_index()

    # Fill NaN std (teams with 1 game) with 0
    for col in ["ScoreStd", "MarginStd", "FG3PctStd"]:
        agg[col] = agg[col].fillna(0)

    return agg


def compute_clutch_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute close-game and pressure performance features.

    Tournament games are disproportionately close. Teams that perform well
    in close games tend to outperform their seed.

    Args:
        games: Normalized game results with Score, OppScore, Win.

    Returns:
        DataFrame with Season, TeamID, and:
        - CloseWinPct: win rate in games decided by <= 5 points
        - CloseGamePct: fraction of games that are close (exposure to variance)
        - ComebackRate: win rate when trailing at half (not available, use OT proxy)
        - OTRecord: win rate in overtime games
    """
    g = games.copy()
    g["Margin"] = (g["Score"] - g["OppScore"]).abs()
    g["IsClose"] = g["Margin"] <= 5

    # Close game stats
    close = g[g["IsClose"]].groupby(["Season", "TeamID"]).agg(
        CloseWins=("Win", "sum"),
        CloseGames=("Win", "count"),
    ).reset_index()
    close["CloseWinPct"] = np.where(
        close["CloseGames"] > 0,
        close["CloseWins"] / close["CloseGames"],
        0.5,  # default for teams with no close games
    )

    # What fraction of games are close
    total = g.groupby(["Season", "TeamID"]).agg(
        TotalGames=("Win", "count"),
    ).reset_index()

    close = close.merge(total, on=["Season", "TeamID"], how="right")
    close["CloseGames"] = close["CloseGames"].fillna(0)
    close["CloseWins"] = close["CloseWins"].fillna(0)
    close["CloseWinPct"] = close["CloseWinPct"].fillna(0.5)
    close["CloseGamePct"] = np.where(
        close["TotalGames"] > 0,
        close["CloseGames"] / close["TotalGames"],
        0,
    )

    # OT record
    if "NumOT" in g.columns:
        ot_games = g[g["NumOT"] > 0]
        ot_agg = ot_games.groupby(["Season", "TeamID"]).agg(
            OTWins=("Win", "sum"),
            OTGames=("Win", "count"),
        ).reset_index()
        ot_agg["OTWinPct"] = np.where(
            ot_agg["OTGames"] > 0,
            ot_agg["OTWins"] / ot_agg["OTGames"],
            0.5,
        )
        close = close.merge(ot_agg[["Season", "TeamID", "OTWinPct"]], on=["Season", "TeamID"], how="left")
        close["OTWinPct"] = close["OTWinPct"].fillna(0.5)
    else:
        close["OTWinPct"] = 0.5

    return close[["Season", "TeamID", "CloseWinPct", "CloseGamePct", "OTWinPct"]]


def compute_style_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute playing style features that capture how a team wins.

    Two teams with identical efficiency can have very different styles,
    and style matchups matter in tournaments.

    Args:
        games: Normalized game results with full box score stats.

    Returns:
        DataFrame with Season, TeamID, and:
        - ThreePtDependence: fraction of points from 3PT (high = volatile)
        - TwoPtPct: 2-point FG% (interior scoring ability)
        - BlkRate: blocks per defensive possession
        - StlRate: steals per defensive possession
        - OppBlkRate: opponent blocks per offensive possession
        - OppStlRate: opponent steals per offensive possession
        - AstTORatio: assist-to-turnover ratio (ball security)
        - DRRate: defensive rebound rate (board control)
    """
    agg = games.groupby(["Season", "TeamID"]).agg(
        TotalScore=("Score", "sum"),
        FGM3=("FGM3", "sum"),
        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),
        FGA3=("FGA3", "sum"),
        Ast=("Ast", "sum"),
        TO=("TO", "sum"),
        Blk=("Blk", "sum"),
        Stl=("Stl", "sum"),
        DR=("DR", "sum"),
        OR_=("OR", "sum"),
        OppFGA=("OppFGA", "sum"),
        OppFGA3=("OppFGA3", "sum"),
        OppFGM3=("OppFGM3", "sum"),
        OppFGM=("OppFGM", "sum"),
        OppFTA=("OppFTA", "sum"),
        OppOR=("OppOR", "sum"),
        OppTO=("OppTO", "sum"),
        OppBlk=("OppBlk", "sum"),
        OppStl=("OppStl", "sum"),
    ).reset_index()

    # 3PT dependence: what fraction of points come from threes
    agg["ThreePtDependence"] = np.where(
        agg["TotalScore"] > 0,
        (3 * agg["FGM3"]) / agg["TotalScore"],
        0,
    )

    # 2PT FG% (separate from eFG which blends 2PT and 3PT)
    fg2a = agg["FGA"] - agg["FGA3"]
    fg2m = agg["FGM"] - agg["FGM3"]
    agg["TwoPtPct"] = np.where(fg2a > 0, fg2m / fg2a, 0)

    # Opponent possessions (for defensive rate stats)
    # Dean Oliver formula: Poss ≈ FGA − OR + TO + 0.475 × FTA
    opp_poss = (agg["OppFGA"] - agg["OppOR"] + agg["OppTO"]
                + POSSESSION_FTA_FACTOR * agg["OppFTA"])
    opp_poss = np.maximum(opp_poss, 1)  # avoid division by zero

    # Own possessions — use team's own FTA (not opponent's)
    own_fta = games.groupby(["Season", "TeamID"])["FTA"].sum().reset_index(name="OwnFTA")
    agg = agg.merge(own_fta, on=["Season", "TeamID"], how="left")
    own_poss = (agg["FGA"] - agg["OR_"] + agg["TO"]
                + POSSESSION_FTA_FACTOR * agg["OwnFTA"])
    own_poss = np.maximum(own_poss, 1)

    # Defensive rates (per opponent possession)
    agg["BlkRate"] = agg["Blk"] / opp_poss
    agg["StlRate"] = agg["Stl"] / opp_poss

    # How often our possessions get blocked/stolen (opponent defensive rates against us)
    agg["OppBlkRate"] = agg["OppBlk"] / own_poss
    agg["OppStlRate"] = agg["OppStl"] / own_poss

    # Assist-to-turnover ratio
    agg["AstTORatio"] = np.where(agg["TO"] > 0, agg["Ast"] / agg["TO"], 0)

    # Defensive rebound rate
    agg["DRRate"] = np.where(
        (agg["DR"] + agg["OppOR"]) > 0,
        agg["DR"] / (agg["DR"] + agg["OppOR"]),
        0,
    )

    # Opponent 3PT% (already have OppFG3Pct but let's add opponent 3PT dependence)
    opp_score = games.groupby(["Season", "TeamID"])["OppScore"].sum().reset_index(name="TotalOppScore")
    agg = agg.merge(opp_score, on=["Season", "TeamID"], how="left")
    agg["OppThreePtDependence"] = np.where(
        agg["TotalOppScore"] > 0,
        (3 * agg["OppFGM3"]) / agg["TotalOppScore"],
        0,
    )

    keep = ["Season", "TeamID", "ThreePtDependence", "TwoPtPct",
            "BlkRate", "StlRate", "OppBlkRate", "OppStlRate",
            "AstTORatio", "DRRate", "OppThreePtDependence"]
    return agg[keep]


def compute_all_advanced_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute all advanced features and merge into a single DataFrame."""
    variance = compute_variance_features(games)
    clutch = compute_clutch_features(games)
    style = compute_style_features(games)

    result = variance.merge(clutch, on=["Season", "TeamID"], how="outer")
    result = result.merge(style, on=["Season", "TeamID"], how="outer")
    return result
