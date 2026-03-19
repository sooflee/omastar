import pandas as pd
import numpy as np

from config import POSSESSION_FTA_FACTOR


def estimate_possessions(row: pd.Series) -> float:
    """Estimate possessions from box score stats.

    Uses the Dean Oliver formula (Basketball on Paper, 2004):
      Poss = FGA - OR + TO + factor * FTA
    where factor ≈ 0.475 accounts for and-ones, technical FTs, etc.
    """
    return row["FGA"] - row["OR"] + row["TO"] + POSSESSION_FTA_FACTOR * row["FTA"]


def compute_team_season_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game-level stats into team-season summaries.

    Args:
        games: Normalized detailed game results (from clean.normalize_detailed_results)
               Must have columns: Season, TeamID, Score, OppScore, Win,
               FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF,
               OppFGM, OppFGA, OppFGM3, OppFGA3, OppFTM, OppFTA, OppOR, OppDR,
               OppAst, OppTO, OppStl, OppBlk, OppPF

    Returns:
        DataFrame with one row per team-season containing aggregated stats.
    """
    stat_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
    opp_stat_cols = [f"Opp{c}" for c in stat_cols]

    # Drop games with missing detailed stats
    required = stat_cols + opp_stat_cols
    games_clean = games.dropna(subset=required)

    # Aggregate per team-season
    agg = games_clean.groupby(["Season", "TeamID"]).agg(
        Games=("Win", "count"),
        Wins=("Win", "sum"),
        TotalScore=("Score", "sum"),
        TotalOppScore=("OppScore", "sum"),
        # Raw box score totals
        **{col: (col, "sum") for col in stat_cols},
        **{col: (col, "sum") for col in opp_stat_cols},
    ).reset_index()

    # Win percentage
    agg["WinPct"] = agg["Wins"] / agg["Games"]

    # Scoring
    agg["PPG"] = agg["TotalScore"] / agg["Games"]
    agg["OppPPG"] = agg["TotalOppScore"] / agg["Games"]
    agg["ScoreMargin"] = agg["PPG"] - agg["OppPPG"]

    # Possessions (Dean Oliver formula, Basketball on Paper 2004)
    agg["Poss"] = agg["FGA"] - agg["OR"] + agg["TO"] + POSSESSION_FTA_FACTOR * agg["FTA"]
    agg["OppPoss"] = (agg["OppFGA"] - agg["OppOR"] + agg["OppTO"]
                      + POSSESSION_FTA_FACTOR * agg["OppFTA"])

    # Efficiency (per 100 possessions)
    agg["OffEff"] = np.where(agg["Poss"] > 0, agg["TotalScore"] / agg["Poss"] * 100, 0)
    agg["DefEff"] = np.where(agg["OppPoss"] > 0, agg["TotalOppScore"] / agg["OppPoss"] * 100, 0)
    agg["NetEff"] = agg["OffEff"] - agg["DefEff"]

    # Tempo (possessions per game)
    agg["Tempo"] = agg["Poss"] / agg["Games"]

    # Four Factors (offense)
    agg["eFGPct"] = np.where(
        agg["FGA"] > 0,
        (agg["FGM"] + 0.5 * agg["FGM3"]) / agg["FGA"],
        0,
    )
    agg["TORate"] = np.where(agg["Poss"] > 0, agg["TO"] / agg["Poss"], 0)
    agg["ORRate"] = np.where(
        (agg["OR"] + agg["OppDR"]) > 0,
        agg["OR"] / (agg["OR"] + agg["OppDR"]),
        0,
    )
    agg["FTRate"] = np.where(agg["FGA"] > 0, agg["FTM"] / agg["FGA"], 0)

    # Four Factors (defense)
    agg["OppeFGPct"] = np.where(
        agg["OppFGA"] > 0,
        (agg["OppFGM"] + 0.5 * agg["OppFGM3"]) / agg["OppFGA"],
        0,
    )
    agg["OppTORate"] = np.where(agg["OppPoss"] > 0, agg["OppTO"] / agg["OppPoss"], 0)
    agg["OppORRate"] = np.where(
        (agg["OppOR"] + agg["DR"]) > 0,
        agg["OppOR"] / (agg["OppOR"] + agg["DR"]),
        0,
    )
    agg["OppFTRate"] = np.where(agg["OppFGA"] > 0, agg["OppFTM"] / agg["OppFGA"], 0)

    # Three-point shooting
    agg["FG3Pct"] = np.where(agg["FGA3"] > 0, agg["FGM3"] / agg["FGA3"], 0)
    agg["OppFG3Pct"] = np.where(agg["OppFGA3"] > 0, agg["OppFGM3"] / agg["OppFGA3"], 0)

    # Assist rate
    agg["AstRate"] = np.where(agg["FGM"] > 0, agg["Ast"] / agg["FGM"], 0)

    return agg


def compute_recent_form(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """Compute stats over each team's last N games of the regular season.

    Returns DataFrame with Season, TeamID, and recent-form features prefixed with 'Recent_'.
    """
    stat_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
    opp_stat_cols = [f"Opp{c}" for c in stat_cols]
    required = stat_cols + opp_stat_cols

    games_clean = games.dropna(subset=required).copy()
    games_clean = games_clean.sort_values(["Season", "TeamID", "DayNum"])

    # Take last N games per team-season
    recent = games_clean.groupby(["Season", "TeamID"]).tail(n_games)

    agg = recent.groupby(["Season", "TeamID"]).agg(
        RecentGames=("Win", "count"),
        RecentWins=("Win", "sum"),
        RecentScore=("Score", "mean"),
        RecentOppScore=("OppScore", "mean"),
    ).reset_index()

    agg["RecentWinPct"] = agg["RecentWins"] / agg["RecentGames"]
    agg["RecentMargin"] = agg["RecentScore"] - agg["RecentOppScore"]

    return agg[["Season", "TeamID", "RecentWinPct", "RecentMargin"]]
