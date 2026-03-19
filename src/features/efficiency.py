import pandas as pd
import numpy as np


def compute_simple_sos(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute simple strength of schedule: average opponent win percentage.

    Args:
        games: Normalized game results with Season, TeamID, OppID columns.
        team_stats: Team-season stats with Season, TeamID, WinPct columns.

    Returns:
        DataFrame with Season, TeamID, SOS columns.
    """
    opp_wp = games[["Season", "TeamID", "OppID"]].merge(
        team_stats[["Season", "TeamID", "WinPct"]].rename(columns={"TeamID": "OppID", "WinPct": "OppWinPct"}),
        on=["Season", "OppID"],
    )

    sos = opp_wp.groupby(["Season", "TeamID"]).agg(
        SOS=("OppWinPct", "mean"),
    ).reset_index()

    return sos


TOP_SYSTEMS = ["POM", "SAG", "MOR", "DOL", "COL", "AP", "USA", "WLK", "RTH"]


def compute_massey_features(ordinals: pd.DataFrame) -> pd.DataFrame:
    """Compute ranking features from Massey ordinals.

    Uses end-of-season rankings. Extracts individual rankings from top systems
    (POM/KenPom, SAG/Sagarin, etc.) and a composite mean/median.

    Returns:
        DataFrame with Season, TeamID, MasseyMean, MasseyMedian, and
        individual system ranks (Rank_POM, Rank_SAG, etc.).
    """
    # Get the latest ranking day per season
    max_days = ordinals.groupby("Season")["RankingDayNum"].max().reset_index()
    max_days.columns = ["Season", "MaxDay"]

    latest = ordinals.merge(max_days, on="Season")
    latest = latest[latest["RankingDayNum"] == latest["MaxDay"]]

    # Composite across all systems
    composite = latest.groupby(["Season", "TeamID"]).agg(
        MasseyMean=("OrdinalRank", "mean"),
        MasseyMedian=("OrdinalRank", "median"),
    ).reset_index()

    # Individual top systems
    for sys_name in TOP_SYSTEMS:
        sys_data = latest[latest["SystemName"] == sys_name][
            ["Season", "TeamID", "OrdinalRank"]
        ].rename(columns={"OrdinalRank": f"Rank_{sys_name}"})
        composite = composite.merge(sys_data, on=["Season", "TeamID"], how="left")

    return composite
