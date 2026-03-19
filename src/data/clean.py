import pandas as pd


def parse_seed(seed_str: str) -> int:
    """Extract numeric seed from strings like 'W01', 'X16a', 'Y11b'."""
    return int(seed_str[1:3])


def parse_seed_region(seed_str: str) -> str:
    """Extract region letter from seed string."""
    return seed_str[0]


def add_parsed_seeds(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric seed and region columns to seeds dataframe."""
    df = seeds_df.copy()
    df["SeedNum"] = df["Seed"].apply(parse_seed)
    df["Region"] = df["Seed"].apply(parse_seed_region)
    return df


def build_seed_lookup(seeds_df: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Build a (Season, TeamID) -> numeric seed lookup from a seeds DataFrame.

    This is the single canonical way to map tournament teams to their seed
    numbers. Use this instead of hand-rolling the same loop.

    Args:
        seeds_df: DataFrame with Season, TeamID, Seed columns
                  (e.g. from load.load_tourney_seeds()).

    Returns:
        Dict mapping (season, team_id) to the integer seed (1–16).
    """
    return {
        (int(row["Season"]), int(row["TeamID"])): parse_seed(row["Seed"])
        for _, row in seeds_df.iterrows()
    }


def normalize_detailed_results(df: pd.DataFrame) -> pd.DataFrame:
    """Convert W/L-oriented detailed results into team-perspective rows (vectorized)."""
    stat_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

    loc_map = {"H": "A", "A": "H", "N": "N"}

    # Winner rows
    w_rename = {
        "WTeamID": "TeamID", "LTeamID": "OppID",
        "WScore": "Score", "LScore": "OppScore",
        "WLoc": "Loc",
    }
    for col in stat_cols:
        w_rename[f"W{col}"] = col
        w_rename[f"L{col}"] = f"Opp{col}"

    winners = df.rename(columns=w_rename)[
        ["Season", "DayNum", "NumOT", "TeamID", "OppID", "Score", "OppScore", "Loc"]
        + stat_cols + [f"Opp{c}" for c in stat_cols]
    ].copy()
    winners["Win"] = 1

    # Loser rows
    l_rename = {
        "LTeamID": "TeamID", "WTeamID": "OppID",
        "LScore": "Score", "WScore": "OppScore",
        "WLoc": "Loc",
    }
    for col in stat_cols:
        l_rename[f"L{col}"] = col
        l_rename[f"W{col}"] = f"Opp{col}"

    losers = df.rename(columns=l_rename)[
        ["Season", "DayNum", "NumOT", "TeamID", "OppID", "Score", "OppScore", "Loc"]
        + stat_cols + [f"Opp{c}" for c in stat_cols]
    ].copy()
    losers["Win"] = 0
    losers["Loc"] = losers["Loc"].map(loc_map).fillna("N")

    return pd.concat([winners, losers], ignore_index=True)


def normalize_compact_results(df: pd.DataFrame) -> pd.DataFrame:
    """Convert W/L-oriented compact results into team-perspective rows (vectorized)."""
    loc_map = {"H": "A", "A": "H", "N": "N"}

    winners = pd.DataFrame({
        "Season": df["Season"],
        "DayNum": df["DayNum"],
        "NumOT": df.get("NumOT", 0),
        "TeamID": df["WTeamID"],
        "OppID": df["LTeamID"],
        "Score": df["WScore"],
        "OppScore": df["LScore"],
        "Loc": df["WLoc"] if "WLoc" in df.columns else "N",
        "Win": 1,
    })

    losers = pd.DataFrame({
        "Season": df["Season"],
        "DayNum": df["DayNum"],
        "NumOT": df.get("NumOT", 0),
        "TeamID": df["LTeamID"],
        "OppID": df["WTeamID"],
        "Score": df["LScore"],
        "OppScore": df["WScore"],
        "Loc": df["WLoc"].map(loc_map).fillna("N") if "WLoc" in df.columns else "N",
        "Win": 0,
    })

    return pd.concat([winners, losers], ignore_index=True)
