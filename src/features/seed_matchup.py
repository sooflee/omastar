"""Seed-aware matchup features that capture non-linear seed relationships.

Key features:
- SeedOverperform: How much better/worse a team is than typical for their seed
"""
import pandas as pd
import numpy as np

from src.data import load
from src.data.clean import build_seed_lookup


def _build_seed_quality_baselines(team_features: pd.DataFrame):
    """Compute average AdjEM per seed number per season (using prior seasons).

    Returns dict: (season, seed_num) -> expected AdjEM for that seed.
    """
    df = team_features[["Season", "TeamID", "SeedNum", "AdjEM"]].dropna().copy()
    all_seasons = sorted(df["Season"].unique())

    baselines = {}
    for season in all_seasons:
        prior = df[df["Season"] < season]
        if len(prior) == 0:
            continue
        avgs = prior.groupby("SeedNum")["AdjEM"].mean()
        for seed_num, avg in avgs.items():
            baselines[(season, seed_num)] = avg

    # For prediction season: use all data
    max_season = max(all_seasons)
    avgs = df.groupby("SeedNum")["AdjEM"].mean()
    for seed_num, avg in avgs.items():
        baselines[(max_season + 1, seed_num)] = avg

    return baselines


def add_seed_matchup_features(
    matchup_features: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Add seed-based matchup features to an existing matchup feature DataFrame.

    Adds:
    - SeedOverperform_diff: How much TeamA overperforms their seed vs TeamB.

    Args:
        matchup_features: DataFrame with Season, TeamA, TeamB, and *_diff columns.
        team_features: Team-season features with SeedNum and AdjEM.

    Returns:
        matchup_features with new columns added.
    """
    result = matchup_features.copy()

    seeds_df = load.load_tourney_seeds()
    seed_lookup = build_seed_lookup(seeds_df)

    # --- SeedOverperform ---
    baselines = _build_seed_quality_baselines(team_features)

    overperform_diffs = []
    for _, row in result.iterrows():
        season = row["Season"]

        # TeamA overperformance
        seed_a = seed_lookup.get((season, row["TeamA"]))
        tf_a = team_features[
            (team_features["Season"] == season) & (team_features["TeamID"] == row["TeamA"])
        ]
        if seed_a is not None and len(tf_a) > 0:
            expected_a = baselines.get((season, seed_a), 0)
            actual_a = tf_a["AdjEM"].values[0] if not pd.isna(tf_a["AdjEM"].values[0]) else 0
            over_a = actual_a - expected_a
        else:
            over_a = 0

        # TeamB overperformance
        seed_b = seed_lookup.get((season, row["TeamB"]))
        tf_b = team_features[
            (team_features["Season"] == season) & (team_features["TeamID"] == row["TeamB"])
        ]
        if seed_b is not None and len(tf_b) > 0:
            expected_b = baselines.get((season, seed_b), 0)
            actual_b = tf_b["AdjEM"].values[0] if not pd.isna(tf_b["AdjEM"].values[0]) else 0
            over_b = actual_b - expected_b
        else:
            over_b = 0

        overperform_diffs.append(over_a - over_b)

    result["SeedOverperform_diff"] = overperform_diffs

    return result
