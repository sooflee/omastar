import pandas as pd
from src.data.clean import add_parsed_seeds


def get_seed_features(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric seed features from tournament seeds.

    Args:
        seeds_df: MNCAATourneySeeds with Season, Seed, TeamID.

    Returns:
        DataFrame with Season, TeamID, SeedNum, Region.
    """
    parsed = add_parsed_seeds(seeds_df)
    return parsed[["Season", "TeamID", "SeedNum", "Region"]]
