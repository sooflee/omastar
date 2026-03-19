import numpy as np
import pandas as pd
from itertools import combinations

from src.features.matchup import build_matchup_features
from src.features.seed_matchup import add_seed_matchup_features


def generate_pairwise_probabilities(
    model,
    team_features: pd.DataFrame,
    season: int,
    feature_cols: list[str],
    team_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Generate win probabilities for all possible matchups in a season.

    Args:
        model: Trained model.
        team_features: Team-season features.
        season: Season to generate predictions for.
        feature_cols: Feature column names (must match training order).
        team_ids: Optional list of team IDs to include. If None, uses all
                  tournament teams for the season.

    Returns:
        DataFrame with TeamA, TeamB, ProbA (probability TeamA wins).
    """
    season_features = team_features[team_features["Season"] == season]

    if team_ids is None:
        team_ids = sorted(season_features["TeamID"].unique())

    # All pairwise combinations (lower ID first)
    matchups = pd.DataFrame(
        [(season, a, b) for a, b in combinations(team_ids, 2)],
        columns=["Season", "TeamA", "TeamB"],
    )

    features = build_matchup_features(team_features, matchups)

    # Add seed matchup features if needed
    seed_feats = {"SeedOverperform_diff"}
    if seed_feats & set(feature_cols):
        features = add_seed_matchup_features(features, team_features)

    X = np.nan_to_num(features[feature_cols].values, nan=0.0)

    probs = model.predict_proba(X)[:, 1]
    probs = np.clip(probs, 0.01, 0.99)

    matchups["ProbA"] = probs

    return matchups[["TeamA", "TeamB", "ProbA"]]


def build_prob_lookup(prob_matrix: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Convert probability DataFrame to O(1) lookup dictionary.

    Keys are (lower_id, higher_id) tuples; values are P(lower_id wins).
    """
    lookup = {}
    for row in prob_matrix.itertuples(index=False):
        lookup[(int(row.TeamA), int(row.TeamB))] = float(row.ProbA)
    return lookup


def lookup_prob(prob_lookup: dict[tuple[int, int], float], team_a: int, team_b: int) -> float:
    """O(1) probability lookup from pre-built dictionary."""
    if team_a < team_b:
        return prob_lookup.get((team_a, team_b), 0.5)
    elif team_a > team_b:
        return 1.0 - prob_lookup.get((team_b, team_a), 0.5)
    return 0.5


def get_win_probability(
    prob_matrix: pd.DataFrame,
    team_a: int,
    team_b: int,
) -> float:
    """Look up P(team_a beats team_b) from probability matrix.

    Handles canonical ordering (lower ID = TeamA in the matrix).
    NOTE: For hot loops, use build_prob_lookup() + lookup_prob() instead.
    """
    low, high = min(team_a, team_b), max(team_a, team_b)

    row = prob_matrix[(prob_matrix["TeamA"] == low) & (prob_matrix["TeamB"] == high)]

    if len(row) == 0:
        return 0.5  # fallback for unknown matchup

    prob_low_wins = row["ProbA"].values[0]

    if team_a == low:
        return prob_low_wins
    else:
        return 1 - prob_low_wins
