import logging

import pandas as pd

from src.data import load, clean

logger = logging.getLogger(__name__)
from src.data.team_season import compute_team_season_stats, compute_recent_form
from src.features.efficiency import compute_simple_sos, compute_massey_features
from src.features.adjusted_efficiency import compute_adjusted_efficiency
from src.features.seed import get_seed_features
from src.features.external import load_all_external
from src.features.coach import compute_coach_tourney_features
from src.features.conference_tourney import compute_conf_tourney_features
from src.features.advanced import compute_all_advanced_features
from src.features.conference_strength import compute_conference_strength
from src.features.trajectory import compute_trajectory_features
from src.features.program_experience import compute_program_tourney_features
from src.features.matchup import build_training_data, build_matchup_features, DIFF_FEATURES
from src.features.seed_matchup import add_seed_matchup_features

from config import FIRST_DETAILED_SEASON


def build_team_features(
    reg_season_detailed: pd.DataFrame | None = None,
    seeds: pd.DataFrame | None = None,
    ordinals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build complete team-season feature table.

    Loads data if not provided, computes all features, and merges into
    a single team-season table.

    Returns:
        DataFrame with Season, TeamID, and all computed features.
    """
    # Load data if not provided
    if reg_season_detailed is None:
        reg_season_detailed = load.load_regular_season_detailed()
    if seeds is None:
        seeds = load.load_tourney_seeds()
    if ordinals is None:
        ordinals = load.load_massey_ordinals()

    # Filter to seasons with detailed stats
    reg_season_detailed = reg_season_detailed[
        reg_season_detailed["Season"] >= FIRST_DETAILED_SEASON
    ]

    # Normalize to team-perspective rows
    games = clean.normalize_detailed_results(reg_season_detailed)

    # Filter to regular season only (DayNum <= 132)
    reg_games = games[games["DayNum"] <= 132]

    # Team-season aggregate stats
    team_stats = compute_team_season_stats(reg_games)

    # Recent form (last 10 games)
    recent = compute_recent_form(reg_games, n_games=10)

    # Opponent-adjusted efficiency (KenPom-style)
    adj_eff = compute_adjusted_efficiency(games)

    # Strength of schedule
    sos = compute_simple_sos(reg_games, team_stats)

    # Massey ordinal rankings
    massey = compute_massey_features(ordinals)

    # Seed features (only for tournament teams)
    seed_feats = get_seed_features(seeds)

    # Coach tournament experience
    coach_feats = compute_coach_tourney_features()

    # Conference tournament performance
    conf_tourney_feats = compute_conf_tourney_features()

    # Advanced features (variance, clutch, style)
    advanced_feats = compute_all_advanced_features(reg_games)

    # Season trajectory (improvement/decline over season)
    trajectory_feats = compute_trajectory_features(reg_games)

    # Program tournament experience (school-level, not coach-level)
    program_feats = compute_program_tourney_features()

    # Merge all features
    features = team_stats[["Season", "TeamID", "Games", "WinPct", "PPG", "OppPPG",
                           "ScoreMargin", "OffEff", "DefEff", "NetEff", "Tempo",
                           "eFGPct", "TORate", "ORRate", "FTRate",
                           "OppeFGPct", "OppTORate", "OppORRate", "OppFTRate",
                           "FG3Pct", "OppFG3Pct", "AstRate"]].copy()

    features = features.merge(recent, on=["Season", "TeamID"], how="left")
    features = features.merge(adj_eff, on=["Season", "TeamID"], how="left")
    features = features.merge(sos, on=["Season", "TeamID"], how="left")
    features = features.merge(massey, on=["Season", "TeamID"], how="left")
    features = features.merge(seed_feats, on=["Season", "TeamID"], how="left")
    features = features.merge(coach_feats, on=["Season", "TeamID"], how="left")
    features = features.merge(conf_tourney_feats, on=["Season", "TeamID"], how="left")
    features = features.merge(advanced_feats, on=["Season", "TeamID"], how="left")
    features = features.merge(trajectory_feats, on=["Season", "TeamID"], how="left")
    features = features.merge(program_feats, on=["Season", "TeamID"], how="left")

    # Conference strength (needs AdjEM, so compute after adj_eff merge)
    conf_strength_feats = compute_conference_strength(features)
    features = features.merge(conf_strength_feats, on=["Season", "TeamID"], how="left")

    # Fill NaNs with 0 for features that may be missing for some teams
    fill_zero_cols = [
        "CoachTourneyWins", "CoachTourneyApps", "ConfTourneyWins", "WonConfTourney",
        "ProgramTourneyApps", "ProgramTourneyWins", "ProgramDeepRuns",
        "ConfStrength", "ConfDepth", "ConfTopHeavy",
        "MarginTrend", "EffTrend", "WinTrendLate",
    ]
    for col in fill_zero_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # External data (real KenPom, 538, EvanMiya, Resumes, etc.)
    teams = load.load_teams()
    try:
        external = load_all_external(teams)
        if external is not None and len(external) > 0:
            features = features.merge(external, on=["Season", "TeamID"], how="left")
    except Exception as e:
        logger.warning("  Could not load external data: %s", e)

    logger.info("  Team features built: %d team-seasons, %d columns",
                len(features), len(features.columns))
    return features


def build_full_training_set() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build complete training dataset for model training.

    Returns:
        Tuple of (features DataFrame, target Series, feature column names).
    """
    # Build team features
    team_features = build_team_features()

    # Load tournament results
    tourney = load.load_tourney_compact()
    tourney = tourney[tourney["Season"] >= FIRST_DETAILED_SEASON]

    # Build matchup features + target
    features, target = build_training_data(team_features, tourney)

    # Add seed matchup features (non-linear seed relationships)
    features = add_seed_matchup_features(features, team_features)

    # Get feature columns (all _diff columns)
    feature_cols = [c for c in features.columns if c.endswith("_diff")]

    # Filter to core features if specified
    from src.model.train import CORE_FEATURES
    if CORE_FEATURES is not None:
        available_core = [f for f in CORE_FEATURES if f in feature_cols]
        if available_core:
            feature_cols = available_core

    return features, target, feature_cols
