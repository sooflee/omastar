"""Tests for src/features — team stats, matchup features, seed matchup."""

import numpy as np
import pandas as pd
import pytest

from src.data.team_season import compute_team_season_stats, compute_recent_form
from src.features.matchup import build_matchup_features, build_training_data, DIFF_FEATURES
from src.features.advanced import (
    compute_variance_features,
    compute_clutch_features,
    compute_style_features,
)
from config import POSSESSION_FTA_FACTOR


# ---------------------------------------------------------------------------
# Team-season stats
# ---------------------------------------------------------------------------

class TestTeamSeasonStats:
    def test_output_shape(self, normalized_games):
        result = compute_team_season_stats(normalized_games)
        # Should have one row per (Season, TeamID) combination
        assert len(result) > 0
        assert result.duplicated(subset=["Season", "TeamID"]).sum() == 0

    def test_win_pct_range(self, normalized_games):
        result = compute_team_season_stats(normalized_games)
        assert (result["WinPct"] >= 0).all()
        assert (result["WinPct"] <= 1).all()

    def test_efficiency_positive(self, normalized_games):
        result = compute_team_season_stats(normalized_games)
        assert (result["OffEff"] >= 0).all()
        assert (result["DefEff"] >= 0).all()

    def test_net_eff_equals_off_minus_def(self, normalized_games):
        result = compute_team_season_stats(normalized_games)
        np.testing.assert_allclose(
            result["NetEff"], result["OffEff"] - result["DefEff"], atol=1e-10,
        )

    def test_four_factors_range(self, normalized_games):
        result = compute_team_season_stats(normalized_games)
        for col in ["eFGPct", "TORate", "ORRate", "FTRate"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"
            assert (result[col] <= 1).all(), f"{col} exceeds 1"

    def test_possession_formula_uses_config_constant(self, normalized_games):
        """Possession estimate should use POSSESSION_FTA_FACTOR from config."""
        result = compute_team_season_stats(normalized_games)
        # Re-compute manually for the first row
        row = result.iloc[0]
        expected_poss = row["FGA"] - row["OR"] + row["TO"] + POSSESSION_FTA_FACTOR * row["FTA"]
        assert abs(row["Poss"] - expected_poss) < 1e-6


# ---------------------------------------------------------------------------
# Recent form
# ---------------------------------------------------------------------------

class TestRecentForm:
    def test_output_columns(self, normalized_games):
        result = compute_recent_form(normalized_games, n_games=5)
        assert "RecentWinPct" in result.columns
        assert "RecentMargin" in result.columns

    def test_recent_win_pct_range(self, normalized_games):
        result = compute_recent_form(normalized_games, n_games=5)
        assert (result["RecentWinPct"] >= 0).all()
        assert (result["RecentWinPct"] <= 1).all()


# ---------------------------------------------------------------------------
# Matchup features
# ---------------------------------------------------------------------------

class TestMatchupFeatures:
    def test_diff_features_created(self, normalized_games):
        team_stats = compute_team_season_stats(normalized_games)
        matchups = pd.DataFrame({
            "Season": [2024, 2024],
            "TeamA": [1101, 1101],
            "TeamB": [1102, 1103],
        })
        result = build_matchup_features(team_stats, matchups)
        diff_cols = [c for c in result.columns if c.endswith("_diff")]
        assert len(diff_cols) > 0

    def test_diff_is_a_minus_b(self, normalized_games):
        """Feature diff should equal TeamA value minus TeamB value."""
        team_stats = compute_team_season_stats(normalized_games)
        matchups = pd.DataFrame({
            "Season": [2024],
            "TeamA": [1101],
            "TeamB": [1102],
        })
        result = build_matchup_features(team_stats, matchups)

        a_row = team_stats[
            (team_stats["Season"] == 2024) & (team_stats["TeamID"] == 1101)
        ]
        b_row = team_stats[
            (team_stats["Season"] == 2024) & (team_stats["TeamID"] == 1102)
        ]
        if len(a_row) > 0 and len(b_row) > 0 and "WinPct_diff" in result.columns:
            expected = float(a_row["WinPct"].values[0] - b_row["WinPct"].values[0])
            actual = float(result["WinPct_diff"].values[0])
            assert abs(actual - expected) < 1e-10


class TestTrainingData:
    def test_target_binary(self, normalized_games, tourney_compact):
        team_stats = compute_team_season_stats(normalized_games)
        features, target = build_training_data(team_stats, tourney_compact)
        assert set(target.unique()).issubset({0, 1})

    def test_canonical_ordering(self, normalized_games, tourney_compact):
        """TeamA should always be the lower TeamID."""
        team_stats = compute_team_season_stats(normalized_games)
        features, _ = build_training_data(team_stats, tourney_compact)
        assert (features["TeamA"] < features["TeamB"]).all()


# ---------------------------------------------------------------------------
# Advanced features
# ---------------------------------------------------------------------------

class TestVarianceFeatures:
    def test_output_columns(self, normalized_games):
        result = compute_variance_features(normalized_games)
        for col in ["ScoreStd", "MarginStd", "WorstMargin", "BestMargin"]:
            assert col in result.columns

    def test_worst_le_best(self, normalized_games):
        result = compute_variance_features(normalized_games)
        assert (result["WorstMargin"] <= result["BestMargin"]).all()


class TestClutchFeatures:
    def test_output_columns(self, normalized_games):
        result = compute_clutch_features(normalized_games)
        for col in ["CloseWinPct", "CloseGamePct", "OTWinPct"]:
            assert col in result.columns

    def test_close_win_pct_range(self, normalized_games):
        result = compute_clutch_features(normalized_games)
        assert (result["CloseWinPct"] >= 0).all()
        assert (result["CloseWinPct"] <= 1).all()


class TestStyleFeatures:
    def test_output_columns(self, normalized_games):
        result = compute_style_features(normalized_games)
        for col in ["ThreePtDependence", "BlkRate", "StlRate", "AstTORatio"]:
            assert col in result.columns

    def test_three_pt_dependence_range(self, normalized_games):
        result = compute_style_features(normalized_games)
        assert (result["ThreePtDependence"] >= 0).all()
        assert (result["ThreePtDependence"] <= 1).all()
