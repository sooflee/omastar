"""Tests for src/data — loading, cleaning, normalization."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    parse_seed,
    parse_seed_region,
    add_parsed_seeds,
    normalize_detailed_results,
    normalize_compact_results,
    build_seed_lookup,
)
from src.data.load import DataValidationError


# ---------------------------------------------------------------------------
# parse_seed / parse_seed_region
# ---------------------------------------------------------------------------

class TestParseSeed:
    def test_basic(self):
        assert parse_seed("W01") == 1
        assert parse_seed("X16") == 16
        assert parse_seed("Y08") == 8

    def test_play_in_suffix(self):
        """Play-in seeds have a letter suffix (e.g. 'X16a')."""
        assert parse_seed("X16a") == 16
        assert parse_seed("Y11b") == 11

    def test_region(self):
        assert parse_seed_region("W01") == "W"
        assert parse_seed_region("Z16a") == "Z"


# ---------------------------------------------------------------------------
# add_parsed_seeds
# ---------------------------------------------------------------------------

class TestAddParsedSeeds:
    def test_columns_added(self, seeds_df):
        result = add_parsed_seeds(seeds_df)
        assert "SeedNum" in result.columns
        assert "Region" in result.columns
        assert (result["SeedNum"] >= 1).all()
        assert (result["SeedNum"] <= 16).all()

    def test_does_not_mutate_input(self, seeds_df):
        original_cols = list(seeds_df.columns)
        add_parsed_seeds(seeds_df)
        assert list(seeds_df.columns) == original_cols


# ---------------------------------------------------------------------------
# build_seed_lookup
# ---------------------------------------------------------------------------

class TestBuildSeedLookup:
    def test_basic(self, seeds_df):
        lookup = build_seed_lookup(seeds_df)
        assert lookup[(2024, 1101)] == 1
        assert lookup[(2024, 1104)] == 16
        assert lookup[(2023, 1103)] == 8

    def test_all_entries_present(self, seeds_df):
        lookup = build_seed_lookup(seeds_df)
        assert len(lookup) == len(seeds_df)


# ---------------------------------------------------------------------------
# normalize_detailed_results
# ---------------------------------------------------------------------------

class TestNormalizeDetailed:
    def test_doubles_row_count(self, raw_detailed_games):
        result = normalize_detailed_results(raw_detailed_games)
        # Each game produces 2 rows (one per team)
        assert len(result) == 2 * len(raw_detailed_games)

    def test_win_flag(self, raw_detailed_games):
        result = normalize_detailed_results(raw_detailed_games)
        # Exactly half should be wins
        assert result["Win"].sum() == len(raw_detailed_games)

    def test_team_columns_exist(self, raw_detailed_games):
        result = normalize_detailed_results(raw_detailed_games)
        for col in ["TeamID", "OppID", "Score", "OppScore", "Win"]:
            assert col in result.columns

    def test_score_consistency(self, raw_detailed_games):
        """Winner's Score in the raw data should match the winner row's Score."""
        result = normalize_detailed_results(raw_detailed_games)
        winners = result[result["Win"] == 1]
        # Score should always be >= OppScore for winners
        assert (winners["Score"] >= winners["OppScore"]).all()


# ---------------------------------------------------------------------------
# normalize_compact_results
# ---------------------------------------------------------------------------

class TestNormalizeCompact:
    def test_doubles_row_count(self, tourney_compact):
        result = normalize_compact_results(tourney_compact)
        assert len(result) == 2 * len(tourney_compact)

    def test_win_flag(self, tourney_compact):
        result = normalize_compact_results(tourney_compact)
        assert result["Win"].sum() == len(tourney_compact)


# ---------------------------------------------------------------------------
# DataValidationError
# ---------------------------------------------------------------------------

class TestDataValidation:
    def test_missing_columns_raises(self, tmp_path, monkeypatch):
        """_load_csv should raise DataValidationError for bad schemas."""
        import src.data.load as load_mod

        csv_path = tmp_path / "MTeams.csv"
        pd.DataFrame({"BadCol": [1]}).to_csv(csv_path, index=False)

        monkeypatch.setattr(load_mod, "RAW_DIR", tmp_path)
        with pytest.raises(DataValidationError, match="missing required columns"):
            load_mod._load_csv("MTeams.csv")

    def test_valid_schema_passes(self, tmp_path, monkeypatch):
        import src.data.load as load_mod

        csv_path = tmp_path / "MTeams.csv"
        pd.DataFrame({"TeamID": [1], "TeamName": ["Test"]}).to_csv(csv_path, index=False)

        monkeypatch.setattr(load_mod, "RAW_DIR", tmp_path)
        df = load_mod._load_csv("MTeams.csv")
        assert len(df) == 1
