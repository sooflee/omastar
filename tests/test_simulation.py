"""Tests for src/simulation — bracket structure, Monte Carlo, results."""

import numpy as np
import pandas as pd
import pytest

from src.simulation.bracket import Bracket
from src.simulation.monte_carlo import simulate_tournament, simulate_single_bracket
from src.simulation.results import advancement_table
from src.model.predict import build_prob_lookup, lookup_prob


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------

class TestBracket:
    def test_seed_to_team(self, slots_df, seeds_df):
        b = Bracket(slots_df, seeds_df, 2024)
        assert b.seed_to_team["W01"] == 1101
        assert b.seed_to_team["Z16"] == 1104

    def test_get_matchup_round5(self, slots_df, seeds_df):
        b = Bracket(slots_df, seeds_df, 2024)
        a, bteam = b.get_matchup("R5WX")
        assert {a, bteam} == {1101, 1104}

    def test_set_winner_propagates(self, slots_df, seeds_df):
        b = Bracket(slots_df, seeds_df, 2024)
        b.set_winner("R5WX", 1101)
        b.set_winner("R5YZ", 1102)
        a, bteam = b.get_matchup("R6CH")
        assert {a, bteam} == {1101, 1102}

    def test_reset(self, slots_df, seeds_df):
        b = Bracket(slots_df, seeds_df, 2024)
        b.set_winner("R5WX", 1101)
        b.reset()
        assert len(b.slot_winner) == 0

    def test_get_round_slots(self, slots_df, seeds_df):
        b = Bracket(slots_df, seeds_df, 2024)
        rounds = b.get_round_slots()
        # R5 slots go to index 5, R6 to index 6; empty rounds 1-4 are stripped
        # after empty play-in round 0 is removed
        # The fixture has R5 and R6 only; rounds 1–4 will be empty
        r5_slots = [s for r in rounds for s in r if s.startswith("R5")]
        r6_slots = [s for r in rounds for s in r if s.startswith("R6")]
        assert len(r5_slots) == 2
        assert len(r6_slots) == 1

    def test_get_all_team_ids(self, slots_df, seeds_df):
        b = Bracket(slots_df, seeds_df, 2024)
        teams = b.get_all_team_ids()
        assert set(teams) == {1101, 1102, 1103, 1104}


# ---------------------------------------------------------------------------
# Probability lookup
# ---------------------------------------------------------------------------

class TestProbLookup:
    def test_canonical_order(self, prob_matrix):
        lookup = build_prob_lookup(prob_matrix)
        assert abs(lookup[(1101, 1102)] - 0.65) < 1e-6

    def test_lookup_both_directions(self, prob_matrix):
        lookup = build_prob_lookup(prob_matrix)
        p_a = lookup_prob(lookup, 1101, 1102)
        p_b = lookup_prob(lookup, 1102, 1101)
        assert abs(p_a + p_b - 1.0) < 1e-6

    def test_same_team_returns_half(self, prob_matrix):
        lookup = build_prob_lookup(prob_matrix)
        assert lookup_prob(lookup, 1101, 1101) == 0.5

    def test_unknown_matchup_returns_half(self, prob_matrix):
        lookup = build_prob_lookup(prob_matrix)
        assert lookup_prob(lookup, 9999, 8888) == 0.5


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

class TestSimulateTournament:
    def test_advancement_probabilities_sum(self, slots_df, seeds_df, prob_matrix):
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=1000, show_progress=False,
        )
        adv = results["advancement"]
        # All 4 teams should be present
        assert len(adv) == 4

    def test_champion_probs_sum_to_one(self, slots_df, seeds_df, prob_matrix):
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=5000, show_progress=False,
        )
        total = sum(results["champion_probs"].values())
        assert abs(total - 1.0) < 0.02  # allow small Monte Carlo variance

    def test_confidence_intervals_present(self, slots_df, seeds_df, prob_matrix):
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=1000, show_progress=False,
        )
        assert "advancement_ci" in results
        assert "champion_ci" in results
        assert "n_simulations" in results

    def test_ci_positive(self, slots_df, seeds_df, prob_matrix):
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=1000, show_progress=False,
        )
        for team_id, ci_arr in results["advancement_ci"].items():
            assert (ci_arr >= 0).all()

    def test_higher_seed_favored(self, slots_df, seeds_df, prob_matrix):
        """With our fixture probs, team 1101 (1-seed, 0.95 vs 16) should win most."""
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=5000, show_progress=False,
        )
        # 1101 should have the highest championship probability
        best_team = max(results["champion_probs"], key=results["champion_probs"].get)
        assert best_team == 1101


class TestSimulateSingleBracket:
    def test_returns_all_slots(self, slots_df, seeds_df, prob_matrix):
        b = Bracket(slots_df, seeds_df, 2024)
        picks = simulate_single_bracket(b, prob_matrix)
        assert "R5WX" in picks
        assert "R5YZ" in picks
        assert "R6CH" in picks

    def test_champion_is_valid_team(self, slots_df, seeds_df, prob_matrix):
        b = Bracket(slots_df, seeds_df, 2024)
        picks = simulate_single_bracket(b, prob_matrix)
        assert picks["R6CH"] in {1101, 1102, 1103, 1104}


# ---------------------------------------------------------------------------
# Advancement table
# ---------------------------------------------------------------------------

class TestAdvancementTable:
    def test_output_shape(self, slots_df, seeds_df, prob_matrix, teams_df):
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=500, show_progress=False,
        )
        table = advancement_table(
            results, teams_df, seeds_df, 2024, top_n=4, has_playin=False,
        )
        assert len(table) == 4
        assert "Team" in table.columns
        assert "Seed" in table.columns

    def test_ci_columns_present(self, slots_df, seeds_df, prob_matrix, teams_df):
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=500, show_progress=False,
        )
        table = advancement_table(
            results, teams_df, seeds_df, 2024, top_n=4, has_playin=False,
            show_ci=True,
        )
        assert "Title_CI" in table.columns

    def test_no_ci_columns_when_disabled(self, slots_df, seeds_df, prob_matrix, teams_df):
        b = Bracket(slots_df, seeds_df, 2024)
        results = simulate_tournament(
            b, prob_matrix, n_simulations=500, show_progress=False,
        )
        table = advancement_table(
            results, teams_df, seeds_df, 2024, top_n=4, has_playin=False,
            show_ci=False,
        )
        assert "Title_CI" not in table.columns
