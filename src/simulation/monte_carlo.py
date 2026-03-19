import logging
import numpy as np
import pandas as pd

from src.simulation.bracket import Bracket
from src.model.predict import build_prob_lookup, lookup_prob
from config import DEFAULT_N_SIMULATIONS, DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


def _simulate_batch(
    bracket: Bracket,
    prob_lookup: dict[tuple[int, int], float],
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Vectorized simulation of N tournaments using NumPy.

    Instead of looping sim-by-sim (N × 63 Python iterations), this loops
    over 63 slots and resolves all N simulations per slot in one vectorized
    step. For each slot, unique matchups are identified and looked up once.

    Returns:
        winners: int32 array of shape (n_simulations, n_slots) with team IDs.
        slot_order: slot names corresponding to columns (round order).
    """
    rounds = bracket.get_round_slots()

    # Flatten slots in round order
    slot_order = []
    for round_slots in rounds:
        slot_order.extend(round_slots)
    slot_to_col = {slot: i for i, slot in enumerate(slot_order)}

    n_slots = len(slot_order)
    winners = np.zeros((n_simulations, n_slots), dtype=np.int32)
    randoms = rng.random((n_simulations, n_slots))

    for round_slots in rounds:
        for slot in round_slots:
            col = slot_to_col[slot]
            src1, src2 = bracket.slot_sources[slot]

            # Resolve source to team arrays (N values each)
            if src1 in bracket.seed_to_team:
                teams_a = np.full(n_simulations, bracket.seed_to_team[src1], dtype=np.int32)
            else:
                teams_a = winners[:, slot_to_col[src1]]

            if src2 in bracket.seed_to_team:
                teams_b = np.full(n_simulations, bracket.seed_to_team[src2], dtype=np.int32)
            else:
                teams_b = winners[:, slot_to_col[src2]]

            # Find unique matchups to minimize dict lookups
            matchups = np.column_stack([teams_a, teams_b])
            unique_pairs, inverse = np.unique(matchups, axis=0, return_inverse=True)

            unique_probs = np.array([
                lookup_prob(prob_lookup, int(p[0]), int(p[1]))
                for p in unique_pairs
            ])

            probs = unique_probs[inverse]
            a_wins = randoms[:, col] < probs
            winners[:, col] = np.where(a_wins, teams_a, teams_b)

    return winners, slot_order


def simulate_tournament(
    bracket: Bracket,
    prob_matrix: pd.DataFrame,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
    show_progress: bool = True,
) -> dict:
    """Run Monte Carlo simulation of the tournament bracket.

    Args:
        bracket: Bracket object with structure and seed-team mappings.
        prob_matrix: Pairwise probability DataFrame (TeamA, TeamB, ProbA).
        n_simulations: Number of simulations to run.
        seed: Random seed for reproducibility.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with:
        - 'advancement': dict[team_id -> array of shape (6,)] with probability
          of advancing to each round
        - 'champion_probs': dict[team_id -> float]
        - 'final_four_probs': dict[team_id -> float]
        - 'sim_champions': array of champion team IDs per simulation
    """
    if show_progress:
        logger.info("  Simulating %s tournaments (vectorized)...", f"{n_simulations:,}")

    rng = np.random.default_rng(seed)
    prob_lookup = build_prob_lookup(prob_matrix)

    winners, slot_order = _simulate_batch(bracket, prob_lookup, n_simulations, rng)

    # Compute results
    rounds = bracket.get_round_slots()
    all_teams = bracket.get_all_team_ids()
    n_rounds = len(rounds)
    slot_to_col = {slot: i for i, slot in enumerate(slot_order)}
    max_team_id = max(all_teams) + 1

    # Advancement: use bincount per round (6 calls instead of 64×63 comparisons)
    advancement = {t: np.zeros(n_rounds) for t in all_teams}
    for round_num, round_slots in enumerate(rounds):
        round_cols = [slot_to_col[slot] for slot in round_slots]
        round_winners = winners[:, round_cols].ravel()
        counts = np.bincount(round_winners.astype(np.intp), minlength=max_team_id)
        for t in all_teams:
            advancement[t][round_num] = counts[t] / n_simulations

    # Compute 95% confidence intervals using the normal approximation for
    # binomial proportions:  CI = p ± z * sqrt(p*(1-p)/n)
    # With n = 50,000 sims and p = 0.50, the half-width is ±0.4%.
    z = 1.96  # 95% CI
    advancement_ci = {}
    for t in all_teams:
        ci = np.zeros(n_rounds)
        for r in range(n_rounds):
            p = advancement[t][r]
            ci[r] = z * np.sqrt(p * (1 - p) / n_simulations)
        advancement_ci[t] = ci

    # Champions
    champ_col = slot_to_col[[s for s in slot_order if s.startswith("R6")][0]]
    champ_array = winners[:, champ_col]

    champ_probs = {}
    champ_ci = {}
    for t in all_teams:
        p = float(np.mean(champ_array == t))
        champ_probs[t] = p
        champ_ci[t] = z * np.sqrt(p * (1 - p) / n_simulations)

    # Final Four is the second-to-last round (R5)
    ff_round = n_rounds - 2
    ff_probs = {t: advancement[t][ff_round] for t in all_teams}

    return {
        "advancement": advancement,
        "advancement_ci": advancement_ci,
        "champion_probs": champ_probs,
        "champion_ci": champ_ci,
        "final_four_probs": ff_probs,
        "sim_champions": champ_array,
        "n_simulations": n_simulations,
    }


def simulate_batch(
    bracket: Bracket,
    prob_matrix: pd.DataFrame,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Simulate N complete brackets as a batch (for optimizer use).

    Returns:
        winners: int32 array shape (n_simulations, n_slots) with team IDs.
        slot_order: slot names corresponding to columns.
    """
    prob_lookup = build_prob_lookup(prob_matrix)
    return _simulate_batch(bracket, prob_lookup, n_simulations, rng)


def simulate_single_bracket(
    bracket: Bracket,
    prob_matrix: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """Simulate one complete bracket, returning slot -> winner mapping."""
    if rng is None:
        rng = np.random.default_rng()

    prob_lookup = build_prob_lookup(prob_matrix)
    rounds = bracket.get_round_slots()
    bracket.reset()

    for round_slots in rounds:
        for slot in round_slots:
            team_a, team_b = bracket.get_matchup(slot)
            if team_a is None or team_b is None:
                continue

            p = lookup_prob(prob_lookup, team_a, team_b)
            winner = team_a if rng.random() < p else team_b
            bracket.set_winner(slot, winner)

    return dict(bracket.slot_winner)
