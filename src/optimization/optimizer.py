import logging

import numpy as np
import pandas as pd
from tqdm import trange

from src.simulation.bracket import Bracket
from src.simulation.monte_carlo import simulate_batch
from src.model.predict import build_prob_lookup, lookup_prob
from src.optimization.scoring import ScoringSystem, STANDARD

logger = logging.getLogger(__name__)


def make_chalk_bracket(
    bracket: Bracket,
    prob_matrix: pd.DataFrame,
) -> dict[str, int]:
    """Generate the "chalk" bracket: always pick the higher-probability team.

    Returns:
        dict of slot -> predicted winner team ID.
    """
    prob_lookup = build_prob_lookup(prob_matrix)
    rounds = bracket.get_round_slots()
    bracket.reset()
    picks = {}

    for round_slots in rounds:
        for slot in round_slots:
            team_a, team_b = bracket.get_matchup(slot)
            if team_a is None or team_b is None:
                continue

            p = lookup_prob(prob_lookup, team_a, team_b)
            winner = team_a if p >= 0.5 else team_b
            bracket.set_winner(slot, winner)
            picks[slot] = winner

    return picks


def optimize_bracket(
    bracket: Bracket,
    prob_matrix: pd.DataFrame,
    scoring: ScoringSystem = STANDARD,
    pool_size: int = 10,
    n_mc_outcomes: int = 10_000,
    n_iterations: int = 5_000,
    initial_temp: float = 2.0,
    cooling_rate: float = 0.9995,
    seed: int = 42,
    show_progress: bool = True,
    early_stop_patience: int = 500,
) -> dict:
    """Optimize bracket picks using simulated annealing.

    Maximizes expected score by considering both tournament outcome uncertainty
    and (optionally) pool competition.

    Args:
        bracket: Tournament bracket structure.
        prob_matrix: Pairwise probabilities.
        scoring: Scoring system to use.
        pool_size: Number of people in the pool (1 = maximize own score only).
        n_mc_outcomes: Number of tournament outcomes to simulate for evaluation.
        n_iterations: Maximum number of simulated annealing iterations.
        initial_temp: Starting temperature.
        cooling_rate: Temperature decay factor per iteration.
        seed: Random seed.
        show_progress: Whether to show progress bar.
        early_stop_patience: Stop early if the best score has not improved
            for this many consecutive iterations.  Set to 0 to disable.

    Returns:
        Dictionary with:
        - 'picks': optimized bracket (slot -> team_id)
        - 'expected_score': expected score of optimized bracket
        - 'chalk_expected_score': expected score of chalk bracket for comparison
    """
    rng = np.random.default_rng(seed)
    rounds = bracket.get_round_slots()
    all_slots = [slot for round_slots in rounds for slot in round_slots]
    slot_to_idx = {slot: i for i, slot in enumerate(all_slots)}
    n_slots = len(all_slots)

    # Pre-compute point values per slot
    # Play-in slots (non-R-prefixed) score 0; main rounds map to scoring system
    slot_points = np.zeros(n_slots)
    scoring_idx = 0
    for round_slots in rounds:
        is_playin = any(not s.startswith("R") for s in round_slots)
        if is_playin:
            pts = 0
        else:
            pts = scoring.points_per_round[scoring_idx] if scoring_idx < len(scoring.points_per_round) else 0
            scoring_idx += 1
        for slot in round_slots:
            slot_points[slot_to_idx[slot]] = pts

    # Step 1: Batch-simulate tournament outcomes (vectorized, not a Python loop)
    winners, slot_order = simulate_batch(bracket, prob_matrix, n_mc_outcomes, rng)
    # Reorder columns to match all_slots ordering
    slot_order_map = {slot: i for i, slot in enumerate(slot_order)}
    col_indices = [slot_order_map[slot] for slot in all_slots]
    outcomes_array = winners[:, col_indices]  # shape (n_mc_outcomes, n_slots)

    # Step 2: Score the chalk bracket as baseline
    chalk = make_chalk_bracket(bracket, prob_matrix)
    chalk_array = np.array([chalk.get(slot, -1) for slot in all_slots], dtype=np.int32)
    chalk_score = _expected_score_fast(chalk_array, outcomes_array, slot_points)

    # Step 3: Simulated annealing with vectorized scoring
    current_array = chalk_array.copy()
    current_ev = chalk_score
    best_array = current_array.copy()
    best_ev = current_ev
    temp = initial_temp

    prob_lookup = build_prob_lookup(prob_matrix)
    downstream = _build_downstream_map(bracket)

    iters_since_improvement = 0

    iterator = trange(n_iterations, desc="Optimizing", disable=not show_progress)

    for i in iterator:
        # Pick a random slot to flip
        slot_name = rng.choice(all_slots)
        slot_idx = slot_to_idx[slot_name]
        team_a, team_b = _get_matchup_for_slot(
            bracket, slot_name, current_array, slot_to_idx,
        )

        if team_a is None or team_b is None:
            continue

        # Flip the pick
        candidate_array = current_array.copy()
        old_winner = int(candidate_array[slot_idx])
        new_winner = team_b if old_winner == team_a else team_a
        candidate_array[slot_idx] = new_winner

        # Propagate downstream
        _propagate_flip_array(
            candidate_array, slot_name, old_winner, new_winner,
            downstream, slot_to_idx,
        )

        candidate_ev = _expected_score_fast(candidate_array, outcomes_array, slot_points)

        # Accept or reject
        delta = candidate_ev - current_ev
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
            current_array = candidate_array
            current_ev = candidate_ev
            if current_ev > best_ev:
                best_array = current_array.copy()
                best_ev = current_ev
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1
        else:
            iters_since_improvement += 1

        temp *= cooling_rate

        if i % 500 == 0:
            iterator.set_postfix({"EV": f"{best_ev:.1f}", "T": f"{temp:.3f}"})

        # Early stopping: halt once score has plateaued
        if early_stop_patience > 0 and iters_since_improvement >= early_stop_patience:
            logger.info("  Early stopping at iteration %d (no improvement for %d iters)",
                        i, early_stop_patience)
            break

    # Convert best picks array back to dict
    best_picks = {}
    for slot, idx in slot_to_idx.items():
        val = int(best_array[idx])
        if val > 0:
            best_picks[slot] = val

    return {
        "picks": best_picks,
        "expected_score": best_ev,
        "chalk_expected_score": chalk_score,
    }


def _expected_score_fast(
    picks_array: np.ndarray,
    outcomes_array: np.ndarray,
    slot_points: np.ndarray,
) -> float:
    """Vectorized expected score: single NumPy expression replaces nested Python loops."""
    matches = outcomes_array == picks_array  # (n_outcomes, n_slots) boolean
    return float((matches * slot_points).sum(axis=1).mean())


def _build_downstream_map(bracket: Bracket) -> dict[str, list[str]]:
    """Build map: slot -> list of slots that depend on this slot's winner."""
    downstream = {slot: [] for slot in bracket.slot_sources}

    for slot, (src1, src2) in bracket.slot_sources.items():
        if src1 in downstream:
            downstream[src1].append(slot)
        if src2 in downstream:
            downstream[src2].append(slot)

    return downstream


def _get_matchup_for_slot(
    bracket: Bracket,
    slot_name: str,
    current_array: np.ndarray,
    slot_to_idx: dict[str, int],
) -> tuple[int | None, int | None]:
    """Get the two teams that could play in a slot given current picks."""
    src1, src2 = bracket.slot_sources[slot_name]

    team_a = bracket.seed_to_team.get(src1)
    if team_a is None:
        idx = slot_to_idx.get(src1)
        if idx is not None:
            val = int(current_array[idx])
            team_a = val if val > 0 else None

    team_b = bracket.seed_to_team.get(src2)
    if team_b is None:
        idx = slot_to_idx.get(src2)
        if idx is not None:
            val = int(current_array[idx])
            team_b = val if val > 0 else None

    return team_a, team_b


def _propagate_flip_array(
    picks_array: np.ndarray,
    flipped_slot: str,
    old_winner: int,
    new_winner: int,
    downstream: dict[str, list[str]],
    slot_to_idx: dict[str, int],
):
    """Propagate a pick change through downstream slots in the array."""
    to_check = list(downstream.get(flipped_slot, []))

    while to_check:
        slot = to_check.pop(0)
        idx = slot_to_idx[slot]
        if picks_array[idx] == old_winner:
            picks_array[idx] = new_winner
            to_check.extend(downstream.get(slot, []))
