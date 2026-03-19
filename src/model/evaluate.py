import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)


def print_cv_results(results: dict) -> None:
    """Print LOSO cross-validation results summary."""
    logger.info("=" * 60)
    logger.info("Leave-One-Season-Out Cross-Validation Results")
    logger.info("=" * 60)
    logger.info("Overall Log-Loss:  %.4f", results['overall_logloss'])
    logger.info("Overall Brier:     %.4f", results['overall_brier'])
    logger.info("Overall Accuracy:  %.4f", results['overall_accuracy'])
    logger.info("%-10s %-10s %-10s %-8s", "Season", "LogLoss", "Accuracy", "Games")
    logger.info("-" * 40)

    for s in results["per_season"]:
        logger.info("%-10s %-10.4f %-10.4f %-8d",
                     s['season'], s['logloss'], s['accuracy'], s['n_games'])


def plot_calibration(results: dict, n_bins: int = 10, save_path: str | None = None) -> None:
    """Plot calibration curve from CV results."""
    preds = results["all_preds"]
    targets = results["all_targets"]

    prob_true, prob_pred = calibration_curve(targets, preds, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    ax1.plot(prob_pred, prob_true, "s-", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Prediction distribution
    ax2.hist(preds, bins=30, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_season_performance(results: dict, save_path: str | None = None) -> None:
    """Plot per-season log-loss and accuracy."""
    seasons = [s["season"] for s in results["per_season"]]
    logloss = [s["logloss"] for s in results["per_season"]]
    accuracy = [s["accuracy"] for s in results["per_season"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.bar(seasons, logloss, color="steelblue", alpha=0.8)
    ax1.axhline(results["overall_logloss"], color="red", linestyle="--", label="Overall")
    ax1.set_ylabel("Log-Loss")
    ax1.set_title("Per-Season Log-Loss (lower is better)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(seasons, accuracy, color="seagreen", alpha=0.8)
    ax2.axhline(results["overall_accuracy"], color="red", linestyle="--", label="Overall")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Season")
    ax2.set_title("Per-Season Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def baseline_seed_logloss(seeds: pd.DataFrame, tourney: pd.DataFrame) -> float:
    """Compute baseline log-loss using historical seed win rates."""
    # Historical first-round seed matchup win rates (1985–2024).
    # Source: NCAA tournament game results compiled from Sports Reference.
    # These are round-of-64 empirical averages; later-round matchups fall
    # back to a linear estimate of 0.03 per seed difference.
    seed_win_rates = {
        (1, 16): 0.99, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.79,
        (5, 12): 0.65, (6, 11): 0.63, (7, 10): 0.61, (8, 9): 0.51,
    }

    from src.data.clean import build_seed_lookup

    seed_map = build_seed_lookup(seeds)

    preds = []
    targets = []
    for _, game in tourney.iterrows():
        w_seed = seed_map.get((game["Season"], game["WTeamID"]))
        l_seed = seed_map.get((game["Season"], game["LTeamID"]))
        if w_seed is None or l_seed is None:
            continue

        # Canonical ordering: lower TeamID = TeamA
        team_a = min(game["WTeamID"], game["LTeamID"])
        team_a_seed = seed_map[(game["Season"], team_a)]
        team_b = max(game["WTeamID"], game["LTeamID"])
        team_b_seed = seed_map[(game["Season"], team_b)]

        target = 1 if game["WTeamID"] == team_a else 0

        # Probability TeamA wins based on seed matchup
        if team_a_seed < team_b_seed:
            key = (team_a_seed, team_b_seed)
            p = seed_win_rates.get(key, 0.5 + (team_b_seed - team_a_seed) * 0.03)
        elif team_a_seed > team_b_seed:
            key = (team_b_seed, team_a_seed)
            p = 1 - seed_win_rates.get(key, 0.5 + (team_a_seed - team_b_seed) * 0.03)
        else:
            p = 0.5

        preds.append(np.clip(p, 0.01, 0.99))
        targets.append(target)

    return log_loss(targets, preds)
