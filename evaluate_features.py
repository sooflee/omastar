"""Evaluate candidate features individually against the current baseline.

Tests each new feature by adding it one-at-a-time to the core feature set
and measuring LOSO CV performance. Reports accuracy, log-loss, and Brier score
changes vs. baseline.
"""
import sys
import os
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    from src.features.builder import build_full_training_set, build_team_features
    from src.model.train import train_loso_cv, CORE_FEATURES
    from src.features.matchup import build_training_data
    from src.features.seed_matchup import add_seed_matchup_features
    from src.data import load
    from config import FIRST_DETAILED_SEASON

    # Build full team features (includes new features)
    logger.info("Building team features...")
    team_features = build_team_features()

    # Build matchup features (all _diff columns)
    tourney = load.load_tourney_compact()
    tourney = tourney[tourney["Season"] >= FIRST_DETAILED_SEASON]
    features, target = build_training_data(team_features, tourney)
    features = add_seed_matchup_features(features, team_features)

    all_diff_cols = [c for c in features.columns if c.endswith("_diff")]
    logger.info("Total available features: %d", len(all_diff_cols))

    # Current baseline: CORE_FEATURES
    baseline_cols = [f for f in CORE_FEATURES if f in all_diff_cols]
    logger.info("Baseline features (%d): %s", len(baseline_cols), baseline_cols)

    # Seed-only baseline
    logger.info("\n--- Seed-Only Baseline ---")
    seed_result = train_loso_cv(features, target, ["SeedNum_diff"], model_type="logistic",
                                calibrate=False)
    seed_acc = seed_result["overall_accuracy"] * 100
    seed_ll = seed_result["overall_logloss"]
    seed_brier = seed_result["overall_brier"]
    logger.info("  Seed Only:  acc=%.2f%%  ll=%.4f  brier=%.4f", seed_acc, seed_ll, seed_brier)

    # Current model baseline (using ensemble)
    logger.info("\n--- Current Model Baseline ---")
    base_result = train_loso_cv(features, target, baseline_cols, model_type="ensemble",
                                calibrate=False)
    base_acc = base_result["overall_accuracy"] * 100
    base_ll = base_result["overall_logloss"]
    base_brier = base_result["overall_brier"]
    logger.info("  Baseline:   acc=%.2f%%  ll=%.4f  brier=%.4f", base_acc, base_ll, base_brier)

    # Candidate feature groups to test individually
    candidate_groups = {
        "MarginStd (consistency)": ["MarginStd_diff"],
        "ScoreStd (scoring variance)": ["ScoreStd_diff"],
        "FG3PctStd (3PT variance)": ["FG3PctStd_diff"],
        "All variance": ["MarginStd_diff", "ScoreStd_diff", "FG3PctStd_diff", "BestMargin_diff"],
        "CloseWinPct": ["CloseWinPct_diff"],
        "CloseGamePct": ["CloseGamePct_diff"],
        "All clutch": ["CloseWinPct_diff", "CloseGamePct_diff", "OTWinPct_diff"],
        "ConfStrength": ["ConfStrength_diff"],
        "ConfDepth": ["ConfDepth_diff"],
        "All conference": ["ConfStrength_diff", "ConfDepth_diff", "ConfTopHeavy_diff"],
        "MarginTrend": ["MarginTrend_diff"],
        "EffTrend": ["EffTrend_diff"],
        "WinTrendLate": ["WinTrendLate_diff"],
        "All trajectory": ["MarginTrend_diff", "EffTrend_diff", "WinTrendLate_diff"],
        "ProgramTourneyWins": ["ProgramTourneyWins_diff"],
        "ProgramTourneyApps": ["ProgramTourneyApps_diff"],
        "ProgramDeepRuns": ["ProgramDeepRuns_diff"],
        "All program exp": ["ProgramTourneyApps_diff", "ProgramTourneyWins_diff", "ProgramDeepRuns_diff"],
        "CoachTourneyWins": ["CoachTourneyWins_diff"],
        "All coach+program": ["CoachTourneyWins_diff", "CoachTourneyApps_diff",
                              "ProgramTourneyApps_diff", "ProgramTourneyWins_diff", "ProgramDeepRuns_diff"],
        # --- NEW EXTERNAL DATA ---
        "InjuryRank": ["InjuryRank_diff"],
        "APFinalRank": ["APFinalRank_diff"],
        "APFinalVotes": ["APFinalVotes_diff"],
        "All AP Poll": ["APFinalRank_diff", "APFinalVotes_diff"],
        "RPPFRating": ["RPPFRating_diff"],
        "RPPFAdjEM": ["RPPFAdjEM_diff"],
        "All RPPF": ["RPPFRating_diff", "RPPFAdjEM_diff", "RPPFSOS_diff"],
        "DunksShare": ["DunksShare_diff"],
        "CloseTwosShare": ["CloseTwosShare_diff"],
        "CloseTwosFGPct": ["CloseTwosFGPct_diff"],
        "ThreesShare": ["ThreesShare_diff"],
        "Shot quality (offense)": ["DunksShare_diff", "DunksFGPct_diff", "CloseTwosShare_diff",
                                    "CloseTwosFGPct_diff", "ThreesShare_diff"],
        "Shot quality (defense)": ["DunksDShare_diff", "CloseTwosDFGPct_diff", "ThreesDFGPct_diff"],
        "All shooting splits": ["DunksShare_diff", "DunksFGPct_diff", "CloseTwosShare_diff",
                                  "CloseTwosFGPct_diff", "ThreesShare_diff",
                                  "DunksDShare_diff", "CloseTwosDFGPct_diff", "ThreesDFGPct_diff"],
        "RosterRank (talent)": ["RosterRank_diff"],
    }

    # Filter to features that actually exist in the data
    for name in list(candidate_groups.keys()):
        available = [f for f in candidate_groups[name] if f in all_diff_cols]
        if not available:
            logger.warning("  Skipping %s — features not found", name)
            del candidate_groups[name]
        else:
            candidate_groups[name] = available

    # Test each candidate
    logger.info("\n" + "=" * 85)
    logger.info("%-30s  %7s  %7s  %7s  |  %7s  %7s  %7s",
                "Feature", "Acc%", "LL", "Brier", "dAcc", "dLL", "dBrier")
    logger.info("=" * 85)

    results = []
    for name, feat_cols in candidate_groups.items():
        test_cols = baseline_cols + feat_cols
        # Remove duplicates while preserving order
        test_cols = list(dict.fromkeys(test_cols))

        result = train_loso_cv(features, target, test_cols, model_type="ensemble",
                               calibrate=False)
        acc = result["overall_accuracy"] * 100
        ll = result["overall_logloss"]
        brier = result["overall_brier"]

        d_acc = acc - base_acc
        d_ll = ll - base_ll  # negative is better
        d_brier = brier - base_brier  # negative is better

        # Symbols for quick reading
        acc_sym = "+" if d_acc > 0.1 else ("-" if d_acc < -0.1 else " ")
        ll_sym = "+" if d_ll < -0.001 else ("-" if d_ll > 0.001 else " ")

        logger.info("%-30s  %6.2f%%  %.4f  %.4f  | %+6.2f%s %+.4f%s %+.4f",
                     name, acc, ll, brier, d_acc, acc_sym, d_ll, ll_sym, d_brier)

        results.append({
            "name": name,
            "features": feat_cols,
            "accuracy": acc,
            "logloss": ll,
            "brier": brier,
            "d_accuracy": d_acc,
            "d_logloss": d_ll,
            "d_brier": d_brier,
        })

    # Summary: best additions
    logger.info("\n" + "=" * 85)
    logger.info("BEST BY ACCURACY GAIN:")
    by_acc = sorted(results, key=lambda x: x["d_accuracy"], reverse=True)
    for r in by_acc[:5]:
        logger.info("  %+.2f%%  %-30s  (ll: %+.4f)", r["d_accuracy"], r["name"], r["d_logloss"])

    logger.info("\nBEST BY LOG-LOSS IMPROVEMENT:")
    by_ll = sorted(results, key=lambda x: x["d_logloss"])
    for r in by_ll[:5]:
        logger.info("  %+.4f  %-30s  (acc: %+.2f%%)", r["d_logloss"], r["name"], r["d_accuracy"])

    # Test best combination
    logger.info("\n--- Testing Best Combination ---")
    # Take features that improved both accuracy and log-loss
    good_features = []
    for r in results:
        if r["d_accuracy"] > 0 and r["d_logloss"] < 0:
            good_features.extend(r["features"])
    good_features = list(dict.fromkeys(good_features))  # dedupe

    if good_features:
        combo_cols = list(dict.fromkeys(baseline_cols + good_features))
        logger.info("Combined features (%d new): %s", len(good_features), good_features)
        combo_result = train_loso_cv(features, target, combo_cols, model_type="ensemble",
                                     calibrate=False)
        combo_acc = combo_result["overall_accuracy"] * 100
        combo_ll = combo_result["overall_logloss"]
        combo_brier = combo_result["overall_brier"]
        logger.info("  Combined:   acc=%.2f%% (%+.2f)  ll=%.4f (%+.4f)  brier=%.4f (%+.4f)",
                     combo_acc, combo_acc - base_acc, combo_ll, combo_ll - base_ll,
                     combo_brier, combo_brier - base_brier)
    else:
        logger.info("No features improved both accuracy and log-loss simultaneously.")


if __name__ == "__main__":
    main()
