"""
Main pipeline: train model, simulate tournament, optimize bracket.

Usage:
    python run.py --season 2024 --simulations 50000
    python run.py --season 2024 --tune --tune-trials 30
    python run.py --season 2024 --model stacked

Prerequisites:
    Download Kaggle "March Machine Learning Mania" dataset and extract CSVs to data/raw/
    https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data
"""

import argparse
import logging
import sys
import os
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_DIR, PROCESSED_DIR, MODEL_DIR, PREDICTIONS_DIR,
    BRACKETS_DIR, FIGURES_DIR, DEFAULT_N_SIMULATIONS,
    setup_logging,
)

logger = logging.getLogger(__name__)


def ensure_dirs():
    for d in [PROCESSED_DIR, MODEL_DIR, PREDICTIONS_DIR, BRACKETS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def check_data():
    required = ["MTeams.csv", "MNCAATourneySeeds.csv",
                "MRegularSeasonDetailedResults.csv", "MNCAATourneyCompactResults.csv",
                "MNCAATourneySlots.csv"]
    missing = [f for f in required if not (RAW_DIR / f).exists()]
    if missing:
        logger.error("Missing required data files in data/raw/:")
        for f in missing:
            logger.error("  - %s", f)
        logger.error("Download from Kaggle 'March Machine Learning Mania' and extract to data/raw/")
        sys.exit(1)


def check_season_data(season: int):
    """Validate that the target season has tournament data available."""
    import pandas as pd
    seeds_path = RAW_DIR / "MNCAATourneySeeds.csv"
    slots_path = RAW_DIR / "MNCAATourneySlots.csv"
    if not seeds_path.exists() or not slots_path.exists():
        return  # check_data() will catch missing files

    seeds = pd.read_csv(seeds_path)
    if season not in seeds["Season"].values:
        logger.error("No tournament seeds found for season %d.", season)
        logger.error("Available seasons: %d-%d",
                      seeds["Season"].min(), seeds["Season"].max())
        logger.error("Make sure your Kaggle data includes season %d.", season)
        sys.exit(1)

    slots = pd.read_csv(slots_path)
    if season not in slots["Season"].values:
        logger.error("No tournament bracket slots found for season %d.", season)
        logger.error("Make sure your Kaggle data includes season %d.", season)
        sys.exit(1)

    n_teams = len(seeds[seeds["Season"] == season])
    logger.info("  Season %d: %d tournament teams found", season, n_teams)


def step_build_features():
    logger.info("[1/6] Building features...")
    from src.features.builder import build_full_training_set
    features, target, feature_cols = build_full_training_set()
    logger.info("  Training set: %d matchups, %d features", len(features), len(feature_cols))
    logger.info("  Seasons: %d - %d", features['Season'].min(), features['Season'].max())
    return features, target, feature_cols


def step_tune(features, target, feature_cols, n_trials=50):
    logger.info("[2/6] Tuning hyperparameters (%d trials per model)...", n_trials)
    from src.model.tuning import tune_xgboost, tune_lightgbm
    from src.model import train as train_module

    logger.info("  --- Tuning XGBoost ---")
    xgb_params = tune_xgboost(features, target, feature_cols, n_trials=n_trials)
    train_module.DEFAULT_XGB_PARAMS.update(xgb_params)

    logger.info("  --- Tuning LightGBM ---")
    lgbm_params = tune_lightgbm(features, target, feature_cols, n_trials=n_trials)
    train_module.DEFAULT_LGBM_PARAMS.update(lgbm_params)


def step_evaluate_model(features, target, feature_cols):
    logger.info("[3/6] Evaluating models (LOSO cross-validation)...")
    from src.model.train import train_loso_cv
    from src.model.evaluate import print_cv_results, plot_calibration

    model_types = ["logistic", "xgboost", "lightgbm", "ensemble", "stacked"]

    best_type = None
    best_logloss = float("inf")
    results = None

    for mtype in model_types:
        logger.info("  --- %s ---", mtype.upper())
        try:
            r = train_loso_cv(features, target, feature_cols, model_type=mtype)
            uncal = r["overall_logloss"]
            cal = r.get("calibrated_logloss")
            cal_str = f"  |  Calibrated: {cal:.4f}" if cal is not None else ""
            logger.info("  Log-Loss: %.4f%s  |  Accuracy: %.4f",
                        uncal, cal_str, r['overall_accuracy'])

            effective_ll = cal if cal is not None else uncal
            if effective_ll < best_logloss:
                best_logloss = effective_ll
                best_type = mtype
                results = r
        except Exception as e:
            logger.warning("  Skipped %s: %s", mtype, e)

    logger.info("  Best model: %s (log-loss: %.4f)", best_type.upper(), best_logloss)
    print_cv_results(results)

    try:
        plot_calibration(results, save_path=str(FIGURES_DIR / "calibration.png"))
    except Exception:
        logger.debug("  Could not generate calibration plot", exc_info=True)

    return results, best_type


def step_train_final_model(features, target, feature_cols, model_type="stacked",
                           calibrator=None):
    logger.info("[4/6] Training final %s model on all data...", model_type)
    from src.model.train import train_final_model

    model = train_final_model(
        features, target, feature_cols,
        model_type=model_type,
        calibrator=calibrator,
    )
    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(model, model_path)
    logger.info("  Model saved to %s", model_path)
    return model


def step_shap_analysis(model, features, feature_cols):
    logger.info("[5/6] Running SHAP feature importance analysis...")
    from src.model.shap_analysis import compute_shap_importance

    importance = compute_shap_importance(
        model, features, feature_cols,
        save_dir=FIGURES_DIR,
    )

    logger.info("  Top 15 features by SHAP importance:")
    for i, feat in enumerate(importance[:15], 1):
        logger.info("    %2d. %-25s %.4f", i, feat['feature'], feat['importance'])

    return importance


def step_simulate(model, feature_cols, season, n_simulations):
    logger.info("[6/6] Running Monte Carlo simulation (%s sims)...", f"{n_simulations:,}")
    from src.data.load import load_tourney_seeds, load_tourney_slots, load_teams
    from src.features.builder import build_team_features
    from src.model.predict import generate_pairwise_probabilities
    from src.simulation.bracket import Bracket
    from src.simulation.monte_carlo import simulate_tournament
    from src.simulation.results import advancement_table, print_advancement_table

    seeds = load_tourney_seeds()
    slots = load_tourney_slots()
    teams = load_teams()

    team_features = build_team_features()

    season_teams = seeds[seeds["Season"] == season]["TeamID"].tolist()
    if not season_teams:
        logger.error("  No tournament teams found for season %d", season)
        sys.exit(1)

    prob_matrix = generate_pairwise_probabilities(
        model, team_features, season, feature_cols, season_teams,
    )
    prob_matrix.to_csv(PREDICTIONS_DIR / f"probabilities_{season}.csv", index=False)

    bracket = Bracket(slots, seeds, season)
    sim_results = simulate_tournament(bracket, prob_matrix, n_simulations)

    table = advancement_table(sim_results, teams, seeds, season)
    print_advancement_table(table)

    table.to_csv(PREDICTIONS_DIR / f"advancement_{season}.csv", index=False)

    return bracket, prob_matrix, sim_results


def step_optimize(bracket, prob_matrix, season):
    logger.info("Optimizing bracket...")
    from src.optimization.optimizer import optimize_bracket
    from src.optimization.scoring import STANDARD
    from src.data.load import load_teams

    teams = load_teams()
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))

    result = optimize_bracket(
        bracket, prob_matrix, scoring=STANDARD,
        pool_size=10, n_mc_outcomes=5000, n_iterations=3000,
    )

    logger.info("  Chalk bracket expected score: %.1f", result['chalk_expected_score'])
    logger.info("  Optimized bracket expected score: %.1f", result['expected_score'])

    rounds = bracket.get_round_slots()
    round_names = bracket.ROUND_NAMES

    for round_num, (round_slots, name) in enumerate(zip(rounds, round_names)):
        logger.info("  %s:", name)
        for slot in sorted(round_slots):
            winner = result["picks"].get(slot)
            if winner:
                logger.info("    %s: %s", slot, team_names.get(winner, winner))

    # Save picks
    picks_df = pd.DataFrame([
        {"Slot": slot, "TeamID": team_id, "Team": team_names.get(team_id, team_id)}
        for slot, team_id in sorted(result["picks"].items())
    ])
    picks_df.to_csv(BRACKETS_DIR / f"bracket_{season}.csv", index=False)
    logger.info("  Bracket saved to %s", BRACKETS_DIR / f"bracket_{season}.csv")

    return result


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="March Madness Bracket Predictor")
    parser.add_argument("--season", type=int, required=True,
                        help="Tournament season to predict (e.g., 2024)")
    parser.add_argument("--simulations", type=int, default=DEFAULT_N_SIMULATIONS,
                        help=f"Number of Monte Carlo simulations (default: {DEFAULT_N_SIMULATIONS:,})")
    parser.add_argument("--model", type=str, default=None,
                        choices=["logistic", "xgboost", "lightgbm", "catboost",
                                 "ensemble", "stacked"],
                        help="Model type (default: best from evaluation)")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning before training")
    parser.add_argument("--tune-trials", type=int, default=50,
                        help="Number of Optuna trials per model (default: 50)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip cross-validation evaluation step")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip bracket optimization step")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis step")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Disable probability calibration")
    parser.add_argument("--no-time-weights", action="store_true",
                        help="Disable time-decay sample weighting")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging")

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    ensure_dirs()
    check_data()
    check_season_data(args.season)

    features, target, feature_cols = step_build_features()

    if args.tune:
        step_tune(features, target, feature_cols, n_trials=args.tune_trials)

    best_type = args.model or "ensemble"
    calibrator = None
    if not args.skip_eval:
        results, best_type = step_evaluate_model(features, target, feature_cols)
        if args.model:
            best_type = args.model
        calibrator = results.get("calibrator") if not args.no_calibrate else None

    model = step_train_final_model(
        features, target, feature_cols, best_type,
        calibrator=calibrator,
    )

    shap_importance = None
    if not args.skip_shap:
        shap_importance = step_shap_analysis(model, features, feature_cols)

    bracket, prob_matrix, sim_results = step_simulate(
        model, feature_cols, args.season, args.simulations,
    )

    if not args.skip_optimize:
        step_optimize(bracket, prob_matrix, args.season)

    logger.info("Done!")
