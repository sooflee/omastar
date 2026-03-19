"""Generate dashboard data JSON from model outputs."""
import logging
import sys
import os
import json
import warnings
import numpy as np

# LightGBM triggers harmless sklearn feature-name warnings when used inside
# a Pipeline with StandardScaler (numpy arrays lose column names between steps).
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but StandardScaler was fitted without feature names",
    category=UserWarning,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, DEFAULT_N_SIMULATIONS, setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    logger.info("Generating dashboard data...")

    from src.features.builder import build_full_training_set, build_team_features
    from src.model.train import train_loso_cv, train_final_model
    from src.model.predict import generate_pairwise_probabilities
    from src.model.shap_analysis import compute_shap_importance
    from src.simulation.bracket import Bracket
    from src.simulation.monte_carlo import simulate_tournament
    from src.data import load
    from src.data.clean import parse_seed

    features, target, feature_cols = build_full_training_set()
    logger.info("  %d matchups, %d features", len(features), len(feature_cols))

    # --- Seed-only baseline ---
    logger.info("  Computing seed baseline...")
    seed_results = train_loso_cv(features, target, ["SeedNum_diff"], model_type="logistic")

    # --- Full model comparisons ---
    model_configs = [
        ("logistic", "Logistic Regression"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("ensemble", "Ensemble (LR + XGB)"),
        ("stacked", "Stacked Ensemble"),
    ]

    all_results = {}
    for mtype, display_name in model_configs:
        logger.info("  Training %s...", display_name)
        try:
            r = train_loso_cv(features, target, feature_cols, model_type=mtype)
            all_results[mtype] = (r, display_name)
        except Exception as e:
            logger.warning("    Skipped %s: %s", mtype, e)

    # Find best model
    best_type = min(all_results, key=lambda k: all_results[k][0]["overall_logloss"])
    best_results = all_results[best_type][0]

    # --- Train final model and simulate ---
    logger.info("  Training final model...")
    calibrator = best_results.get("calibrator")
    model = train_final_model(
        features, target, feature_cols,
        model_type=best_type,
        calibrator=calibrator,
    )

    seeds = load.load_tourney_seeds()
    slots = load.load_tourney_slots()
    teams = load.load_teams()
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))
    team_features = build_team_features()

    season = 2026
    season_teams = seeds[seeds["Season"] == season]["TeamID"].tolist()

    logger.info("  Generating predictions...")
    prob_matrix = generate_pairwise_probabilities(
        model, team_features, season, feature_cols, season_teams,
    )

    logger.info("  Running simulations...")
    bracket = Bracket(slots, seeds, season)
    sim_results = simulate_tournament(bracket, prob_matrix, n_simulations=DEFAULT_N_SIMULATIONS)

    # --- Seed-only simulation for comparison ---
    logger.info("  Running seed-only simulations...")
    seed_model = train_final_model(features, target, ["SeedNum_diff"], model_type="logistic")
    seed_prob_matrix = generate_pairwise_probabilities(
        seed_model, team_features, season, ["SeedNum_diff"], season_teams,
    )
    bracket_seed = Bracket(slots, seeds, season)
    seed_sim = simulate_tournament(bracket_seed, seed_prob_matrix, n_simulations=DEFAULT_N_SIMULATIONS, show_progress=False)

    # --- SHAP feature importance ---
    logger.info("  Computing SHAP feature importance...")
    from config import FIGURES_DIR
    shap_importance = compute_shap_importance(
        model, features, feature_cols,
        save_dir=FIGURES_DIR,
    )

    # --- Build output JSON ---
    seed_map = {}
    for _, row in seeds[seeds["Season"] == season].iterrows():
        seed_map[int(row["TeamID"])] = row["Seed"]

    def per_season_data(results):
        return [
            {
                "season": int(s["season"]),
                "logloss": round(s["logloss"], 4),
                "accuracy": round(s["accuracy"], 4),
                "n_games": int(s["n_games"]),
            }
            for s in results["per_season"]
        ]

    def team_probs(sim_res, top_n=68):
        adv = sim_res["advancement"]
        champ = sim_res["champion_probs"]
        n_rounds = len(list(adv.values())[0])
        has_playin = n_rounds == 7
        offset = 1 if has_playin else 0

        teams_list = []
        for tid in adv:
            teams_list.append({
                "id": int(tid),
                "name": team_names.get(tid, str(tid)),
                "seed": seed_map.get(tid, "?"),
                "seedNum": parse_seed(seed_map[tid]) if tid in seed_map else 99,
                "r64": round(float(adv[tid][offset]) * 100, 1),
                "r32": round(float(adv[tid][offset + 1]) * 100, 1),
                "s16": round(float(adv[tid][offset + 2]) * 100, 1),
                "e8": round(float(adv[tid][offset + 3]) * 100, 1),
                "f4": round(float(adv[tid][offset + 4]) * 100, 1),
                "champ": round(float(champ.get(tid, 0)) * 100, 1),
            })
        teams_list.sort(key=lambda x: x["champ"], reverse=True)
        return teams_list[:top_n]

    # Build models section
    models_data = {
        "seedOnly": {
            "name": "Seed Only",
            "logloss": round(seed_results["overall_logloss"], 4),
            "accuracy": round(seed_results["overall_accuracy"] * 100, 1),
            "brier": round(seed_results["overall_brier"], 4),
            "perSeason": per_season_data(seed_results),
        },
    }

    for mtype, (r, display_name) in all_results.items():
        entry = {
            "name": display_name,
            "logloss": round(r["overall_logloss"], 4),
            "accuracy": round(r["overall_accuracy"] * 100, 1),
            "brier": round(r["overall_brier"], 4),
            "perSeason": per_season_data(r),
        }
        if r.get("calibrated_logloss") is not None:
            entry["calibratedLogloss"] = round(r["calibrated_logloss"], 4)
        models_data[mtype] = entry

    # Historical seed performance
    seed_perf = {}
    for _, game in load.load_tourney_compact().iterrows():
        s = game["Season"]
        w_seed_row = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == game["WTeamID"])]
        l_seed_row = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == game["LTeamID"])]
        if len(w_seed_row) == 0 or len(l_seed_row) == 0:
            continue
        w_seed = parse_seed(w_seed_row["Seed"].values[0])
        l_seed = parse_seed(l_seed_row["Seed"].values[0])
        for seed_num, won in [(w_seed, True), (l_seed, False)]:
            if seed_num not in seed_perf:
                seed_perf[seed_num] = {"wins": 0, "losses": 0}
            if won:
                seed_perf[seed_num]["wins"] += 1
            else:
                seed_perf[seed_num]["losses"] += 1

    seed_stats = []
    for s in sorted(seed_perf.keys()):
        total = seed_perf[s]["wins"] + seed_perf[s]["losses"]
        seed_stats.append({
            "seed": s,
            "wins": seed_perf[s]["wins"],
            "losses": seed_perf[s]["losses"],
            "winPct": round(seed_perf[s]["wins"] / total * 100, 1) if total > 0 else 0,
        })

    # --- Interactive predictor (pairwise lookup + team features for display) ---
    logger.info("  Exporting pairwise probabilities for predictor...")
    import pandas as pd

    # Build name-based pairwise probability lookup from ensemble model
    pairwise = {}
    for _, row in prob_matrix.iterrows():
        name_a = team_names.get(int(row["TeamA"]), str(int(row["TeamA"])))
        name_b = team_names.get(int(row["TeamB"]), str(int(row["TeamB"])))
        key = name_a + "|" + name_b
        pairwise[key] = round(float(row["ProbA"]), 4)

    # Team feature values for display
    season_tf = team_features[team_features["Season"] == season]
    pred_team_vals = {}
    for _, row in season_tf.iterrows():
        tid = int(row["TeamID"])
        tname = team_names.get(tid, str(tid))
        if tid not in seed_map:
            continue
        tvals = {}
        for col in feature_cols:
            base = col.replace("_diff", "")
            if base in row.index:
                v = row[base]
                tvals[base] = round(float(v), 3) if pd.notna(v) else None
        pred_team_vals[tname] = tvals

    predictor_export = {
        "features": [c.replace("_diff", "") for c in feature_cols],
        "pairwise": pairwise,
        "teamValues": pred_team_vals,
    }

    dashboard_data = {
        "season": season,
        "nFeatures": len(feature_cols),
        "nTrainingGames": len(features),
        "featureNames": [f.replace("_diff", "") for f in feature_cols],
        "models": models_data,
        "bestModel": best_type,
        "featureImportance": shap_importance[:20],
        "teamProbabilities": team_probs(sim_results, top_n=68),
        "seedBaseline": team_probs(seed_sim, top_n=68),
        "seedStats": seed_stats,
        "predictorModel": predictor_export,
    }

    out_path = OUTPUT_DIR / "dashboard_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)
    logger.info("  Dashboard data saved to %s", out_path)


if __name__ == "__main__":
    main()
