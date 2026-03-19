# Omastar

NCAA March Madness bracket prediction using machine learning and Monte Carlo simulation.

## Quickstart

```bash
# Install dependencies (Python 3.10+)
uv pip install -e .

# Download Kaggle "March Machine Learning Mania" data to data/raw/
# https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data

# Run full pipeline
python run.py --season 2026

# Generate dashboard data for the frontend
python generate_dashboard.py

# View dashboard
open frontend/index.html
```

## How it works

### The core insight

Seed alone predicts ~70% of tournament games. Most "advanced" models throw 40-50 features at the problem and barely move the needle because 1,449 training games can't support that many features — overfitting destroys the signal.

Omastar uses **7 features** selected via forward feature selection against LOSO cross-validation. This compact model beats the seed baseline in **19 of 22 seasons** with a calibrated log-loss of **0.5217** (vs 0.5520 seed-only).

### The 7 features

| Feature | What it captures |
|---|---|
| **AdjEM** | Opponent-adjusted efficiency margin (KenPom-style). The single best measure of team quality. |
| **RecentWinPct** | Win rate over the last 10 regular season games. Momentum entering March. |
| **AstRate** | Assists per field goal made. Proxy for ball movement and team chemistry. |
| **RecentMargin** | Scoring margin over the last 10 games. Captures recent dominance. |
| **FTRate** | Free throws made per field goal attempt. Aggressiveness and ability to draw fouls. |
| **OppThreePtDependence** | Opponent's reliance on three-point shooting. Teams that live by the three die by the three in March. |
| **WorstMargin** | Worst single-game scoring margin. Reveals a team's floor — blowout losses signal vulnerability. |

### Pipeline

```
Raw Box Scores (124K+ games, 2003-2025)
    ↓
Team-Season Aggregation (efficiency, four factors, tempo)
    ↓
Opponent Adjustment (20-iteration KenPom-style rating)
    ↓
Matchup Features (TeamA - TeamB differences)
    ↓
Ensemble Model (logistic regression + XGBoost, calibrated)
    ↓
Monte Carlo Simulation (50K tournaments)
    ↓
Bracket Optimization (simulated annealing)
```

### Model evaluation

All models validated with Leave-One-Season-Out (LOSO) cross-validation — train on 21 seasons, predict the 22nd, repeat for each season. No data leakage.

| Model | Log-Loss | Calibrated LL | Accuracy |
|---|---|---|---|
| Seed Only | 0.5658 | 0.5520 | 70.1% |
| Logistic Regression | 0.5427 | 0.5293 | 70.7% |
| XGBoost | 0.5411 | 0.5270 | 70.5% |
| **Ensemble** | **0.5359** | **0.5217** | **70.8%** |

The ensemble (50/50 average of logistic regression and XGBoost probabilities) with isotonic regression calibration produces the best calibrated probabilities, which is what matters for Monte Carlo simulation.

## Project structure

```
omastar/
├── run.py                          # Main pipeline CLI
├── generate_dashboard.py           # Generate frontend data
├── config.py                       # Paths and constants
├── pyproject.toml
├── data/
│   ├── raw/                        # Kaggle CSV files
│   └── external/                   # KenPom, EvanMiya, etc.
├── src/
│   ├── data/
│   │   ├── load.py                 # Data loaders
│   │   ├── clean.py                # Normalization (W/L → team perspective)
│   │   └── team_season.py          # Season-level stat aggregation
│   ├── features/
│   │   ├── builder.py              # Feature pipeline orchestration
│   │   ├── matchup.py              # TeamA - TeamB difference features
│   │   ├── seed_matchup.py         # Non-linear seed matchup rates
│   │   ├── adjusted_efficiency.py  # KenPom-style opponent adjustment
│   │   ├── efficiency.py           # SOS, Massey ordinals
│   │   ├── seed.py                 # Seed number extraction
│   │   ├── coach.py                # Coach tournament experience
│   │   ├── conference_tourney.py   # Conference tournament performance
│   │   └── external.py             # KenPom, 538, EvanMiya, Resumes
│   ├── model/
│   │   ├── train.py                # Model definitions, LOSO CV, calibration
│   │   ├── predict.py              # Pairwise probability generation
│   │   ├── evaluate.py             # CV results printing, calibration plots
│   │   ├── tuning.py               # Optuna hyperparameter search
│   │   └── shap_analysis.py        # SHAP feature importance
│   ├── simulation/
│   │   ├── bracket.py              # Tournament bracket structure
│   │   ├── monte_carlo.py          # Vectorized tournament simulation
│   │   └── results.py              # Advancement table formatting
│   └── optimization/
│       ├── optimizer.py            # Simulated annealing bracket optimizer
│       └── scoring.py              # Bracket scoring systems
├── frontend/
│   └── index.html                  # Data-driven narrative dashboard
└── output/
    ├── models/                     # Saved model artifacts
    ├── predictions/                # Probability matrices, advancement tables
    ├── brackets/                   # Optimized bracket picks
    ├── figures/                    # SHAP plots, calibration curves
    └── dashboard_data.json         # Frontend data
```

## CLI reference

```bash
# Full pipeline with evaluation
python run.py --season 2026

# Skip evaluation (use default ensemble)
python run.py --season 2026 --skip-eval

# Hyperparameter tuning (slow, ~30min)
python run.py --season 2026 --tune --tune-trials 50

# Specific model type
python run.py --season 2026 --model xgboost

# Available flags
--season YEAR         # Required. Tournament year to predict.
--simulations N       # Monte Carlo simulations (default: 50,000)
--model TYPE          # logistic, xgboost, lightgbm, ensemble, stacked
--tune                # Run Optuna hyperparameter search
--tune-trials N       # Trials per model (default: 50)
--skip-eval           # Skip LOSO cross-validation
--skip-optimize       # Skip bracket optimization
--skip-shap           # Skip SHAP analysis
--no-calibrate        # Disable isotonic regression calibration
--no-time-weights     # Disable exponential time-decay weighting
```

## Data sources

**Required** (Kaggle March Machine Learning Mania):
- `MRegularSeasonDetailedResults.csv` — game-level box scores
- `MNCAATourneyCompactResults.csv` — tournament outcomes
- `MNCAATourneySeeds.csv` — team seedings
- `MNCAATourneySlots.csv` — bracket structure
- `MTeams.csv` — team names
- `MMasseyOrdinals.csv` — computer rankings

**Optional** (in `data/external/nishaan/`):
- `KenPom Barttorvik.csv` — real KenPom/Barttorvik ratings
- `538 Ratings.csv` — FiveThirtyEight power ratings
- `EvanMiya.csv` — EvanMiya ratings, roster rank
- `Resumes.csv` — Q1/Q2 wins, ELO, WAB rank

**Optional** (in `data/external/`):
- `vegas_lines.csv` — Vegas closing lines (Season, TeamA, TeamB, Spread)

## Design decisions

**Why only 8 features?** Forward feature selection shows that adding a 9th feature doesn't improve LOSO CV log-loss meaningfully. With 1,449 training games (~63 per season), the bias-variance tradeoff strongly favors simpler models. All 70+ available features are computed and remain accessible via `CORE_FEATURES = None` in `train.py`.

**Why ensemble over stacked?** The stacked ensemble (LR + XGB + LightGBM with learned meta-learner) slightly overfits with only 8 features. The simple 50/50 ensemble is more robust.

**Why isotonic calibration?** Raw model probabilities are systematically overconfident on near-50/50 games and underconfident on lopsided matchups. Isotonic regression on out-of-fold predictions corrects this, improving log-loss by ~0.015 for free.

**Why time-decay weighting?** College basketball has evolved (3-point revolution, pace changes). Weighting recent seasons more heavily (0.95^age) improves predictions slightly and consistently.
