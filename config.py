import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the project.

    Call once at the entry-point (run.py / generate_dashboard.py).
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
BRACKETS_DIR = OUTPUT_DIR / "brackets"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Tournament constants
# ---------------------------------------------------------------------------
FIRST_DETAILED_SEASON = 2003  # first season with detailed box score stats in Kaggle dataset
TOURNAMENT_ROUNDS = 6
GAMES_PER_TOURNAMENT = 63  # 64-team single elimination (excluding play-in)
CURRENT_SEASON = 2026  # year to generate predictions / dashboard for

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
DEFAULT_N_SIMULATIONS = 50_000
DEFAULT_RANDOM_SEED = 42

# Exponential time-decay factor for sample weighting during training.
# weight = DECAY^(max_season - season).  0.95 means a season 10 years ago
# gets weight ~0.60.  Chosen to balance recency with having enough effective
# training samples (~1,449 tournament games across 23 seasons).
# Source: standard practice in sports analytics; validated via LOSO CV.
TIME_DECAY = 0.95

# Approximate fraction of free-throw attempts that end a possession.
# Used in the possession-estimation formula:
#   Poss ≈ FGA − OR + TO + 0.475 × FTA
# Source: Oliver, Dean. *Basketball on Paper* (2004), Chapter 14.
# The exact value varies by era (0.44–0.475); 0.475 is the modern consensus.
POSSESSION_FTA_FACTOR = 0.475

# ---------------------------------------------------------------------------
# Scoring systems  (points awarded per correct pick in each round)
# ---------------------------------------------------------------------------
SCORING_STANDARD = [1, 2, 4, 8, 16, 32]  # standard bracket pool scoring
SCORING_ESPN = [10, 20, 40, 80, 160, 320]  # ESPN Tournament Challenge
