"""Shared fixtures for the omastar test suite.

All fixtures produce small, self-contained DataFrames so the tests run
without the full Kaggle dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so `import config` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Minimal game data (detailed box-score format coming from Kaggle)
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_detailed_games() -> pd.DataFrame:
    """A handful of fake regular-season detailed results (W/L oriented)."""
    np.random.seed(0)
    rows = []
    teams = [1101, 1102, 1103, 1104]
    for season in [2023, 2024]:
        for day in range(1, 20):
            t1, t2 = np.random.choice(teams, 2, replace=False)
            score_w = np.random.randint(60, 100)
            score_l = np.random.randint(50, score_w)
            base = {
                "Season": season, "DayNum": day, "NumOT": 0,
                "WTeamID": int(t1), "LTeamID": int(t2),
                "WScore": score_w, "LScore": score_l,
                "WLoc": "H",
            }
            for prefix in ("W", "L"):
                base[f"{prefix}FGM"] = np.random.randint(20, 35)
                base[f"{prefix}FGA"] = base[f"{prefix}FGM"] + np.random.randint(10, 30)
                base[f"{prefix}FGM3"] = np.random.randint(3, 12)
                base[f"{prefix}FGA3"] = base[f"{prefix}FGM3"] + np.random.randint(3, 15)
                base[f"{prefix}FTM"] = np.random.randint(5, 20)
                base[f"{prefix}FTA"] = base[f"{prefix}FTM"] + np.random.randint(0, 8)
                base[f"{prefix}OR"] = np.random.randint(5, 15)
                base[f"{prefix}DR"] = np.random.randint(15, 30)
                base[f"{prefix}Ast"] = np.random.randint(8, 22)
                base[f"{prefix}TO"] = np.random.randint(8, 20)
                base[f"{prefix}Stl"] = np.random.randint(3, 12)
                base[f"{prefix}Blk"] = np.random.randint(1, 8)
                base[f"{prefix}PF"] = np.random.randint(10, 25)
            rows.append(base)

    return pd.DataFrame(rows)


@pytest.fixture()
def normalized_games(raw_detailed_games):
    """Team-perspective rows produced by normalize_detailed_results."""
    from src.data.clean import normalize_detailed_results
    return normalize_detailed_results(raw_detailed_games)


@pytest.fixture()
def seeds_df() -> pd.DataFrame:
    """Minimal tournament seeds for 4 teams across 2 seasons."""
    rows = []
    for season in [2023, 2024]:
        for team, seed_str in [(1101, "W01"), (1102, "X02"), (1103, "Y08"), (1104, "Z16")]:
            rows.append({"Season": season, "Seed": seed_str, "TeamID": team})
    return pd.DataFrame(rows)


@pytest.fixture()
def tourney_compact() -> pd.DataFrame:
    """Minimal tournament compact results."""
    return pd.DataFrame([
        {"Season": 2023, "DayNum": 136, "WTeamID": 1101, "LTeamID": 1104,
         "WScore": 80, "LScore": 60, "NumOT": 0},
        {"Season": 2023, "DayNum": 137, "WTeamID": 1101, "LTeamID": 1103,
         "WScore": 75, "LScore": 70, "NumOT": 0},
        {"Season": 2023, "DayNum": 138, "WTeamID": 1102, "LTeamID": 1103,
         "WScore": 72, "LScore": 68, "NumOT": 0},
        {"Season": 2024, "DayNum": 136, "WTeamID": 1102, "LTeamID": 1104,
         "WScore": 85, "LScore": 65, "NumOT": 0},
        {"Season": 2024, "DayNum": 137, "WTeamID": 1101, "LTeamID": 1102,
         "WScore": 78, "LScore": 74, "NumOT": 0},
        {"Season": 2024, "DayNum": 138, "WTeamID": 1101, "LTeamID": 1103,
         "WScore": 90, "LScore": 80, "NumOT": 1},
    ])


@pytest.fixture()
def prob_matrix() -> pd.DataFrame:
    """Pairwise win probabilities for 4 teams."""
    return pd.DataFrame([
        {"TeamA": 1101, "TeamB": 1102, "ProbA": 0.65},
        {"TeamA": 1101, "TeamB": 1103, "ProbA": 0.80},
        {"TeamA": 1101, "TeamB": 1104, "ProbA": 0.95},
        {"TeamA": 1102, "TeamB": 1103, "ProbA": 0.70},
        {"TeamA": 1102, "TeamB": 1104, "ProbA": 0.90},
        {"TeamA": 1103, "TeamB": 1104, "ProbA": 0.75},
    ])


@pytest.fixture()
def slots_df() -> pd.DataFrame:
    """Minimal 4-team bracket using R5/R6 slots to match real tournament structure.

    The real bracket uses R1–R6 slots; simulate_tournament expects R6 for the
    championship.  We skip R1–R4 and use R5 as 'round of 4' (semis) and R6 as
    the final, which is the minimal setup the code accepts.
    """
    return pd.DataFrame([
        {"Season": 2024, "Slot": "R5WX", "StrongSeed": "W01", "WeakSeed": "Z16"},
        {"Season": 2024, "Slot": "R5YZ", "StrongSeed": "X02", "WeakSeed": "Y08"},
        {"Season": 2024, "Slot": "R6CH", "StrongSeed": "R5WX", "WeakSeed": "R5YZ"},
    ])


@pytest.fixture()
def teams_df() -> pd.DataFrame:
    """Minimal teams lookup."""
    return pd.DataFrame([
        {"TeamID": 1101, "TeamName": "Alpha"},
        {"TeamID": 1102, "TeamName": "Bravo"},
        {"TeamID": 1103, "TeamName": "Charlie"},
        {"TeamID": 1104, "TeamName": "Delta"},
    ])
