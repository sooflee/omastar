import logging

import pandas as pd

from config import RAW_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected column schemas for CSV validation.
# Each key is a filename, each value is the set of columns the pipeline needs.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "MTeams.csv": {"TeamID", "TeamName"},
    "MSeasons.csv": {"Season"},
    "MNCAATourneySeeds.csv": {"Season", "Seed", "TeamID"},
    "MNCAATourneySlots.csv": {"Season", "Slot", "StrongSeed", "WeakSeed"},
    "MRegularSeasonDetailedResults.csv": {
        "Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc",
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
        "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
        "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
    },
    "MRegularSeasonCompactResults.csv": {
        "Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore",
    },
    "MNCAATourneyDetailedResults.csv": {
        "Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore",
    },
    "MNCAATourneyCompactResults.csv": {
        "Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore",
    },
    "MMasseyOrdinals.csv": {
        "Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank",
    },
    "MTeamConferences.csv": {"Season", "TeamID"},
    "MTeamCoaches.csv": {"Season", "TeamID", "CoachName", "LastDayNum"},
    "MConferenceTourneyGames.csv": {
        "Season", "ConfAbbrev", "DayNum", "WTeamID", "LTeamID",
    },
}


class DataValidationError(Exception):
    """Raised when a loaded CSV is missing required columns."""


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from RAW_DIR and validate its columns.

    Raises:
        FileNotFoundError: if the file does not exist.
        DataValidationError: if required columns are missing.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    expected = REQUIRED_COLUMNS.get(filename)
    if expected is not None:
        missing = expected - set(df.columns)
        if missing:
            raise DataValidationError(
                f"{filename} is missing required columns: {sorted(missing)}. "
                f"Found columns: {sorted(df.columns)}"
            )

    logger.debug("Loaded %s: %d rows, %d cols", filename, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Public loaders — each wraps _load_csv with the appropriate filename.
# ---------------------------------------------------------------------------

def load_teams() -> pd.DataFrame:
    return _load_csv("MTeams.csv")


def load_seasons() -> pd.DataFrame:
    return _load_csv("MSeasons.csv")


def load_tourney_seeds() -> pd.DataFrame:
    return _load_csv("MNCAATourneySeeds.csv")


def load_tourney_slots() -> pd.DataFrame:
    return _load_csv("MNCAATourneySlots.csv")


def load_regular_season_detailed() -> pd.DataFrame:
    return _load_csv("MRegularSeasonDetailedResults.csv")


def load_regular_season_compact() -> pd.DataFrame:
    return _load_csv("MRegularSeasonCompactResults.csv")


def load_tourney_detailed() -> pd.DataFrame:
    return _load_csv("MNCAATourneyDetailedResults.csv")


def load_tourney_compact() -> pd.DataFrame:
    return _load_csv("MNCAATourneyCompactResults.csv")


def load_massey_ordinals() -> pd.DataFrame:
    return _load_csv("MMasseyOrdinals.csv")


def load_conferences() -> pd.DataFrame:
    return _load_csv("MTeamConferences.csv")


def load_coaches() -> pd.DataFrame:
    return _load_csv("MTeamCoaches.csv")


def load_conference_tourney_games() -> pd.DataFrame:
    return _load_csv("MConferenceTourneyGames.csv")
