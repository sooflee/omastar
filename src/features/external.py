"""Integrate external data sources (KenPom, 538, EvanMiya, etc.) into the feature pipeline."""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from config import EXTERNAL_DIR

logger = logging.getLogger(__name__)

NISHAAN_DIR = EXTERNAL_DIR / "nishaan"


def _build_name_map(our_teams: pd.DataFrame, external_names: list[str]) -> dict[str, int]:
    """Build a mapping from external team names to our TeamID.

    Uses exact match first, then fuzzy matching for common discrepancies.
    """
    # Build lookup from our names
    our_lookup = {}
    for _, row in our_teams.iterrows():
        our_lookup[row["TeamName"].lower().strip()] = row["TeamID"]

    # Common name aliases between datasets
    aliases = {
        "uconn": "connecticut",
        "unc": "north carolina",
        "nc state": "nc state",
        "lsu": "lsu",
        "vcu": "vcu",
        "ucf": "ucf",
        "usc": "usc",
        "smu": "smu",
        "byu": "byu",
        "ole miss": "mississippi",
        "umass": "massachusetts",
        "miami": "miami fl",
        "miami fl": "miami fl",
        "pitt": "pittsburgh",
        "saint mary's": "st mary's ca",
        "saint mary's (ca)": "st mary's ca",
        "st. mary's": "st mary's ca",
        "saint joseph's": "st joseph's pa",
        "texas a&m": "texas a&m",
        "tcu": "tcu",
        "cal baptist": "cal baptist",
        "north dakota st": "n dakota st",
        "north dakota state": "n dakota st",
        "south florida": "south florida",
        "northern iowa": "northern iowa",
        "uc davis": "uc davis",
        "uc irvine": "uc irvine",
        "uc santa barbara": "uc santa barb",
        "utah state": "utah st",
        "ohio state": "ohio st",
        "michigan state": "michigan st",
        "iowa state": "iowa st",
        "penn state": "penn st",
        "wright state": "wright st",
        "kent state": "kent st",
        "texas tech": "texas tech",
        "virginia tech": "virginia tech",
        "queens nc": "queens nc",
        "liu brooklyn": "liu brooklyn",
        "liu": "liu brooklyn",
        "prairie view a&m": "prairie view",
        "prairie view": "prairie view",
        "mount st. mary's": "mt st mary's",
        "st. louis": "st louis",
        "saint louis": "st louis",
        "santa clara": "santa clara",
        "high point": "high point",
        "mcneese": "mcneese st",
        "mcneese state": "mcneese st",
    }

    name_to_id = {}
    for ext_name in external_names:
        lower = ext_name.lower().strip()

        # Direct match
        if lower in our_lookup:
            name_to_id[ext_name] = our_lookup[lower]
            continue

        # Alias match
        alias = aliases.get(lower)
        if alias and alias in our_lookup:
            name_to_id[ext_name] = our_lookup[alias]
            continue

        # Partial match (external name contained in our name or vice versa)
        for our_name, tid in our_lookup.items():
            if lower in our_name or our_name in lower:
                name_to_id[ext_name] = tid
                break

    return name_to_id


def _map_teams(df: pd.DataFrame, our_teams: pd.DataFrame) -> pd.DataFrame:
    """Map external TEAM names to TeamID and rename YEAR to Season."""
    name_map = _build_name_map(our_teams, df["TEAM"].unique().tolist())
    df = df.copy()
    df["TeamID"] = df["TEAM"].map(name_map)
    df = df.dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    df["Season"] = df["YEAR"]
    return df


def load_kenpom_barttorvik(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load real KenPom/Barttorvik data from Nishaan Amin dataset."""
    path = NISHAAN_DIR / "KenPom Barttorvik.csv"
    if not path.exists():
        return None

    kp = _map_teams(pd.read_csv(path), our_teams)

    result = pd.DataFrame({
        "Season": kp["Season"],
        "TeamID": kp["TeamID"],
        "RealKAdjO": kp["KADJ O"],
        "RealKAdjD": kp["KADJ D"],
        "RealKAdjEM": kp["KADJ EM"],
        "RealKAdjT": kp["KADJ T"] if "KADJ T" in kp.columns else kp.get("K TEMPO"),
        "BAdjEM": kp["BADJ EM"],
        "BAdjO": kp["BADJ O"],
        "BAdjD": kp["BADJ D"],
    })

    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_538_ratings(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load FiveThirtyEight power ratings."""
    path = NISHAAN_DIR / "538 Ratings.csv"
    if not path.exists():
        return None

    r538 = _map_teams(pd.read_csv(path), our_teams)

    result = pd.DataFrame({
        "Season": r538["Season"],
        "TeamID": r538["TeamID"],
        "PowerRating538": r538["POWER RATING"],
    })

    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_evanmiya(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load EvanMiya ratings.

    Returns DataFrame with Season, TeamID, EvanMiyaRating, RosterRank,
    KillshotsMargin, or None if file not found.
    """
    path = NISHAAN_DIR / "EvanMiya.csv"
    if not path.exists():
        return None

    em = _map_teams(pd.read_csv(path), our_teams)

    cols = {"Season": em["Season"], "TeamID": em["TeamID"]}

    if "RELATIVE RATING" in em.columns:
        cols["EvanMiyaRating"] = pd.to_numeric(em["RELATIVE RATING"], errors="coerce")
    if "ROSTER RANK" in em.columns:
        cols["RosterRank"] = pd.to_numeric(em["ROSTER RANK"], errors="coerce")
    if "KILLSHOTS MARGIN" in em.columns:
        cols["KillshotsMargin"] = pd.to_numeric(em["KILLSHOTS MARGIN"], errors="coerce")

    result = pd.DataFrame(cols)
    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_resume_data(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load resume/quality wins data (Q1/Q2 wins, ELO, WAB rank).

    Returns DataFrame with Season, TeamID, Q1Wins, Q2Wins, Elo, WABRank,
    or None if file not found.
    """
    path = NISHAAN_DIR / "Resumes.csv"
    if not path.exists():
        return None

    res = _map_teams(pd.read_csv(path), our_teams)

    cols = {"Season": res["Season"], "TeamID": res["TeamID"]}

    if "Q1 W" in res.columns:
        cols["Q1Wins"] = pd.to_numeric(res["Q1 W"], errors="coerce")
    if "Q2 W" in res.columns:
        cols["Q2Wins"] = pd.to_numeric(res["Q2 W"], errors="coerce")
    if "ELO" in res.columns:
        cols["Elo"] = pd.to_numeric(res["ELO"], errors="coerce")
    if "WAB RANK" in res.columns:
        cols["WABRank"] = pd.to_numeric(res["WAB RANK"], errors="coerce")

    result = pd.DataFrame(cols)
    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_barttorvik_away_neutral(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load Barttorvik away/neutral-site efficiency metrics.

    Tournament games are played on neutral courts, so away/neutral efficiency
    is more predictive than overall efficiency (which includes home games).

    Returns DataFrame with Season, TeamID, and away/neutral metrics.
    """
    path = NISHAAN_DIR / "Barttorvik Away-Neutral.csv"
    if not path.exists():
        return None

    ban = _map_teams(pd.read_csv(path), our_teams)

    result = pd.DataFrame({
        "Season": ban["Season"],
        "TeamID": ban["TeamID"],
        "AwayNeutralAdjEM": pd.to_numeric(ban["BADJ EM"], errors="coerce"),
        "AwayNeutralAdjO": pd.to_numeric(ban["BADJ O"], errors="coerce"),
        "AwayNeutralAdjD": pd.to_numeric(ban["BADJ D"], errors="coerce"),
        "Talent": pd.to_numeric(ban["TALENT"], errors="coerce"),
        "Experience": pd.to_numeric(ban["EXP"], errors="coerce"),
        "AvgHeight": pd.to_numeric(ban["AVG HGT"], errors="coerce"),
        "EffHeight": pd.to_numeric(ban["EFF HGT"], errors="coerce"),
        "EliteSOS": pd.to_numeric(ban["ELITE SOS"], errors="coerce"),
    })
    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_preseason_improvement(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load preseason-to-current KenPom improvement.

    Teams that exceed preseason expectations may be on a genuine upward
    trajectory, while teams that disappoint may be regressing.

    Returns DataFrame with Season, TeamID, and improvement metrics.
    """
    path = NISHAAN_DIR / "KenPom Preseason.csv"
    if not path.exists():
        return None

    kpp = _map_teams(pd.read_csv(path), our_teams)

    result = pd.DataFrame({
        "Season": kpp["Season"],
        "TeamID": kpp["TeamID"],
        "PreseasonAdjEM": pd.to_numeric(kpp["PRESEASON KADJ EM"], errors="coerce"),
        "AdjEMImprovement": pd.to_numeric(kpp["KADJ EM CHANGE"], errors="coerce"),
        "PreseasonRank": pd.to_numeric(kpp["PRESEASON KADJ EM RANK"], errors="coerce"),
        "RankImprovement": pd.to_numeric(kpp["KADJ EM RANK CHANGE"], errors="coerce"),
    })
    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_injury_rank(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load EvanMiya injury rank (lower = healthier/more available).

    Returns DataFrame with Season, TeamID, InjuryRank.
    """
    path = NISHAAN_DIR / "EvanMiya.csv"
    if not path.exists():
        return None

    em = _map_teams(pd.read_csv(path), our_teams)

    if "INJURY RANK" not in em.columns:
        return None

    result = pd.DataFrame({
        "Season": em["Season"],
        "TeamID": em["TeamID"],
        "InjuryRank": pd.to_numeric(em["INJURY RANK"], errors="coerce"),
    })
    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_ap_poll_final(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load final pre-tournament AP Poll ranking.

    Takes the last week's AP rank for each team-season as a pre-tournament
    strength signal.

    Returns DataFrame with Season, TeamID, APFinalRank, APFinalVotes.
    """
    path = NISHAAN_DIR / "AP Poll Data.csv"
    if not path.exists():
        return None

    ap = pd.read_csv(path)
    # Take the final week per team per season
    ap = ap.sort_values("WEEK")
    final = ap.groupby(["YEAR", "TEAM"]).last().reset_index()

    final = _map_teams(final, our_teams)

    result = pd.DataFrame({
        "Season": final["Season"],
        "TeamID": final["TeamID"],
        "APFinalRank": pd.to_numeric(final["AP RANK"], errors="coerce"),
        "APFinalVotes": pd.to_numeric(final["AP VOTES"], errors="coerce"),
    })
    # Unranked teams get rank 0 or NaN — set to a high value
    result["APFinalRank"] = result["APFinalRank"].fillna(200)
    result["APFinalVotes"] = result["APFinalVotes"].fillna(0)

    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_rppf_ratings(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load RPPF independent rating system.

    Returns DataFrame with Season, TeamID, RPPFRating, RPPFAdjEM.
    """
    path = NISHAAN_DIR / "RPPF Ratings.csv"
    if not path.exists():
        return None

    rppf = _map_teams(pd.read_csv(path), our_teams)

    result = pd.DataFrame({
        "Season": rppf["Season"],
        "TeamID": rppf["TeamID"],
        "RPPFRating": pd.to_numeric(rppf["RPPF RATING"], errors="coerce"),
        "RPPFAdjEM": pd.to_numeric(rppf["RADJ EM"], errors="coerce"),
        "RPPFSOS": pd.to_numeric(rppf["R SOS"], errors="coerce"),
    })
    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_shooting_splits(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load shot quality/distribution data.

    Captures how teams generate their offense (dunks vs. close 2s vs. far 2s
    vs. 3s) and how they defend each shot type. Shot quality matters in
    tournament play where pace slows and half-court offense dominates.

    Returns DataFrame with Season, TeamID, and shot quality features.
    """
    path = NISHAAN_DIR / "Shooting Splits.csv"
    if not path.exists():
        return None

    ss = _map_teams(pd.read_csv(path), our_teams)

    result = pd.DataFrame({
        "Season": ss["Season"],
        "TeamID": ss["TeamID"],
        # Offensive shot quality
        "DunksShare": pd.to_numeric(ss["DUNKS SHARE"], errors="coerce"),
        "DunksFGPct": pd.to_numeric(ss["DUNKS FG%"], errors="coerce"),
        "CloseTwosShare": pd.to_numeric(ss["CLOSE TWOS SHARE"], errors="coerce"),
        "CloseTwosFGPct": pd.to_numeric(ss["CLOSE TWOS FG%"], errors="coerce"),
        "ThreesShare": pd.to_numeric(ss["THREES SHARE"], errors="coerce"),
        # Defensive shot quality
        "DunksDShare": pd.to_numeric(ss["DUNKS D SHARE"], errors="coerce"),
        "CloseTwosDFGPct": pd.to_numeric(ss["CLOSE TWOS FG%D"], errors="coerce"),
        "ThreesDFGPct": pd.to_numeric(ss["THREES FG%D"], errors="coerce"),
    })
    return result.drop_duplicates(subset=["Season", "TeamID"], keep="first")


def load_vegas_lines() -> pd.DataFrame | None:
    """Load Vegas closing lines if available.

    Expected format: CSV with Season, TeamA, TeamB, Spread columns
    at data/external/vegas_lines.csv.

    Returns DataFrame or None if not available.
    """
    path = EXTERNAL_DIR / "vegas_lines.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_all_external(our_teams: pd.DataFrame) -> pd.DataFrame | None:
    """Load and merge all external data sources.

    Logs which sources were loaded and which were skipped so the user
    can assess their impact on predictions.
    """
    loaders = [
        ("KenPom/Barttorvik", load_kenpom_barttorvik),
        ("538 Ratings", load_538_ratings),
        ("EvanMiya", load_evanmiya),
        ("Resumes", load_resume_data),
        ("Barttorvik Away-Neutral", load_barttorvik_away_neutral),
        ("KenPom Preseason", load_preseason_improvement),
        ("InjuryRank", load_injury_rank),
        ("AP Poll", load_ap_poll_final),
        ("RPPF Ratings", load_rppf_ratings),
        ("Shooting Splits", load_shooting_splits),
    ]

    result = None
    loaded = []
    skipped = []

    for name, loader in loaders:
        try:
            data = loader(our_teams)
        except Exception as e:
            logger.warning("  External source '%s' failed to load: %s", name, e)
            skipped.append(name)
            continue

        if data is None:
            skipped.append(name)
            continue

        loaded.append(f"{name} ({len(data)} rows)")
        if result is None:
            result = data
        else:
            result = result.merge(data, on=["Season", "TeamID"], how="outer")

    if loaded:
        logger.info("  External data loaded: %s", ", ".join(loaded))
    if skipped:
        logger.info("  External data not available (skipped): %s", ", ".join(skipped))

    return result
