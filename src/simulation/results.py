import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def advancement_table(
    sim_results: dict,
    teams: pd.DataFrame,
    seeds: pd.DataFrame,
    season: int,
    top_n: int = 30,
    has_playin: bool = True,
    show_ci: bool = True,
) -> pd.DataFrame:
    """Create a readable advancement probability table.

    Args:
        sim_results: Output from simulate_tournament.
        teams: MTeams DataFrame with TeamID, TeamName.
        seeds: MNCAATourneySeeds DataFrame.
        season: Tournament season.
        top_n: Number of teams to show (sorted by championship probability).
        has_playin: Whether round 0 is play-in (shifts indices by 1).
        show_ci: Whether to include 95% CI columns (± values).

    Returns:
        DataFrame with team name, seed, and probability of reaching each round.
    """
    adv = sim_results["advancement"]
    adv_ci = sim_results.get("advancement_ci", {})
    champ = sim_results["champion_probs"]
    champ_ci = sim_results.get("champion_ci", {})

    season_seeds = seeds[seeds["Season"] == season].copy()
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))

    # Determine round indices based on whether play-in exists
    offset = 1 if has_playin else 0

    rows = []
    for team_id in adv:
        seed_row = season_seeds[season_seeds["TeamID"] == team_id]
        seed_str = seed_row["Seed"].values[0] if len(seed_row) > 0 else "?"

        n_rounds = len(adv[team_id])
        row = {
            "Team": team_names.get(team_id, str(team_id)),
            "Seed": seed_str,
        }

        round_names = ["R64", "R32", "S16", "E8", "F4", "Champ"]
        team_ci = adv_ci.get(team_id)
        for i, name in enumerate(round_names):
            idx = i + offset
            row[name] = adv[team_id][idx] if idx < n_rounds else 0.0
            if show_ci and team_ci is not None:
                row[f"{name}_CI"] = team_ci[idx] if idx < n_rounds else 0.0

        row["Title"] = champ.get(team_id, 0)
        if show_ci:
            row["Title_CI"] = champ_ci.get(team_id, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Title", ascending=False).head(top_n)
    df = df.reset_index(drop=True)

    # Format as percentages
    pct_cols = ["R64", "R32", "S16", "E8", "F4", "Champ", "Title"]
    ci_cols = [f"{c}_CI" for c in pct_cols]
    for col in pct_cols + ci_cols:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1)

    return df


def print_advancement_table(table: pd.DataFrame) -> None:
    """Pretty-print the advancement table.

    If CI columns are present, displays them as ± values next to the
    probability columns.
    """
    # Build a display-friendly copy
    display = table.copy()

    # Merge probability and CI columns into "value ± ci" strings
    pct_cols = ["R64", "R32", "S16", "E8", "F4", "Champ", "Title"]
    for col in pct_cols:
        ci_col = f"{col}_CI"
        if ci_col in display.columns:
            display[col] = display.apply(
                lambda r, c=col, cc=ci_col: f"{r[c]:5.1f}±{r[cc]:.1f}", axis=1
            )
            display = display.drop(columns=[ci_col])

    logger.info("\n" + "=" * 85)
    logger.info("Tournament Advancement Probabilities (%%, 95%% CI)")
    logger.info("=" * 85)
    logger.info("\n%s\n", display.to_string(index=False))
