import pandas as pd
from src.data.clean import parse_seed


class Bracket:
    """Represents a tournament bracket structure.

    Parses MNCAATourneySlots to build the bracket tree and maps seeds to teams.
    """

    ROUND_NAMES = [
        "Play-In", "Round of 64", "Round of 32", "Sweet 16",
        "Elite 8", "Final Four", "Championship",
    ]

    def __init__(self, slots: pd.DataFrame, seeds: pd.DataFrame, season: int):
        """Initialize bracket for a given season.

        Args:
            slots: MNCAATourneySlots DataFrame.
            seeds: MNCAATourneySeeds DataFrame.
            season: Tournament year.
        """
        self.season = season
        self.slots = slots[slots["Season"] == season].copy()
        self.seeds = seeds[seeds["Season"] == season].copy()

        # Map seed strings to team IDs
        self.seed_to_team = dict(
            zip(self.seeds["Seed"], self.seeds["TeamID"])
        )
        self.team_to_seed = {v: k for k, v in self.seed_to_team.items()}

        # Build slot structure
        self._build_slot_tree()

    def _build_slot_tree(self):
        """Build the bracket tree from slot data."""
        self.slot_sources = {}  # slot -> (source1, source2)
        self.slot_winner = {}   # slot -> winning team ID

        for _, row in self.slots.iterrows():
            slot = row["Slot"]
            strong = row["StrongSeed"]
            weak = row["WeakSeed"]
            self.slot_sources[slot] = (strong, weak)

    def get_team_for_source(self, source: str) -> int | None:
        """Resolve a source (seed string or slot name) to a team ID."""
        # If it's a seed string, look up directly
        if source in self.seed_to_team:
            return self.seed_to_team[source]
        # If it's a slot, look up the winner
        return self.slot_winner.get(source)

    def get_matchup(self, slot: str) -> tuple[int | None, int | None]:
        """Get the two teams playing in a given slot."""
        if slot not in self.slot_sources:
            return None, None
        source1, source2 = self.slot_sources[slot]
        return self.get_team_for_source(source1), self.get_team_for_source(source2)

    def set_winner(self, slot: str, team_id: int):
        """Record the winner of a slot."""
        self.slot_winner[slot] = team_id

    def get_playin_slots(self) -> list[str]:
        """Get play-in game slots (not prefixed with R)."""
        return [slot for slot in self.slot_sources if not slot.startswith("R")]

    def get_round_slots(self) -> list[list[str]]:
        """Group slots by tournament round.

        Returns list of lists, where index 0 = play-in, 1 = Round of 64, etc.
        Play-in slots (non-R-prefixed) are included as round 0 so the
        simulation resolves them before the main bracket.
        """
        rounds = [[] for _ in range(7)]  # 0=playin, 1-6=R1-R6

        for slot in self.slot_sources:
            if slot.startswith("R1"):
                rounds[1].append(slot)
            elif slot.startswith("R2"):
                rounds[2].append(slot)
            elif slot.startswith("R3"):
                rounds[3].append(slot)
            elif slot.startswith("R4"):
                rounds[4].append(slot)
            elif slot.startswith("R5"):
                rounds[5].append(slot)
            elif slot.startswith("R6"):
                rounds[6].append(slot)
            else:
                rounds[0].append(slot)  # play-in

        # Remove empty play-in round if no play-in games
        if not rounds[0]:
            rounds = rounds[1:]

        return rounds

    def get_all_team_ids(self) -> list[int]:
        """Get all team IDs in this tournament."""
        return sorted(self.seed_to_team.values())

    def reset(self):
        """Clear all winners for a fresh simulation."""
        self.slot_winner = {}

    def get_champion(self) -> int | None:
        """Get the championship game winner."""
        for slot in self.slot_sources:
            if slot.startswith("R6"):
                return self.slot_winner.get(slot)
        return None
