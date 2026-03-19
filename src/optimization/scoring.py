from config import SCORING_STANDARD, SCORING_ESPN


class ScoringSystem:
    """Tournament bracket scoring system."""

    def __init__(self, points_per_round: list[int], name: str = "Custom"):
        """
        Args:
            points_per_round: Points awarded per correct pick in each round.
                              Length 6: [R64, R32, S16, E8, F4, Championship].
            name: Scoring system name.
        """
        self.points_per_round = points_per_round
        self.name = name

    def score_bracket(
        self,
        picks: dict[str, int],
        actual: dict[str, int],
        round_slots: list[list[str]],
    ) -> int:
        """Score a bracket against actual results.

        Args:
            picks: dict of slot -> predicted winner team ID.
            actual: dict of slot -> actual winner team ID.
            round_slots: list of lists grouping slots by round.

        Returns:
            Total bracket score.
        """
        total = 0
        for round_num, slots in enumerate(round_slots):
            pts = self.points_per_round[round_num]
            for slot in slots:
                if slot in picks and slot in actual:
                    if picks[slot] == actual[slot]:
                        total += pts
        return total

    def max_possible_score(self, round_slots: list[list[str]]) -> int:
        """Maximum possible score (all picks correct)."""
        total = 0
        for round_num, slots in enumerate(round_slots):
            total += len(slots) * self.points_per_round[round_num]
        return total


STANDARD = ScoringSystem(SCORING_STANDARD, "Standard (1-2-4-8-16-32)")
ESPN = ScoringSystem(SCORING_ESPN, "ESPN (10-20-40-80-160-320)")


class UpsetBonusScoring(ScoringSystem):
    """Scoring where correct upset picks earn bonus points equal to the winning seed."""

    def __init__(self, base_points: list[int] | None = None):
        super().__init__(base_points or [1, 2, 4, 8, 16, 32], "Upset Bonus")

    def score_bracket(
        self,
        picks: dict[str, int],
        actual: dict[str, int],
        round_slots: list[list[str]],
        seed_map: dict[int, int] | None = None,
    ) -> int:
        """Score with upset bonuses.

        Args:
            seed_map: dict of team_id -> numeric seed (needed for bonus calculation).
        """
        total = 0
        for round_num, slots in enumerate(round_slots):
            pts = self.points_per_round[round_num]
            for slot in slots:
                if slot in picks and slot in actual:
                    if picks[slot] == actual[slot]:
                        total += pts
                        # Add seed bonus for upsets
                        if seed_map and picks[slot] in seed_map:
                            total += seed_map[picks[slot]]
        return total
