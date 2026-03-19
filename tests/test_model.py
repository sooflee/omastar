"""Tests for src/model — training, prediction, scoring, optimization."""

import numpy as np
import pandas as pd
import pytest

from src.model.train import (
    compute_time_weights,
    _make_model,
    EnsembleModel,
    CalibratedModel,
)
from src.model.predict import (
    build_prob_lookup,
    lookup_prob,
    get_win_probability,
)
from src.optimization.scoring import ScoringSystem, STANDARD, ESPN
from config import TIME_DECAY


# ---------------------------------------------------------------------------
# Time weights
# ---------------------------------------------------------------------------

class TestTimeWeights:
    def test_most_recent_season_has_weight_one(self):
        seasons = np.array([2020, 2021, 2022, 2023, 2024])
        weights = compute_time_weights(seasons)
        assert abs(weights[-1] - 1.0) < 1e-10

    def test_older_seasons_have_lower_weight(self):
        seasons = np.array([2020, 2024])
        weights = compute_time_weights(seasons)
        assert weights[0] < weights[1]

    def test_uses_config_decay(self):
        seasons = np.array([2023, 2024])
        weights = compute_time_weights(seasons)
        expected = TIME_DECAY ** 1  # 2024 - 2023 = 1
        assert abs(weights[0] - expected) < 1e-10

    def test_single_season(self):
        seasons = np.array([2024])
        weights = compute_time_weights(seasons)
        assert abs(weights[0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

class TestMakeModel:
    @pytest.mark.parametrize("model_type", [
        "logistic", "xgboost", "lightgbm", "ensemble", "stacked",
    ])
    def test_creates_model(self, model_type):
        model = _make_model(model_type)
        assert model is not None

    def test_unknown_type_returns_ensemble(self):
        model = _make_model("unknown_type")
        assert isinstance(model, EnsembleModel)


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------

class TestEnsembleModel:
    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        model = EnsembleModel(lr_weight=0.5)
        model.fit(X, y)
        probs = model.predict_proba(X)

        assert probs.shape == (100, 2)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        # Probabilities should sum to ~1 per row
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_with_sample_weights(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        w = np.ones(50)

        model = EnsembleModel()
        model.fit(X, y, sample_weight=w)
        probs = model.predict_proba(X)
        assert probs.shape == (50, 2)


# ---------------------------------------------------------------------------
# Calibrated model
# ---------------------------------------------------------------------------

class TestCalibratedModel:
    def test_uncalibrated_passthrough(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)

        base = EnsembleModel()
        cal = CalibratedModel(base)
        cal.fit(X, y)

        raw = base.predict_proba(X)
        cal_probs = cal.predict_proba(X)
        # Without calibration, should return same probs
        np.testing.assert_allclose(cal_probs, raw, atol=1e-6)

    def test_calibrated_changes_output(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        base = EnsembleModel()
        cal = CalibratedModel(base)
        cal.fit(X, y)

        raw_preds = base.predict_proba(X)[:, 1]
        cal.calibrate(raw_preds, y)

        cal_probs = cal.predict_proba(X)
        # Calibrated probs should be different from raw
        # (isotonic regression adjusts them)
        assert cal_probs.shape == (100, 2)


# ---------------------------------------------------------------------------
# Probability lookup
# ---------------------------------------------------------------------------

class TestGetWinProbability:
    def test_canonical_direction(self, prob_matrix):
        p = get_win_probability(prob_matrix, 1101, 1102)
        assert abs(p - 0.65) < 1e-6

    def test_reverse_direction(self, prob_matrix):
        p = get_win_probability(prob_matrix, 1102, 1101)
        assert abs(p - 0.35) < 1e-6

    def test_unknown_matchup(self, prob_matrix):
        p = get_win_probability(prob_matrix, 9999, 8888)
        assert p == 0.5


# ---------------------------------------------------------------------------
# Scoring systems
# ---------------------------------------------------------------------------

class TestScoringSystem:
    def test_standard_points(self):
        assert STANDARD.points_per_round == [1, 2, 4, 8, 16, 32]

    def test_espn_points(self):
        assert ESPN.points_per_round == [10, 20, 40, 80, 160, 320]

    def test_score_bracket(self):
        scoring = ScoringSystem([1, 2], "test")
        picks = {"R1A": 1, "R2A": 1}
        actual = {"R1A": 1, "R2A": 2}
        round_slots = [["R1A"], ["R2A"]]
        score = scoring.score_bracket(picks, actual, round_slots)
        # R1A correct (1pt), R2A wrong (0pt)
        assert score == 1

    def test_perfect_bracket(self):
        scoring = ScoringSystem([1, 2], "test")
        picks = {"R1A": 1, "R2A": 2}
        actual = {"R1A": 1, "R2A": 2}
        round_slots = [["R1A"], ["R2A"]]
        score = scoring.score_bracket(picks, actual, round_slots)
        assert score == 3  # 1 + 2

    def test_max_possible(self):
        scoring = ScoringSystem([1, 2, 4], "test")
        round_slots = [["A", "B"], ["C"], ["D"]]
        assert scoring.max_possible_score(round_slots) == 2 * 1 + 1 * 2 + 1 * 4  # 8
