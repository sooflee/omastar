import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import DEFAULT_RANDOM_SEED, TIME_DECAY

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "min_child_weight": 5,
    "random_state": DEFAULT_RANDOM_SEED,
}

DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "min_child_samples": 10,
    "random_state": DEFAULT_RANDOM_SEED,
    "verbose": -1,
}

DEFAULT_CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3.0,
    "random_seed": DEFAULT_RANDOM_SEED,
    "verbose": 0,
}

# Optimal feature set via greedy forward selection (LOSO CV).
# 15 features selected from 109 candidates; each added only if it improved log-loss
# or improved accuracy without hurting log-loss.
CORE_FEATURES = [
    "AdjEM_diff",                   # opponent-adjusted efficiency margin
    "RecentWinPct_diff",            # last 10 games win rate (momentum)
    "AstRate_diff",                 # assist rate (team chemistry)
    "RecentMargin_diff",            # last 10 games scoring margin
    "FTRate_diff",                  # free throw rate (aggressiveness / drawing fouls)
    "OppThreePtDependence_diff",    # opponent 3PT reliance (style vulnerability)
    "WorstMargin_diff",             # worst game margin (performance floor)
    "ProgramDeepRuns_diff",         # Sweet 16+ appearances in last 5 years
    "WinTrendLate_diff",            # late-season vs early-season win rate trend
    "ConfDepth_diff",               # number of quality teams in conference
    "InjuryRank_diff",              # team health/availability (EvanMiya)
    "APFinalVotes_diff",            # pre-tournament AP poll votes (crowd wisdom)
    "AdjEMImprovement_diff",        # preseason-to-current KenPom EM improvement
    "RankImprovement_diff",         # preseason-to-current rank improvement
    "PreseasonAdjEM_diff",          # preseason KenPom adjusted efficiency margin
]

# Re-export for convenience; authoritative value lives in config.py


class EnsembleModel:
    """Ensemble of logistic regression and XGBoost with averaged probabilities."""

    def __init__(self, lr_weight=0.5):
        self.lr_weight = lr_weight
        self.lr = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        ])
        self.xgb = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(**DEFAULT_XGB_PARAMS)),
        ])

    def fit(self, X, y, sample_weight=None):
        kw = {}
        if sample_weight is not None:
            kw["model__sample_weight"] = sample_weight
        self.lr.fit(X, y, **kw)
        self.xgb.fit(X, y, **kw)
        return self

    def predict_proba(self, X):
        lr_probs = self.lr.predict_proba(X)
        xgb_probs = self.xgb.predict_proba(X)
        return self.lr_weight * lr_probs + (1 - self.lr_weight) * xgb_probs


class StackedEnsemble:
    """Stacked ensemble with learned meta-learner weights.

    Base models: LogisticRegression, XGBoost, LightGBM (+ CatBoost if available).
    Meta-learner: LogisticRegression trained on out-of-fold predictions.
    """

    def __init__(self, n_inner_folds=5):
        self.n_inner_folds = n_inner_folds
        self.base_models = self._make_base_models()
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        self.fitted_models = None

    def _make_base_models(self):
        models = [
            ("lr", Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
            ])),
            ("xgb", Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBClassifier(**DEFAULT_XGB_PARAMS)),
            ])),
            ("lgbm", Pipeline([
                ("scaler", StandardScaler()),
                ("model", LGBMClassifier(**DEFAULT_LGBM_PARAMS)),
            ])),
        ]
        if HAS_CATBOOST:
            models.append(("catboost", Pipeline([
                ("scaler", StandardScaler()),
                ("model", CatBoostClassifier(**DEFAULT_CATBOOST_PARAMS)),
            ])))
        return models

    def fit(self, X, y, sample_weight=None):
        n_samples = len(y)
        n_models = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models))

        kf = StratifiedKFold(
            n_splits=self.n_inner_folds, shuffle=True,
            random_state=DEFAULT_RANDOM_SEED,
        )

        kw = {}
        if sample_weight is not None:
            kw["model__sample_weight"] = sample_weight

        # Collect OOF predictions
        for fold_train_idx, fold_val_idx in kf.split(X, y):
            X_fold_train, y_fold_train = X[fold_train_idx], y[fold_train_idx]
            X_fold_val = X[fold_val_idx]

            fold_kw = {}
            if sample_weight is not None:
                fold_kw["model__sample_weight"] = sample_weight[fold_train_idx]

            for i, (name, model) in enumerate(self.base_models):
                m = clone(model)
                m.fit(X_fold_train, y_fold_train, **fold_kw)
                oof_preds[fold_val_idx, i] = m.predict_proba(X_fold_val)[:, 1]

        # Fit meta-learner on OOF predictions
        self.meta_learner.fit(oof_preds, y)

        # Refit all base models on full data
        self.fitted_models = []
        for name, model in self.base_models:
            m = clone(model)
            m.fit(X, y, **kw)
            self.fitted_models.append((name, m))

        return self

    def predict_proba(self, X):
        base_preds = np.column_stack([
            model.predict_proba(X)[:, 1]
            for _, model in self.fitted_models
        ])
        meta_probs = self.meta_learner.predict_proba(base_preds)
        return meta_probs


class CalibratedModel:
    """Wraps a model with isotonic regression calibration."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip",
        )
        self._is_calibrated = False

    def fit(self, X, y, sample_weight=None):
        self.base_model.fit(X, y, sample_weight=sample_weight)
        return self

    def calibrate(self, raw_preds, true_labels):
        """Fit calibration on out-of-fold predictions."""
        self.calibrator.fit(raw_preds, true_labels)
        self._is_calibrated = True

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)
        if not self._is_calibrated:
            return raw
        raw_p1 = raw[:, 1]
        calibrated = self.calibrator.predict(raw_p1)
        return np.column_stack([1 - calibrated, calibrated])


def compute_time_weights(seasons: np.ndarray, decay: float = TIME_DECAY) -> np.ndarray:
    """Compute exponential time decay weights favoring recent seasons."""
    max_season = seasons.max()
    return decay ** (max_season - seasons)


def _make_model(model_type: str = "ensemble"):
    """Create a model pipeline with scaling."""
    if model_type == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        ])
    elif model_type == "xgboost":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(**DEFAULT_XGB_PARAMS)),
        ])
    elif model_type == "lightgbm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(**DEFAULT_LGBM_PARAMS)),
        ])
    elif model_type == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("catboost not installed. Install with: pip install catboost")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", CatBoostClassifier(**DEFAULT_CATBOOST_PARAMS)),
        ])
    elif model_type == "stacked":
        return StackedEnsemble()
    else:  # ensemble (fixed 50/50)
        return EnsembleModel(lr_weight=0.5)


def _fit_model(model, X, y, sample_weight=None):
    """Fit a model, handling sample_weight for different model types."""
    if isinstance(model, (EnsembleModel, StackedEnsemble, CalibratedModel)):
        model.fit(X, y, sample_weight=sample_weight)
    elif isinstance(model, Pipeline):
        kw = {}
        if sample_weight is not None:
            kw["model__sample_weight"] = sample_weight
        model.fit(X, y, **kw)
    else:
        model.fit(X, y)
    return model


def train_loso_cv(
    features: pd.DataFrame,
    target: pd.Series,
    feature_cols: list[str],
    model_type: str = "ensemble",
    use_time_weights: bool = True,
    calibrate: bool = True,
) -> dict:
    """Train model using Leave-One-Season-Out cross-validation.

    Args:
        features: Feature DataFrame with Season, TeamA, TeamB, and feature columns.
        target: Binary target (1 if TeamA won).
        feature_cols: List of feature column names to use.
        model_type: One of "logistic", "xgboost", "lightgbm", "catboost",
                     "ensemble", "stacked".
        use_time_weights: Whether to apply exponential time decay weights.
        calibrate: Whether to calibrate probabilities post-hoc.

    Returns:
        Dictionary with per-season and aggregate results.
    """
    seasons = sorted(features["Season"].unique())

    all_preds = []
    all_targets = []
    per_season = []

    for hold_out in seasons:
        train_mask = features["Season"] != hold_out
        val_mask = features["Season"] == hold_out

        X_train = np.nan_to_num(features.loc[train_mask, feature_cols].values, nan=0.0)
        y_train = target.loc[train_mask].values
        X_val = np.nan_to_num(features.loc[val_mask, feature_cols].values, nan=0.0)
        y_val = target.loc[val_mask].values

        if len(y_val) == 0:
            continue

        # Compute time weights for training data
        sample_weight = None
        if use_time_weights:
            train_seasons = features.loc[train_mask, "Season"].values
            sample_weight = compute_time_weights(train_seasons)

        model = _make_model(model_type)
        _fit_model(model, X_train, y_train, sample_weight=sample_weight)

        preds = model.predict_proba(X_val)[:, 1]
        preds = np.clip(preds, 0.01, 0.99)

        season_logloss = log_loss(y_val, preds)
        season_brier = brier_score_loss(y_val, preds)
        season_acc = np.mean((preds > 0.5) == y_val)

        per_season.append({
            "season": hold_out,
            "logloss": season_logloss,
            "brier": season_brier,
            "accuracy": season_acc,
            "n_games": len(y_val),
            "preds": preds,
            "targets": y_val,
        })

        all_preds.extend(preds)
        all_targets.extend(y_val)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Post-hoc calibration on OOF predictions
    calibrator = None
    if calibrate and len(all_preds) > 0:
        calibrator = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip",
        )
        calibrator.fit(all_preds, all_targets)
        calibrated_preds = calibrator.predict(all_preds)

        # Report calibrated metrics
        cal_logloss = log_loss(all_targets, calibrated_preds)
        cal_brier = brier_score_loss(all_targets, calibrated_preds)
    else:
        calibrated_preds = all_preds
        cal_logloss = None
        cal_brier = None

    return {
        "per_season": per_season,
        "overall_logloss": log_loss(all_targets, all_preds),
        "overall_brier": brier_score_loss(all_targets, all_preds),
        "overall_accuracy": np.mean((all_preds > 0.5) == all_targets),
        "calibrated_logloss": cal_logloss,
        "calibrated_brier": cal_brier,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "calibrator": calibrator,
    }


def train_final_model(
    features: pd.DataFrame,
    target: pd.Series,
    feature_cols: list[str],
    model_type: str = "ensemble",
    use_time_weights: bool = True,
    calibrator=None,
):
    """Train final model on all available data.

    Args:
        features: Feature DataFrame.
        target: Binary target.
        feature_cols: Feature column names.
        model_type: Model type string.
        use_time_weights: Whether to apply time decay weights.
        calibrator: Pre-fitted IsotonicRegression from CV. If provided,
                    wraps the model in CalibratedModel.

    Returns:
        Trained model (optionally wrapped with calibration).
    """
    X = np.nan_to_num(features[feature_cols].values, nan=0.0)
    y = target.values

    sample_weight = None
    if use_time_weights:
        seasons = features["Season"].values
        sample_weight = compute_time_weights(seasons)

    model = _make_model(model_type)
    _fit_model(model, X, y, sample_weight=sample_weight)

    if calibrator is not None:
        cal_model = CalibratedModel(model)
        cal_model.calibrator = calibrator
        cal_model._is_calibrated = True
        return cal_model

    return model
