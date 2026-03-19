"""Hyperparameter tuning with Optuna using LOSO cross-validation."""
import logging
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import DEFAULT_RANDOM_SEED
from src.model.train import compute_time_weights

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


def _evaluate_params(features, target, feature_cols, make_model_fn, use_time_weights=True):
    """Evaluate a model via LOSO CV and return overall log-loss."""
    seasons = sorted(features["Season"].unique())
    all_preds = []
    all_targets = []

    for hold_out in seasons:
        train_mask = features["Season"] != hold_out
        val_mask = features["Season"] == hold_out

        X_train = features.loc[train_mask, feature_cols].fillna(0)
        y_train = target.loc[train_mask].values
        X_val = features.loc[val_mask, feature_cols].fillna(0)
        y_val = target.loc[val_mask].values

        if len(y_val) == 0:
            continue

        sample_weight = None
        if use_time_weights:
            train_seasons = features.loc[train_mask, "Season"].values
            sample_weight = compute_time_weights(train_seasons)

        model = make_model_fn()
        kw = {}
        if sample_weight is not None:
            kw["model__sample_weight"] = sample_weight
        model.fit(X_train, y_train, **kw)

        preds = np.clip(model.predict_proba(X_val)[:, 1], 0.01, 0.99)
        all_preds.extend(preds)
        all_targets.extend(y_val)

    return log_loss(all_targets, all_preds)


def tune_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    feature_cols: list[str],
    n_trials: int = 50,
) -> dict:
    """Tune XGBoost hyperparameters with Optuna.

    Returns:
        Best parameters dictionary.
    """
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "random_state": DEFAULT_RANDOM_SEED,
        }

        def make_model():
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBClassifier(**params)),
            ])

        return _evaluate_params(features, target, feature_cols, make_model)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["objective"] = "binary:logistic"
    best["eval_metric"] = "logloss"
    best["random_state"] = DEFAULT_RANDOM_SEED

    logger.info("  Best XGBoost log-loss: %.4f", study.best_value)
    logger.info("  Best params: %s", best)

    return best


def tune_lightgbm(
    features: pd.DataFrame,
    target: pd.Series,
    feature_cols: list[str],
    n_trials: int = 50,
) -> dict:
    """Tune LightGBM hyperparameters with Optuna.

    Returns:
        Best parameters dictionary.
    """
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "random_state": DEFAULT_RANDOM_SEED,
            "verbose": -1,
        }

        def make_model():
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", LGBMClassifier(**params)),
            ])

        return _evaluate_params(features, target, feature_cols, make_model)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["objective"] = "binary"
    best["metric"] = "binary_logloss"
    best["random_state"] = DEFAULT_RANDOM_SEED
    best["verbose"] = -1

    logger.info("  Best LightGBM log-loss: %.4f", study.best_value)
    logger.info("  Best params: %s", best)

    return best
