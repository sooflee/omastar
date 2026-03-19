"""SHAP-based feature importance analysis."""
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from pathlib import Path


def compute_shap_importance(
    model,
    features: pd.DataFrame,
    feature_cols: list[str],
    save_dir: Path | None = None,
) -> list[dict]:
    """Compute SHAP feature importance for a trained model.

    Extracts the tree-based model from a Pipeline if needed and uses
    TreeExplainer for XGBoost/LightGBM, or KernelExplainer as fallback.

    Args:
        model: Trained model (Pipeline, EnsembleModel, StackedEnsemble, or CalibratedModel).
        features: Feature DataFrame with Season and feature columns.
        feature_cols: Feature column names.
        save_dir: Directory to save plots. If None, plots are not saved.

    Returns:
        List of dicts with 'feature' and 'importance' keys, sorted descending.
    """
    X = features[feature_cols].fillna(0)
    display_names = [c.replace("_diff", "") for c in feature_cols]

    # Extract the underlying model and transform data
    tree_model, X_transformed = _extract_tree_model(model, X)

    if tree_model is not None:
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # Fallback: use KernelExplainer with a sample
        sample_size = min(100, len(X))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]

        def predict_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, X_sample)
        shap_values = explainer.shap_values(X, nsamples=100)

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = [
        {"feature": name, "importance": round(float(val), 4)}
        for name, val in zip(display_names, mean_abs_shap)
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)

    # Generate plots
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.3)))
        shap.summary_plot(
            shap_values, X_transformed,
            feature_names=display_names,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(save_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.3)))
        shap.summary_plot(
            shap_values, X_transformed,
            feature_names=display_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(save_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

    return importance


def _extract_tree_model(model, X):
    """Extract a tree-based model from a Pipeline/Ensemble for SHAP TreeExplainer.

    Returns (tree_model, X_transformed) or (None, X) if no tree model found.
    """
    from src.model.train import CalibratedModel, EnsembleModel, StackedEnsemble

    # Unwrap CalibratedModel
    if isinstance(model, CalibratedModel):
        model = model.base_model

    # For EnsembleModel, use the XGBoost component
    if isinstance(model, EnsembleModel):
        pipeline = model.xgb
        scaler = pipeline.named_steps["scaler"]
        X_transformed = scaler.transform(X)
        return pipeline.named_steps["model"], X_transformed

    # For StackedEnsemble, use the XGBoost base model
    if isinstance(model, StackedEnsemble):
        for name, m in model.fitted_models:
            if name == "xgb":
                scaler = m.named_steps["scaler"]
                X_transformed = scaler.transform(X)
                return m.named_steps["model"], X_transformed
        # Fallback to first model
        if model.fitted_models:
            name, m = model.fitted_models[0]
            if isinstance(m, Pipeline) and "scaler" in m.named_steps:
                scaler = m.named_steps["scaler"]
                X_transformed = scaler.transform(X)
                return m.named_steps["model"], X_transformed

    # For plain Pipeline
    if isinstance(model, Pipeline):
        underlying = model.named_steps.get("model")
        if underlying is not None:
            scaler = model.named_steps.get("scaler")
            if scaler is not None:
                X_transformed = scaler.transform(X)
            else:
                X_transformed = X
            return underlying, X_transformed

    return None, X
