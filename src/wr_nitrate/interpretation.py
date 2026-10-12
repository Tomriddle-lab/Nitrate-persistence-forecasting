from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def linear_shap_values(model, features: pd.DataFrame, background: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return exact additive linear attributions for a fitted linear pipeline."""
    if background is None:
        background = features
    if not list(features.columns) == list(background.columns):
        raise ValueError("features and background must share feature ordering")
    estimator = model[-1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "coef_"):
        raise TypeError("linear_shap_values requires a fitted linear estimator")
    transformed = model[:-1].transform(features) if hasattr(model, "steps") else features.to_numpy(dtype=float)
    baseline = model[:-1].transform(background).mean(axis=0) if hasattr(model, "steps") else background.to_numpy(dtype=float).mean(axis=0)
    coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    if transformed.shape[1] != len(coefficients):
        raise ValueError("feature and coefficient counts do not match")
    return pd.DataFrame((transformed - baseline) * coefficients, columns=features.columns, index=features.index)


def shap_values(model, features: pd.DataFrame, background: pd.DataFrame | None = None, seed: int = 42) -> pd.DataFrame:
    """Use SHAP when installed; use the exact linear fallback for linear models."""
    try:
        import shap
    except ImportError:
        return linear_shap_values(model, features, background)
    np.random.seed(seed)
    explainer = shap.Explainer(model, background if background is not None else features)
    values = explainer(features).values
    if values.ndim == 3:
        values = values[:, :, 0]
    return pd.DataFrame(values, columns=features.columns, index=features.index)


def grouped_day_block_skill_loss(
    model,
    features: pd.DataFrame,
    actual: pd.Series | np.ndarray,
    groups: Mapping[str, list[str]],
    timestamps: pd.Series | np.ndarray,
    repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure documented grouped day-block permutation loss in Persistence Skill.

    Every day must contain the same within-day timestamp offsets. The function
    shuffles complete day blocks for one feature group at a time and reports
    pooled Skill loss, not RMSE increase.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    timestamp = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="raise"))
    if len(timestamp) != len(features) or timestamp.hasnans or timestamp.has_duplicates or not timestamp.is_monotonic_increasing:
        raise ValueError("features and timestamps must be aligned, sorted, and unique")
    actual_array = np.asarray(actual, dtype=float)
    if len(actual_array) != len(features) or not np.isfinite(actual_array).all():
        raise ValueError("actual values must be aligned and finite")
    baseline = _rmse(actual_array, np.asarray(model.predict(features), dtype=float))
    persistence_rmse = _rmse(actual_array, np.zeros_like(actual_array))
    if persistence_rmse <= 0 or not np.isfinite(persistence_rmse):
        raise ValueError("Persistence RMSE must be positive and finite")
    baseline_skill = 1.0 - baseline / persistence_rmse
    day = timestamp.normalize()
    day_groups = [np.flatnonzero(day == value) for value in pd.unique(day)]
    offsets = [tuple(timestamp[index].time) for index in day_groups]
    if any(len(index) != len(day_groups[0]) for index in day_groups) or any(offset != offsets[0] for offset in offsets[1:]):
        raise ValueError("all day blocks must share identical within-day timestamps")
    rng = np.random.default_rng(seed)
    rows = []
    for group_name, columns in groups.items():
        missing = [column for column in columns if column not in features]
        if missing:
            raise ValueError(f"missing feature columns for group {group_name}: {missing[:3]}")
        losses = []
        for _ in range(repeats):
            shuffled = features.copy()
            source_days = rng.permutation(len(day_groups))
            for target_day, source_day in enumerate(source_days):
                shuffled.loc[shuffled.index[day_groups[target_day]], columns] = features.iloc[day_groups[source_day]][columns].to_numpy()
            permuted_skill = 1.0 - _rmse(actual_array, np.asarray(model.predict(shuffled), dtype=float)) / persistence_rmse
            losses.append(baseline_skill - permuted_skill)
        rows.append({"feature_group": group_name, "baseline_skill_fraction": baseline_skill, "mean_skill_loss_fraction": float(np.mean(losses)), "sd_skill_loss_fraction": float(np.std(losses, ddof=1)) if repeats > 1 else 0.0, "baseline_skill_percent": 100.0 * baseline_skill, "mean_skill_loss_percent": 100.0 * float(np.mean(losses)), "repeats": repeats})
    return pd.DataFrame(rows)


def grouped_row_permutation_rmse_increase(
    model,
    features: pd.DataFrame,
    actual: pd.Series | np.ndarray,
    groups: Mapping[str, list[str]],
    repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Simplified exploratory row permutation; not the manuscript analysis."""
    actual_array = np.asarray(actual, dtype=float)
    baseline = _rmse(actual_array, np.asarray(model.predict(features), dtype=float))
    rng = np.random.default_rng(seed)
    rows = []
    for group_name, columns in groups.items():
        missing = [column for column in columns if column not in features]
        if missing:
            raise ValueError(f"missing feature columns for group {group_name}: {missing[:3]}")
        increases = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled.loc[:, columns] = shuffled.loc[:, columns].iloc[rng.permutation(len(shuffled))].to_numpy()
            increases.append(_rmse(actual_array, np.asarray(model.predict(shuffled))) - baseline)
        rows.append({"feature_group": group_name, "baseline_rmse": baseline, "mean_rmse_increase": float(np.mean(increases)), "sd_rmse_increase": float(np.std(increases, ddof=1)) if repeats > 1 else 0.0, "repeats": repeats, "scope": "simplified_exploratory_not_manuscript_reproduction"})
    return pd.DataFrame(rows)


def grouped_permutation_importance(*args, **kwargs) -> pd.DataFrame:
    """Compatibility wrapper for the simplified exploratory row permutation."""
    return grouped_row_permutation_rmse_increase(*args, **kwargs)


def ale_1d(model, features: pd.DataFrame, feature: str, bins: int = 10) -> pd.DataFrame:
    """Compute a simple one-dimensional accumulated local effect curve."""
    if feature not in features or bins < 2:
        raise ValueError("feature must exist and bins must be at least two")
    values = pd.to_numeric(features[feature], errors="coerce")
    edges = np.unique(values.dropna().quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        raise ValueError("feature does not have enough unique values for ALE")
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = values.ge(left) & values.le(right) if right == edges[-1] else values.ge(left) & values.lt(right)
        if not mask.any():
            continue
        lower = features.loc[mask].copy()
        upper = lower.copy()
        lower[feature] = left
        upper[feature] = right
        local_effect = np.asarray(model.predict(upper)) - np.asarray(model.predict(lower))
        rows.append({"feature": feature, "bin_left": left, "bin_right": right, "n": int(mask.sum()), "local_effect": float(np.mean(local_effect))})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["ale"] = result["local_effect"].cumsum()
    result["ale"] -= result["ale"].mean()
    return result
