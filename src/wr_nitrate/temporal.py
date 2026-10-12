from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _validated_series(values: pd.Series) -> pd.Series:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("values must use a DatetimeIndex")
    if values.index.has_duplicates:
        raise ValueError("timestamps must be unique")
    if not values.index.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        raise ValueError("target contains nonnumeric observations")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("target contains infinite observations")
    return numeric


def exact_change_targets(values: pd.Series, horizon: str | pd.Timedelta) -> pd.DataFrame:
    """Return only origins with an observed exact t+h target; never interpolate."""
    target = _validated_series(values)
    step = pd.Timedelta(horizon)
    if step <= pd.Timedelta(0):
        raise ValueError("horizon must be positive")
    future = target.reindex(target.index + step)
    future.index = target.index
    valid = target.notna() & future.notna()
    return pd.DataFrame(
        {
            "timestamp": target.index[valid],
            "target_time": target.index[valid] + step,
            "current_value": target.loc[valid].to_numpy(dtype=float),
            "future_value": future.loc[valid].to_numpy(dtype=float),
            "delta": (future.loc[valid] - target.loc[valid]).to_numpy(dtype=float),
        }
    ).reset_index(drop=True)


def expanding_month_folds(
    timestamps: Iterable[pd.Timestamp],
    development_start: str | pd.Timestamp,
    validation_months: Iterable[str],
    horizon: str | pd.Timedelta,
) -> list[dict[str, pd.DatetimeIndex]]:
    """Create expanding monthly folds with exact horizon-aware boundary purging."""
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("fold timestamps must be sorted and unique")
    step = pd.Timedelta(horizon)
    start = pd.Timestamp(development_start)
    folds = []
    for month in validation_months:
        val_start = pd.Period(month, freq="M").start_time
        val_end = (pd.Period(month, freq="M") + 1).start_time - pd.Timedelta(nanoseconds=1)
        train_end = val_start - pd.Timedelta(nanoseconds=1)
        train_mask = (index >= start) & (index + step <= train_end)
        valid_mask = (index >= val_start) & (index <= val_end) & (index + step <= val_end)
        folds.append(
            {
                "train": index[train_mask],
                "validation": index[valid_mask],
                "train_end": train_end,
                "validation_start": val_start,
                "validation_end": val_end,
            }
        )
    return folds


def persistence_skill(actual_delta: Iterable[float], predicted_delta: Iterable[float]) -> dict[str, float]:
    """Score delta predictions against Persistence (zero predicted change)."""
    actual = np.asarray(list(actual_delta), dtype=float)
    predicted = np.asarray(list(predicted_delta), dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must have the same nonzero length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite paired observations")
    model_rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))
    persistence_rmse = float(np.sqrt(np.mean(np.square(actual))))
    skill = float("nan") if persistence_rmse == 0 else 1.0 - model_rmse / persistence_rmse
    skill_fraction = skill
    return {
        "n": int(actual.size),
        "model_rmse": model_rmse,
        "persistence_rmse": persistence_rmse,
        "skill_fraction": skill_fraction,
        "skill_percent": 100.0 * skill_fraction,
    }


def pooled_skill(actual_delta: Iterable[float], predicted_delta: Iterable[float]) -> dict[str, float]:
    """Alias with explicit pooled terminology for row-wise OOF scoring."""
    result = persistence_skill(actual_delta, predicted_delta)
    result["scope"] = "pooled"
    return result
