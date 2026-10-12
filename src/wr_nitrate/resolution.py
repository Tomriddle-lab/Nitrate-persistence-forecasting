from __future__ import annotations

import numpy as np
import pandas as pd

from .ensemble import assert_aligned_timestamps
from .temporal import pooled_skill


HISTORICAL_RESOLUTIONS_MIN = (1, 2, 5, 10, 15, 30, 60)


def validate_resolution_predictions(rows: pd.DataFrame) -> None:
    required = {"resolution_min", "horizon_hours", "timestamp", "actual_delta", "predicted_delta"}
    if not required.issubset(rows.columns):
        raise ValueError(f"resolution rows require {sorted(required)}")
    if rows.empty:
        raise ValueError("resolution rows cannot be empty")
    for horizon, horizon_rows in rows.groupby("horizon_hours", sort=False):
        reference_timestamp = None
        reference_actual = None
        for resolution, group in horizon_rows.groupby("resolution_min", sort=False):
            timestamp = pd.DatetimeIndex(pd.to_datetime(group["timestamp"], errors="raise"))
            actual = pd.to_numeric(group["actual_delta"], errors="coerce").to_numpy(dtype=float)
            predicted = pd.to_numeric(group["predicted_delta"], errors="coerce").to_numpy(dtype=float)
            if timestamp.hasnans or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
                raise ValueError("resolution rows require non-missing, finite timestamps and predictions")
            assert_aligned_timestamps([timestamp.to_numpy(), timestamp.to_numpy()])
            if reference_timestamp is None:
                reference_timestamp = timestamp
                reference_actual = actual
            else:
                if not reference_timestamp.equals(timestamp):
                    raise ValueError(f"resolution {resolution} has unmatched timestamps for horizon {horizon}")
                if not np.array_equal(reference_actual, actual):
                    raise ValueError(f"resolution {resolution} has unmatched observed target changes for horizon {horizon}")


def resolution_metrics(rows: pd.DataFrame, reference_resolution: int = 1) -> pd.DataFrame:
    """Compute pooled Skill and retention from matched OOF resolution rows."""
    validate_resolution_predictions(rows)
    output = []
    for (resolution, horizon), group in rows.groupby(["resolution_min", "horizon_hours"], sort=True):
        result = pooled_skill(group["actual_delta"], group["predicted_delta"])
        output.append({"resolution_min": int(resolution), "horizon_hours": int(horizon), **result})
    metrics = pd.DataFrame(output)
    reference = metrics.loc[metrics["resolution_min"].eq(reference_resolution), ["horizon_hours", "skill_fraction"]].rename(columns={"skill_fraction": "reference_skill_fraction"})
    metrics = metrics.merge(reference, on="horizon_hours", how="left", validate="many_to_one")
    if metrics["reference_skill_fraction"].isna().any() or not np.isfinite(metrics["reference_skill_fraction"]).all():
        raise ValueError("reference Skill is missing or non-finite")
    if (metrics["reference_skill_fraction"] <= 0).any():
        raise ValueError("reference Skill must be finite and positive before retention is calculated")
    metrics["skill_retention"] = metrics["skill_fraction"] / metrics["reference_skill_fraction"]
    metrics["observation_density_percent"] = 100.0 / metrics["resolution_min"]
    return metrics


def minimum_sufficient_resolution(metrics: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    selected = []
    for horizon, group in metrics.groupby("horizon_hours", sort=True):
        eligible = group.loc[group["skill_retention"] >= threshold].sort_values("resolution_min")
        selected.append({
            "horizon_hours": int(horizon),
            "threshold": threshold,
            "minimum_sufficient_resolution_min": int(eligible["resolution_min"].max()) if not eligible.empty else np.nan,
            "status": "PASS" if not eligible.empty else "NO_RESOLUTION_MEETS_THRESHOLD",
        })
    return pd.DataFrame(selected)
