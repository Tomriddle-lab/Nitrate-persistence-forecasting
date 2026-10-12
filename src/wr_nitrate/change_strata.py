from __future__ import annotations

import numpy as np
import pandas as pd

from .temporal import pooled_skill


def label_change_strata(actual_delta: pd.Series | np.ndarray, q50: float, q90: float) -> pd.Series:
    """Apply training-only thresholds to retrospective realized-change labels."""
    try:
        thresholds = np.asarray([q50, q90], dtype=float)
        magnitude = np.abs(np.asarray(actual_delta, dtype=float))
    except (TypeError, ValueError) as error:
        raise ValueError("thresholds and realized changes must be numeric") from error
    if not np.isfinite(thresholds).all() or thresholds[0] < 0 or thresholds[1] < thresholds[0]:
        raise ValueError("thresholds must be finite, nonnegative, and q90 must be >= q50")
    if not np.isfinite(magnitude).all():
        raise ValueError("realized changes must be finite and cannot be missing")
    return pd.Series(np.select([magnitude <= thresholds[0], magnitude < thresholds[1]], ["Low", "Moderate"], default="Rapid"))


def retrospective_stratified_skill(rows: pd.DataFrame, q50_by_fold: dict[object, float], q90_by_fold: dict[object, float]) -> pd.DataFrame:
    required = {"fold", "actual_delta", "predicted_delta"}
    if not required.issubset(rows.columns):
        raise ValueError(f"rows require {sorted(required)}")
    pieces = []
    for fold, group in rows.groupby("fold", sort=True):
        if fold not in q50_by_fold or fold not in q90_by_fold:
            raise ValueError(f"missing training thresholds for fold {fold}")
        labeled = group.copy()
        labeled["change_stratum"] = label_change_strata(labeled["actual_delta"], q50_by_fold[fold], q90_by_fold[fold]).to_numpy()
        pieces.append(labeled)
    labeled_rows = pd.concat(pieces, ignore_index=True)
    output = []
    for stratum, group in labeled_rows.groupby("change_stratum", sort=True):
        result = pooled_skill(group["actual_delta"], group["predicted_delta"])
        output.append({"change_stratum": stratum, **result})
    return pd.DataFrame(output)
