from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .ensemble import assert_aligned_timestamps, equal_weight_average
from .learners import MemberSpec, make_learner
from .temporal import exact_change_targets, pooled_skill


def _validate_oof_folds(labels: pd.DataFrame, folds: list[dict[str, object]], horizon_hours: int) -> None:
    if not folds:
        raise ValueError("at least one temporal fold is required")
    step = pd.Timedelta(hours=horizon_hours)
    previous_validation_end = None
    validation_seen = set()
    for fold_number, fold in enumerate(folds, start=1):
        required = {"train", "validation", "train_end", "validation_start", "validation_end"}
        if not isinstance(fold, Mapping) or not required.issubset(fold):
            raise ValueError(f"fold {fold_number} lacks explicit temporal boundary metadata")
        train_times = pd.DatetimeIndex(pd.to_datetime(fold["train"], errors="raise"))
        valid_times = pd.DatetimeIndex(pd.to_datetime(fold["validation"], errors="raise"))
        train_end = pd.Timestamp(fold["train_end"])
        valid_start = pd.Timestamp(fold["validation_start"])
        valid_end = pd.Timestamp(fold["validation_end"])
        if train_times.empty or valid_times.empty:
            raise ValueError(f"fold {fold_number} has an empty train or validation partition")
        for name, times in (("train", train_times), ("validation", valid_times)):
            if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
                raise ValueError(f"fold {fold_number} {name} timestamps must be sorted and unique")
        if not train_end < valid_start <= valid_end:
            raise ValueError(f"fold {fold_number} has invalid chronological boundaries")
        if set(train_times).intersection(set(valid_times)):
            raise ValueError(f"fold {fold_number} has overlapping train and validation timestamps")
        if train_times.max() >= valid_start or (train_times + step > train_end).any():
            raise ValueError(f"fold {fold_number} training labels cross the training boundary")
        if valid_times.min() < valid_start or (valid_times + step > valid_end).any():
            raise ValueError(f"fold {fold_number} validation labels cross the validation boundary")
        if not set(train_times).issubset(set(labels.index)) or not set(valid_times).issubset(set(labels.index)):
            raise ValueError(f"fold {fold_number} contains origins without exact finite labels")
        if previous_validation_end is not None and valid_start <= previous_validation_end:
            raise ValueError("validation folds overlap or are not chronological")
        if validation_seen.intersection(set(valid_times)):
            raise ValueError("validation timestamps occur in more than one fold")
        validation_seen.update(valid_times)
        previous_validation_end = valid_end


def _check_frame(frame: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.DataFrame:
    if timestamp_col not in frame or target_col not in frame:
        raise ValueError("input must contain timestamp and target columns")
    result = frame.copy()
    timestamp = pd.DatetimeIndex(pd.to_datetime(result[timestamp_col]))
    if timestamp.has_duplicates or not timestamp.is_monotonic_increasing:
        raise ValueError("input timestamps must be sorted and unique")
    target = pd.to_numeric(result[target_col], errors="coerce")
    if (result[target_col].notna() & target.isna()).any():
        raise ValueError("target contains nonnumeric observations")
    if np.isinf(target.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("target contains infinite observations")
    result[timestamp_col] = timestamp
    result[target_col] = target
    return result


def score_predictions(
    timestamps: object,
    actual_delta: object,
    predicted_delta: object,
) -> dict[str, float]:
    """Score already paired predictions; Skill is fractional and percentage."""
    timestamp = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="raise"))
    actual = np.asarray(list(actual_delta), dtype=float)
    predicted = np.asarray(list(predicted_delta), dtype=float)
    if len(timestamp) != len(actual) or len(timestamp) != len(predicted):
        raise ValueError("timestamps, actual values, and predictions must have identical lengths")
    if timestamp.empty or timestamp.hasnans or timestamp.has_duplicates or not timestamp.is_monotonic_increasing:
        raise ValueError("prediction timestamps must be sorted and unique")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("evaluation values must be finite")
    return pooled_skill(actual, predicted)


def oof_predict(
    frame: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    feature_groups: Mapping[str, list[str]],
    horizon_hours: int,
    folds: list[dict[str, object]],
    members: tuple[MemberSpec, ...],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit fixed members in chronological folds and return pooled OOF rows."""
    data = _check_frame(frame, timestamp_col, target_col)
    indexed = data.set_index(timestamp_col)
    labels = exact_change_targets(indexed[target_col], f"{horizon_hours}h").set_index("timestamp")
    _validate_oof_folds(labels, folds, horizon_hours)
    records: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(folds, start=1):
        train_times = pd.DatetimeIndex(fold["train"])
        valid_times = pd.DatetimeIndex(fold["validation"])
        train = labels.loc[labels.index.intersection(train_times)]
        valid = labels.loc[labels.index.intersection(valid_times)]
        if train.empty or valid.empty:
            raise ValueError(f"fold {fold_number} has no exact finite labels")
        train.index.name = "timestamp"
        valid.index.name = "timestamp"
        base = pd.DataFrame({
            "timestamp": valid.index,
            "actual_delta": valid["delta"].to_numpy(dtype=float),
        })
        member_predictions = []
        member_timestamps = []
        for member_number, member in enumerate(members, start=1):
            columns = feature_groups[member.feature_group]
            missing = [column for column in columns if column not in data]
            if missing:
                raise ValueError(f"missing prepared feature columns: {missing[:3]}")
            model = make_learner(member.learner, member.parameters)
            model.fit(indexed.loc[train.index, columns], train["delta"])
            predicted = np.asarray(model.predict(indexed.loc[valid.index, columns]), dtype=float)
            member_predictions.append(predicted)
            member_timestamps.append(valid.index.to_numpy())
            base[f"member_{member_number}"] = predicted
        assert_aligned_timestamps(member_timestamps)
        base["ensemble_pred_delta"] = equal_weight_average(member_predictions, member_timestamps)
        base["fold"] = fold_number
        records.append(base)
    if not records:
        raise ValueError("no complete OOF fold was available")
    output = pd.concat(records, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    metrics = {"ensemble": score_predictions(output["timestamp"], output["actual_delta"], output["ensemble_pred_delta"])}
    return output, metrics


def evaluate_frozen(
    rows: pd.DataFrame,
    timestamp_col: str,
    actual_delta_col: str,
    predicted_delta_col: str,
) -> dict[str, float]:
    """Score frozen predictions only; no fit, selection, or model access occurs."""
    if timestamp_col not in rows or actual_delta_col not in rows or predicted_delta_col not in rows:
        raise ValueError("frozen rows require timestamp, actual delta, and predicted delta")
    return score_predictions(rows[timestamp_col], rows[actual_delta_col], rows[predicted_delta_col])


def evaluate_frozen_2025(
    rows: pd.DataFrame,
    timestamp_col: str,
    actual_delta_col: str,
    predicted_delta_col: str,
    start: str = "2025-01-01 00:00:00",
    end: str = "2025-04-30 23:59:59.999999999",
) -> dict[str, float]:
    """Score a locked 2025 table without fitting, selection, or tuning."""
    if timestamp_col not in rows:
        raise ValueError("frozen rows require a timestamp column")
    timestamp = pd.DatetimeIndex(pd.to_datetime(rows[timestamp_col]))
    if timestamp.has_duplicates or not timestamp.is_monotonic_increasing:
        raise ValueError("frozen timestamps must be sorted and unique")
    if timestamp.empty or timestamp.min() < pd.Timestamp(start) or timestamp.max() > pd.Timestamp(end):
        raise ValueError("evaluate_frozen_2025 accepts only the declared 2025 evaluation window")
    return evaluate_frozen(rows, timestamp_col, actual_delta_col, predicted_delta_col)


def frozen_ensemble_predictions(
    member_rows: list[pd.DataFrame],
    timestamp_col: str,
    prediction_col: str,
) -> pd.DataFrame:
    """Fuse already-generated frozen member predictions without retraining."""
    if len(member_rows) < 2:
        raise ValueError("at least two frozen member tables are required")
    timestamps = [row[timestamp_col].to_numpy() for row in member_rows]
    assert_aligned_timestamps(timestamps)
    predictions = [row[prediction_col].to_numpy(dtype=float) for row in member_rows]
    result = pd.DataFrame({timestamp_col: timestamps[0], "ensemble_pred_delta": equal_weight_average(predictions, timestamps)})
    return result
