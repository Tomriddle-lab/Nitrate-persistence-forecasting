from __future__ import annotations

import numpy as np


def assert_aligned_timestamps(timestamp_sets: list[object]) -> None:
    """Reject unequal, duplicated, or unsorted timestamp vectors."""
    if len(timestamp_sets) < 2:
        raise ValueError("at least two timestamp vectors are required")
    normalized = [np.asarray(values) for values in timestamp_sets]
    if any(values.ndim != 1 for values in normalized):
        raise ValueError("timestamps must be one-dimensional")
    if any(len(values) != len(np.unique(values)) for values in normalized):
        raise ValueError("timestamps must be unique")
    if any(np.any(values[1:] < values[:-1]) for values in normalized):
        raise ValueError("timestamps must be sorted")
    first = normalized[0]
    if any(not np.array_equal(first, values) for values in normalized[1:]):
        raise ValueError("ensemble members must share identical timestamps")


def equal_weight_average(predictions: list[np.ndarray], timestamps: list[object] | None = None) -> np.ndarray:
    """Average predeclared member predictions after enforcing aligned shapes."""
    if len(predictions) < 2:
        raise ValueError("an ensemble requires at least two members")
    arrays = [np.asarray(values, dtype=float) for values in predictions]
    if timestamps is not None:
        if len(timestamps) != len(arrays):
            raise ValueError("one timestamp vector is required per member")
        if any(len(timestamp_values) != len(array) for timestamp_values, array in zip(timestamps, arrays)):
            raise ValueError("each member prediction must match its timestamp vector length")
        assert_aligned_timestamps(timestamps)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("each member prediction must be one-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("member predictions must share identical sample alignment")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("member predictions must be finite")
    return np.mean(np.vstack(arrays), axis=0)
