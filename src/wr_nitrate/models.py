from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ridge_pipeline(alpha: float = 10.0) -> Pipeline:
    """Build the documented impute-then-scale Ridge pipeline.

    Fit the returned pipeline separately inside each training fold.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
