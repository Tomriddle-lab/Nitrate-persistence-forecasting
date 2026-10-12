from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemberSpec:
    feature_group: str
    learner: str
    parameters: dict[str, Any]


LEARNER_PARAMETERS: dict[str, dict[str, Any]] = {
    "Ridge": {"alpha": 10.0},
    "ElasticNet": {"alpha": 0.001, "l1_ratio": 0.2, "max_iter": 3000, "random_state": 42},
    "ExtraTrees": {
        "n_estimators": 150, "max_depth": 14, "max_features": 0.7,
        "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42,
    },
    "XGBoost": {
        "n_estimators": 350, "learning_rate": 0.04, "max_depth": 6,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.05, "reg_lambda": 1.0, "objective": "reg:squarederror",
        "tree_method": "hist", "n_jobs": -1, "random_state": 42,
    },
    "LightGBM": {
        "n_estimators": 350, "learning_rate": 0.04, "num_leaves": 31,
        "min_child_samples": 30, "max_depth": -1, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 1.0,
        "objective": "regression", "n_jobs": -1, "random_state": 42, "verbosity": -1,
    },
    "CatBoost": {
        "iterations": 350, "depth": 6, "learning_rate": 0.04,
        "loss_function": "RMSE", "random_seed": 42, "thread_count": -1,
        "verbose": False, "allow_writing_files": False,
    },
}


FINAL_ENSEMBLE_MEMBERS: dict[int, tuple[MemberSpec, ...]] = {
    1: (
        MemberSpec("TARGET_HISTORY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("UPSTREAM_NITRATE_STATE", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("FULL_PRIMARY", "ElasticNet", LEARNER_PARAMETERS["ElasticNet"]),
        MemberSpec("FULL_PRIMARY", "CatBoost", LEARNER_PARAMETERS["CatBoost"]),
    ),
    2: (
        MemberSpec("TARGET_HISTORY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("TARGET_HISTORY", "ElasticNet", LEARNER_PARAMETERS["ElasticNet"]),
        MemberSpec("UPSTREAM_NITRATE_STATE", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("FULL_PRIMARY", "CatBoost", LEARNER_PARAMETERS["CatBoost"]),
    ),
    4: (
        MemberSpec("TARGET_HISTORY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("TARGET_HISTORY", "ElasticNet", LEARNER_PARAMETERS["ElasticNet"]),
        MemberSpec("UPSTREAM_NITRATE_STATE", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("FULL_PRIMARY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
    ),
    6: (
        MemberSpec("TARGET_HISTORY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("TARGET_HISTORY", "ElasticNet", LEARNER_PARAMETERS["ElasticNet"]),
        MemberSpec("UPSTREAM_NITRATE_STATE", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("FULL_PRIMARY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
    ),
    8: (
        MemberSpec("TARGET_HISTORY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("TARGET_HISTORY", "ElasticNet", LEARNER_PARAMETERS["ElasticNet"]),
        MemberSpec("UPSTREAM_NITRATE_STATE", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("FULL_PRIMARY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
    ),
    12: (
        MemberSpec("TARGET_HISTORY", "Ridge", LEARNER_PARAMETERS["Ridge"]),
        MemberSpec("TARGET_HISTORY", "ElasticNet", LEARNER_PARAMETERS["ElasticNet"]),
        MemberSpec("NON_NITRATE_PROCESS_STATE", "XGBoost", LEARNER_PARAMETERS["XGBoost"]),
    ),
}


def ensemble_members(horizon_hours: int) -> tuple[MemberSpec, ...]:
    """Return the predeclared equal-weight members for one horizon."""
    try:
        return FINAL_ENSEMBLE_MEMBERS[int(horizon_hours)]
    except KeyError as error:
        raise ValueError("horizon must be one of 1, 2, 4, 6, 8, or 12 hours") from error


def make_learner(name: str, parameters: dict[str, Any] | None = None):
    """Build one fixed learner; this function contains no search or tuning."""
    params = dict(LEARNER_PARAMETERS[name] if parameters is None else parameters)
    if name == "Ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(**params))])
    if name == "ElasticNet":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", ElasticNet(**params))])
    if name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(**params))])
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    raise ValueError(f"unsupported learner: {name}")
