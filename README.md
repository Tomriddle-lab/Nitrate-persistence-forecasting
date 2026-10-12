# Nitrate Persistence Forecasting Code

Minimal reference implementation for the Water Research study *When Does Machine Learning Add Value beyond Persistence?* The repository is data-free and accepts prepared causal feature tables. It does not contain plant observations, SCADA cleaning, private target reconstruction, fitted study models, or development search history.

## Fully implemented

- Exact direct nitrate-change labels for 1, 2, 4, 6, 8, and 12 h, with no interpolation.
- Zero-change Persistence and pooled Skill as both a fraction and a percentage.
- Expanding chronological folds with explicit horizon-aware label purging.
- Six fixed learners: Ridge, ElasticNet, ExtraTrees, XGBoost, LightGBM, and CatBoost.
- SI-defined horizon-specific equal-weight member configurations.
- Train-only preprocessing and fixed-member chronological OOF fitting.
- Prediction-only frozen 2025 scoring with no fit, tuning, or reselection.
- Matched-origin historical-resolution Skill and retention checks.
- Retrospective Low, Moderate, and Rapid outcome-stratified Skill summaries when fold-training thresholds are supplied.
- Grouped day-block permutation with Persistence-relative Skill loss.

## Analysis utilities with prepared-input requirements

- SHAP attribution requires a supplied fitted estimator, ordered feature matrix, and optional background matrix. It reports predictive attribution only; SHAP uncertainty and the manuscript's date-cluster bootstrap are not implemented.
- ALE requires a supplied fitted estimator and prepared feature matrix.
- Resolution analysis scores supplied predictions and does not construct the study's resolution-specific feature tables.
- Frozen scoring scores supplied predictions and does not generate, refit, or select frozen models.

## Not implemented in this release

Private target measurement reconstruction, raw wastewater or SCADA processing, confidential sensor mappings, Factory2 transfer, model search, final manuscript figures, the complete hourly-phase experiment, rapid-event episode inventory, SHAP uncertainty analysis, and study-specific frozen model generation are excluded.

## Input contract

Use a table with a unique sorted `timestamp`, measured `bt_effluent_no3` in mg/L, and causal predictors available at or before each origin. Future observations are used only to form an exact evaluation label. Missing targets are not imputed. Feature groups must be supplied explicitly as `TARGET_HISTORY`, `UPSTREAM_NITRATE_STATE`, `NON_NITRATE_PROCESS_STATE`, or `FULL_PRIMARY`.

## Quick check

```text
python -m unittest discover -s tests -v
python examples/synthetic_demo.py
```

Install declared dependencies with `python -m pip install -e .`. The demo creates synthetic data at runtime and saves no data or results.

## Release status

Initial public reference implementation: corrected, tested, and published to `Tomriddle-lab/Nitrate-persistence-forecasting`. This is not a complete study reproduction; the restricted inputs and frozen artifacts described above remain required for full manuscript reproduction.
