# Release scope

Status: `INITIAL_PUBLIC_REFERENCE_IMPLEMENTATION`

## Method boundary

This release contains compact public-method implementations and synthetic tests. It is not a complete manuscript reproduction because study-level inputs, feature construction, private mappings, frozen estimators, and derived results are excluded.

Implemented code includes exact labels, Persistence scoring, fixed learner builders, the SI member map, chronological OOF fitting, frozen prediction scoring, matched-resolution scoring, grouped day-block Skill loss, generic SHAP/ALE utilities, and fold-threshold retrospective strata.

Several methods are scoring or interpretation utilities only. They require externally prepared feature matrices, explicit feature groups, predictions, timestamps, training-derived thresholds, or fitted estimators. They do not create those restricted inputs.

## Manuscript and SI differences

The public SI specification lists Ridge alpha 10.0 and the fixed learner settings used here. Some undistributed frozen estimators may contain horizon/group-specific settings not represented by the public fixed specifications. Those estimators are not released; reproducing frozen study outputs requires an authorized estimator release or a separately documented artifact package.

The manuscript reports a complete hourly-phase experiment, SHAP date-cluster uncertainty analysis, and a frozen-2025 model-generation workflow. This release does not claim to reproduce those analyses: it provides generic resolution scoring, generic attribution utilities, and prediction-only frozen scoring.

## Exclusions and audit

No original research file was modified. No Git repository was initialized. No credentials, binary caches, private paths, raw or derived wastewater data, confidential source maps, Factory2 code, internal search history, or study-specific model artifacts are included.

This is a corrected initial public reference implementation, not a complete study reproduction.
