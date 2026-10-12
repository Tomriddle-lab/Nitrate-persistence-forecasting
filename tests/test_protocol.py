import unittest

import numpy as np
import pandas as pd

from wr_nitrate.change_strata import retrospective_stratified_skill
from wr_nitrate.ensemble import assert_aligned_timestamps, equal_weight_average
from wr_nitrate.evaluation import _validate_oof_folds, evaluate_frozen, evaluate_frozen_2025, oof_predict, score_predictions
from wr_nitrate.learners import MemberSpec, ensemble_members
from wr_nitrate.resolution import minimum_sufficient_resolution, resolution_metrics
from wr_nitrate.temporal import exact_change_targets, expanding_month_folds, pooled_skill


class ProtocolTests(unittest.TestCase):
    def test_all_six_horizons_use_exact_future_timestamps(self):
        index = pd.date_range("2024-09-01", periods=20, freq="h")
        values = pd.Series(np.arange(20, dtype=float), index=index)
        for hours in (1, 2, 4, 6, 8, 12):
            labels = exact_change_targets(values, f"{hours}h")
            self.assertTrue((labels["target_time"] == labels["timestamp"] + pd.Timedelta(hours=hours)).all())
            self.assertTrue((labels["delta"] == float(hours)).all())

    def test_unsorted_and_duplicate_timestamps_are_rejected(self):
        index = pd.to_datetime(["2024-09-01 01:00", "2024-09-01 00:00"])
        with self.assertRaises(ValueError):
            exact_change_targets(pd.Series([1.0, 2.0], index=index), "1h")
        duplicate = pd.to_datetime(["2024-09-01", "2024-09-01"])
        with self.assertRaises(ValueError):
            exact_change_targets(pd.Series([1.0, 2.0], index=duplicate), "1h")

    def test_nonnumeric_and_infinite_targets_are_rejected(self):
        index = pd.date_range("2024-09-01", periods=3, freq="h")
        with self.assertRaises(ValueError):
            exact_change_targets(pd.Series([1.0, "bad", 3.0], index=index), "1h")
        with self.assertRaises(ValueError):
            exact_change_targets(pd.Series([1.0, np.inf, 3.0], index=index), "1h")

    def test_missing_future_observations_are_not_filled(self):
        index = pd.date_range("2024-09-01", periods=4, freq="h")
        values = pd.Series([1.0, 2.0, np.nan, 4.0], index=index)
        labels = exact_change_targets(values, "2h")
        self.assertEqual(labels["timestamp"].tolist(), [index[1]])
        self.assertEqual(labels["delta"].tolist(), [2.0])

    def test_folds_purge_training_and_validation_targets_at_boundaries(self):
        index = pd.date_range("2024-08-30 00:00", "2024-09-30 23:00", freq="h")
        fold = expanding_month_folds(index, "2024-04-01", ["2024-09"], "2h")[0]
        self.assertLessEqual(fold["train"].max() + pd.Timedelta("2h"), pd.Timestamp("2024-08-31 23:59:59.999999999"))
        self.assertTrue((fold["validation"] + pd.Timedelta("2h") <= pd.Timestamp("2024-09-30 23:59:59.999999999")).all())

    def test_ensemble_timestamp_alignment_is_required(self):
        good = np.array(["2024-09-01", "2024-09-02"], dtype="datetime64[D]")
        bad = np.array(["2024-09-01", "2024-09-03"], dtype="datetime64[D]")
        assert_aligned_timestamps([good, good])
        with self.assertRaises(ValueError):
            assert_aligned_timestamps([good, bad])
        with self.assertRaises(ValueError):
            equal_weight_average([np.array([1.0]), np.array([2.0])], [good, bad])
        with self.assertRaises(ValueError):
            equal_weight_average([np.array([1.0]), np.array([2.0, 3.0])], [good, good])

    def test_oof_predict_returns_timestamp_column_without_index_lookup_error(self):
        timestamps = pd.date_range("2024-09-01", periods=72, freq="h")
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "target": np.sin(np.arange(72, dtype=float) / 5.0),
            "feature": np.cos(np.arange(72, dtype=float) / 7.0),
        })
        fold = {
            "train": timestamps[:47],
            "validation": timestamps[48:71],
            "train_end": timestamps[47],
            "validation_start": timestamps[48],
            "validation_end": timestamps[71],
        }
        members = (
            MemberSpec("TARGET_HISTORY", "Ridge", {"alpha": 1.0}),
            MemberSpec("TARGET_HISTORY", "ElasticNet", {"alpha": 0.001, "l1_ratio": 0.2, "max_iter": 3000, "random_state": 42}),
        )
        output, metrics = oof_predict(
            frame,
            "timestamp",
            "target",
            {"TARGET_HISTORY": ["feature"]},
            1,
            [fold],
            members,
        )
        self.assertEqual(output["timestamp"].tolist(), list(timestamps[48:71]))
        self.assertEqual(len(output), 23)
        self.assertEqual(metrics["ensemble"]["n"], 23)
        self.assertTrue(np.isfinite(output["ensemble_pred_delta"]).all())

    def test_score_rejects_unequal_or_nonfinite_evaluation_vectors(self):
        timestamps = pd.date_range("2024-09-01", periods=2, freq="h")
        with self.assertRaises(ValueError):
            score_predictions(timestamps, [1.0], [1.0])
        with self.assertRaises(ValueError):
            score_predictions([timestamps[0], pd.NaT], [1.0, 1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            score_predictions(timestamps, [1.0, np.nan], [1.0, 1.0])

    def test_pooled_skill_and_fraction_percent_are_explicit(self):
        result = pooled_skill([1.0, -1.0], [0.5, -0.5])
        self.assertAlmostEqual(result["persistence_rmse"], 1.0)
        self.assertAlmostEqual(result["skill_fraction"], 0.5)
        self.assertAlmostEqual(result["skill_percent"], 50.0)

    def test_final_member_map_covers_all_horizons_without_search(self):
        self.assertEqual([len(ensemble_members(hours)) for hours in (1, 2, 4, 6, 8, 12)], [4, 4, 4, 4, 4, 3])

    def test_frozen_evaluation_scores_rows_without_fit_or_reselection(self):
        rows = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "actual_delta": [1.0, -1.0, 0.0],
            "predicted_delta": [0.5, -0.5, 0.0],
        })
        result = evaluate_frozen(rows, "timestamp", "actual_delta", "predicted_delta")
        self.assertEqual(result["n"], 3)
        self.assertGreater(result["skill_fraction"], 0.0)
        self.assertEqual(evaluate_frozen_2025(rows, "timestamp", "actual_delta", "predicted_delta")["n"], 3)
        rows["timestamp"] = pd.date_range("2024-01-01", periods=3, freq="h")
        with self.assertRaises(ValueError):
            evaluate_frozen_2025(rows, "timestamp", "actual_delta", "predicted_delta")

    def test_resolution_retention_and_threshold_are_pooled(self):
        timestamps = pd.date_range("2024-09-01", periods=4, freq="h")
        rows = pd.concat([
            pd.DataFrame({"resolution_min": 1, "horizon_hours": 1, "timestamp": timestamps, "actual_delta": [1.0, -1.0, 1.0, -1.0], "predicted_delta": [0.5, -0.5, 0.5, -0.5]}),
            pd.DataFrame({"resolution_min": 60, "horizon_hours": 1, "timestamp": timestamps, "actual_delta": [1.0, -1.0, 1.0, -1.0], "predicted_delta": [1.0, -1.0, 1.0, -1.0]}),
        ], ignore_index=True)
        metrics = resolution_metrics(rows)
        self.assertAlmostEqual(float(metrics.loc[metrics["resolution_min"].eq(1), "skill_retention"].iloc[0]), 1.0)
        selected = minimum_sufficient_resolution(metrics, 0.95)
        self.assertEqual(int(selected["minimum_sufficient_resolution_min"].iloc[0]), 60)

    def test_resolution_rejects_mismatched_origins_actuals_and_nonpositive_reference(self):
        timestamps = pd.date_range("2024-09-01", periods=4, freq="h")
        base = {"horizon_hours": 1, "actual_delta": [1.0, -1.0, 1.0, -1.0]}
        rows = pd.concat([
            pd.DataFrame({"resolution_min": 1, "timestamp": timestamps, "predicted_delta": [0.5, -0.5, 0.5, -0.5], **base}),
            pd.DataFrame({"resolution_min": 60, "timestamp": timestamps + pd.Timedelta("1h"), "predicted_delta": [0.5, -0.5, 0.5, -0.5], **base}),
        ], ignore_index=True)
        with self.assertRaises(ValueError):
            resolution_metrics(rows)
        rows.loc[rows["resolution_min"].eq(60), "timestamp"] = timestamps
        rows.loc[rows["resolution_min"].eq(60), "actual_delta"] = [2.0, -2.0, 2.0, -2.0]
        with self.assertRaises(ValueError):
            resolution_metrics(rows)

    def test_oof_rejects_overlapping_and_boundary_crossing_folds(self):
        labels = exact_change_targets(pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2024-09-01", periods=4, freq="h")), "1h").set_index("timestamp")
        fold = {"train": labels.index[:1], "validation": labels.index[:1], "train_end": labels.index[0], "validation_start": labels.index[1], "validation_end": labels.index[2]}
        with self.assertRaises(ValueError):
            _validate_oof_folds(labels, [fold], 1)
        fold = {"train": labels.index[:1], "validation": labels.index[1:], "train_end": labels.index[0], "validation_start": labels.index[1], "validation_end": labels.index[1]}
        with self.assertRaises(ValueError):
            _validate_oof_folds(labels, [fold], 1)

    def test_change_strata_uses_supplied_training_thresholds(self):
        rows = pd.DataFrame({"fold": [1, 1, 1], "actual_delta": [0.1, 0.5, 1.0], "predicted_delta": [0.0, 0.4, 0.8]})
        result = retrospective_stratified_skill(rows, {1: 0.2}, {1: 0.7})
        self.assertEqual(set(result["change_stratum"]), {"Low", "Moderate", "Rapid"})
        with self.assertRaises(ValueError):
            retrospective_stratified_skill(pd.DataFrame({"fold": [1], "actual_delta": [np.nan], "predicted_delta": [0.0]}), {1: 0.2}, {1: 0.7})
        with self.assertRaises(ValueError):
            retrospective_stratified_skill(pd.DataFrame({"fold": [1], "actual_delta": [1.0], "predicted_delta": [0.0]}), {1: np.nan}, {1: 0.7})


if __name__ == "__main__":
    unittest.main()
