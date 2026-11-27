"""Small independent examples for the analytical failure modes that matter."""

import unittest

import pandas as pd

from analyze_cctv import attach, IMAGE_KEY
from scraped_insights import add_features, coverage_tables, matched_comparison, summarize, validate_counts


def sample():
    return pd.DataFrame({
        "camera_id": ["a", "a", "b"], "filename": ["one", "two", "three"],
        "ts_filename": pd.to_datetime(["2026-08-25T07:00+07:00", "2026-08-25T08:00+07:00", "2026-08-25T09:00+07:00"]),
        "car": [0, 1, 6], "motor": [0, 1, 0], "truck": [0, 1, 2], "bus": [0, 1, 0],
        "bicycle": [2, 0, 0], "person": [4, 1, 0], "day_night": ["day", "day", "day"],
        "date_local": pd.to_datetime(["2026-08-25"] * 3), "hour_local": [7, 8, 9],
        "rain_mm_hour": [0., .5, 2.], "day_type": ["weekday"] * 3,
    })


class AnalysisTests(unittest.TestCase):
    def test_source_validation(self):
        source = sample()
        validate_counts(source)
        with self.assertRaisesRegex(ValueError, "truck"):
            validate_counts(source.assign(truck=[0, -1, 2]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_counts(pd.concat([source, source.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "timezone"):
            validate_counts(source.assign(ts_filename=source.ts_filename.dt.tz_localize(None)))

    def test_vehicle_classes_and_undefined_shares(self):
        source = sample()
        result = add_features(source)
        self.assertEqual(result.motor_vehicles_total.tolist(), [0, 4, 8])
        self.assertEqual(result.objects_total.tolist(), [6, 5, 8])
        self.assertTrue(pd.isna(result.motor_share_all.iloc[0]))
        self.assertEqual(result.heavy_vehicle_share.iloc[1], .5)
        pd.testing.assert_frame_equal(result[source.columns], source)

    def test_aggregate_uses_ratio_of_sums(self):
        table = summarize(add_features(sample()), ["day_type"])
        self.assertEqual(table.images.iloc[0], 3)
        self.assertEqual(table.mean_motor_vehicles.iloc[0], 4)
        self.assertAlmostEqual(table.motor_share_all.iloc[0], 1 / 12)
        self.assertAlmostEqual(table.zero_motor_vehicle_fraction.iloc[0], 1 / 3)

    def test_time_and_rain_boundaries(self):
        source = pd.concat([sample().iloc[[0]]] * 8, ignore_index=True)
        source["hour_local"] = [5, 6, 8, 9, 15, 16, 18, 19]
        source["rain_mm_hour"] = [0, .1, .9, 1, 4.9, 5, 0, 0]
        result = add_features(source)
        self.assertEqual(result.time_period.tolist(), ["overnight_22_06", "morning_06_09", "morning_06_09",
                                                       "daytime_09_16", "daytime_09_16", "evening_16_19",
                                                       "evening_16_19", "late_19_22"])
        self.assertEqual(result.rain_band.iloc[:6].tolist(), ["dry_0", "positive_lt1", "positive_lt1", "1_to_lt5", "1_to_lt5", "5_or_more"])
        self.assertEqual(result.clock_day_night.iloc[6], "night")

    def test_matched_comparison_requires_both_categories_and_minimum(self):
        rows = []
        for camera, dry, wet in [("a", [10]*3, [6]*6), ("b", [1]*3, [3]*3), ("c", [7]*3, []), ("d", [8], [9])]:
            for category, values in [("dry", dry), ("wet", wet)]:
                rows.extend({"camera_id": camera, "rain_group": category, "motor_vehicles_total": x} for x in values)
        pairs = matched_comparison(pd.DataFrame(rows), ["camera_id"], "rain_group", "dry", "wet")
        self.assertEqual(pairs.camera_id.tolist(), ["a", "b"])
        self.assertEqual(pairs.comparison_minus_baseline.mean(), -1)
        dry_only = pd.DataFrame(rows).query("rain_group == 'dry'")
        self.assertTrue(matched_comparison(dry_only, ["camera_id"], "rain_group", "dry", "wet").empty)

    def test_missing_frames_are_not_zero_traffic(self):
        manifest = sample()[IMAGE_KEY]
        table = coverage_tables(manifest.iloc[[0, 2]], manifest)["coverage_camera_hour"]
        missing = table.loc[table.hour_local.eq(8)].iloc[0]
        self.assertEqual(missing.inventory_images, 1)
        self.assertEqual(missing.analyzed_images, 0)
        self.assertEqual(missing.missing_counts, 1)
        with self.assertRaisesRegex(ValueError, "not found"):
            coverage_tables(manifest.assign(filename=["unknown", "two", "three"]), manifest)

    def test_join_rejects_missing_or_duplicate_lookup_keys(self):
        left = pd.DataFrame({"camera_id": ["a", "a", "b"]})
        with self.assertRaisesRegex(ValueError, "unmatched"):
            attach(left, pd.DataFrame({"camera_id": ["a"], "rain": [0]}), ["camera_id"], "test")
        with self.assertRaises(pd.errors.MergeError):
            attach(left, pd.DataFrame({"camera_id": ["a", "a", "b"], "rain": [0, 1, 0]}), ["camera_id"], "test")


if __name__ == "__main__":
    unittest.main()
