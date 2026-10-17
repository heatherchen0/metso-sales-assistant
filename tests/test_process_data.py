"""Synthetic CSV reader and shared-statistics regressions, without answer policy."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from metso_assistant.process_data import (
    IRON_COLUMN, SILICA_COLUMN, _distribution, _load, quantile,
)


class ProcessDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "synthetic.csv"

    def write(self, rows, delimiter=",", extra=None):
        headers = ["date", IRON_COLUMN, SILICA_COLUMN, *(extra or [])]
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(headers)
            writer.writerows(rows)
        return self.path

    def test_linear_quantiles_and_distribution_are_explicit(self):
        values = [4.0, 2.0, 1.0, 2.0]
        self.assertEqual(quantile(values, .25), 1.75)
        self.assertEqual(quantile(values, .75), 2.5)
        self.assertEqual(quantile(values, 0), 1)
        self.assertEqual(quantile(values, 1), 4)
        self.assertEqual(_distribution(values), {
            "n": 4, "min": 1, "q25": 1.75, "median": 2, "q75": 2.5, "max": 4,
        })
        self.assertEqual(values, [4.0, 2.0, 1.0, 2.0])
        for observations, probability in [([], .5), (values, -.1), (values, 1.1)]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                quantile(observations, probability)

    def test_duplicate_hours_do_not_vote_twice_and_conflicts_are_not_averaged(self):
        self.write([
            ["2020-01-01 00:00:00", 50, 6], ["2020-01-01 00:00:00", 50, 6],
            ["2020-01-01 01:00:00", 60, 4], ["2020-01-01 01:00:00", 80, 4],
            ["2020-01-01 02:00:00", 70, 2],
        ])
        records, _, profile, _ = _load(self.path)
        self.assertEqual(profile["source_rows"], 5)
        self.assertEqual(profile["unique_valid_hours"], 3)
        self.assertEqual(profile["duplicate_timestamp_rows"], 2)
        self.assertEqual(profile["duplicate_timestamp_groups"], 2)
        self.assertEqual(profile["cell_issues"][IRON_COLUMN]["conflicting_duplicate_hours"], 1)
        self.assertEqual(records[0]["source_rows"], [2, 3])
        self.assertIsNone(records[1][IRON_COLUMN])
        self.assertEqual(records[1][SILICA_COLUMN], 4)
        values = [row[IRON_COLUMN] for row in records if row[IRON_COLUMN] is not None]
        self.assertEqual(_distribution(values)["n"], 2)
        self.assertEqual(_distribution(values)["median"], 60)
        self.assertEqual(profile["complete_quality_hours"], 2)

    def test_decimal_comma_in_semicolon_csv_and_missing_zero_distinction(self):
        self.write([
            ["2020-01-01 00:00:00", "50,5", "6,5", "0"],
            ["2020-01-01 01:00:00", "60,5", "4,5", ""],
            ["2020-01-01 02:00:00", "70,5", "2,5", "10,5"],
        ], delimiter=";", extra=["Starch Flow"])
        records, source, profile, _ = _load(self.path)
        self.assertEqual(profile["delimiter"], ";")
        self.assertEqual(_distribution([row[IRON_COLUMN] for row in records])["median"], 60.5)
        flow = [row["Starch Flow"] for row in records if row["Starch Flow"] is not None]
        self.assertEqual(_distribution(flow)["n"], 2)
        self.assertEqual(_distribution(flow)["min"], 0)
        self.assertIsNone(records[1]["Starch Flow"])
        self.assertEqual(profile["cell_issues"]["Starch Flow"]["missing"], 1)
        self.assertEqual(source["columns"][-1], "Starch Flow")

    def test_quoted_decimal_comma_in_comma_csv_is_numeric(self):
        self.write([["2020-01-01 00:00:00", "50,5", "6,5"],
                    ["2020-01-01 01:00:00", "60,5", "4,5"]])
        records, _, profile, _ = _load(self.path)
        self.assertEqual(profile["delimiter"], ",")
        self.assertEqual(_distribution([row[IRON_COLUMN] for row in records])["min"], 50.5)

    def test_missing_quality_invalid_percent_and_time_gaps_are_not_imputed(self):
        self.write([
            ["2020-01-01 00:00:00", 50, 6], ["2020-01-01 01:00:00", "", 5],
            ["2020-01-01 02:00:00", 101, 4], ["2020-01-01 04:00:00", 70, 2],
        ])
        records, _, profile, warnings = _load(self.path)
        self.assertEqual(len(records), 4)
        self.assertEqual([row["timestamp"].hour for row in records], [0, 1, 2, 4])
        self.assertEqual(profile["missing_hours"], 1)
        self.assertEqual(profile["complete_quality_hours"], 2)
        self.assertEqual(profile["cell_issues"][IRON_COLUMN]["missing"], 1)
        self.assertEqual(profile["cell_issues"][IRON_COLUMN]["invalid"], 1)
        self.assertIsNone(records[1][IRON_COLUMN])
        self.assertIsNone(records[2][IRON_COLUMN])
        paired = [row for row in records
                  if row[IRON_COLUMN] is not None and row[SILICA_COLUMN] is not None]
        self.assertEqual(_distribution([row[SILICA_COLUMN] for row in paired])["n"], 2)
        self.assertEqual(_distribution([row[SILICA_COLUMN] for row in paired])["median"], 4)
        self.assertTrue(any("no numeric imputation" in warning for warning in warnings))

    def test_repeated_pairs_only_count_consecutive_hours(self):
        self.write([
            ["2020-01-01 00:00:00", 50, 6], ["2020-01-01 01:00:00", 50, 6],
            ["2020-01-01 04:00:00", 50, 6], ["2020-01-01 05:00:00", 70, 2],
        ])
        _, _, profile, warnings = _load(self.path)
        self.assertEqual(profile["adjacent_repeated_quality_pairs"], 1)
        self.assertEqual(profile["unique_quality_pairs"], 2)
        self.assertTrue(any("interpolation" in warning for warning in warnings))

    def test_source_change_recomputes_hash_and_statistics(self):
        self.write([["2020-01-01 00:00:00", 40, 8], ["2020-01-01 01:00:00", 50, 7]])
        first, first_source, _, _ = _load(self.path)
        self.assertEqual(first_source["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.write([["2020-01-01 00:00:00", 70, 2], ["2020-01-01 01:00:00", 80, 1]])
        second, second_source, _, _ = _load(self.path)
        self.assertNotEqual(first_source["sha256"], second_source["sha256"])
        self.assertEqual(second_source["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(_distribution([row[IRON_COLUMN] for row in first])["median"], 45)
        self.assertEqual(_distribution([row[IRON_COLUMN] for row in second])["median"], 75)

    def test_one_hour_has_a_valid_reader_snapshot_without_inventing_more_history(self):
        self.write([["2020-01-01 00:00:00", 50, 6]])
        records, source, profile, warnings = _load(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(profile["complete_quality_hours"], 1)
        self.assertEqual(source["date_start"], source["date_end"])
        self.assertTrue(any("Fewer than 30" in warning for warning in warnings))

    def test_missing_columns_and_missing_file_fail_explicitly(self):
        self.path.write_text("date,other\n2020-01-01,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing CSV columns"):
            _load(self.path)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            _load(self.path)

    def test_constant_quality_and_empty_distribution_are_finite(self):
        self.write([["2020-01-01 00:00:00", 50, 6], ["2020-01-01 01:00:00", 50, 6]])
        records, _, profile, _ = _load(self.path)
        stats = _distribution([row[IRON_COLUMN] for row in records])
        self.assertEqual(stats, {"n": 2, "min": 50, "q25": 50, "median": 50, "q75": 50, "max": 50})
        self.assertEqual(profile["unique_quality_pairs"], 1)
        self.assertEqual(_distribution([]), {
            "n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None,
        })
        json.dumps({"records": records, "profile": profile, "distribution": stats},
                   default=str, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
