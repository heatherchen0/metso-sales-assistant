"""Generic CSV-operation boundaries, independent of the demonstration readings."""
import csv
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metso_assistant.process_ops import ProcessWorkspace


IRON = "% Iron Concentrate"
SILICA = "% Silica Concentrate"
FLOW = "Synthetic Reagent Flow"


def user(identifier, value, unit="percent"):
    return {"id": identifier, "label": "Explicit synthetic user threshold", "value": value,
            "unit": unit, "kind": "user_input", "parents": [], "source_ids": [], "assumptions": []}


class ProcessOperationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.csv"

    def write(self, rows, extra=(FLOW,), timestamps=None, delimiter=","):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(["date", IRON, SILICA, *extra])
            for index, values in enumerate(rows):
                stamp = timestamps[index] if timestamps else (datetime(2021, 2, 3) + timedelta(hours=index)).isoformat(sep=" ")
                writer.writerow([stamp, *values])
        return self.path

    def baseline(self):
        return self.write([(50, 6, 100), (55, 5, 200), (60, 4, 300), (65, 3, 400)])

    @staticmethod
    def values(result):
        return {fact["id"]: fact["value"] for fact in result["facts"]}

    def test_describe_exposes_actual_columns_units_and_numeric_diagnostic_refs(self):
        workspace = ProcessWorkspace(self.baseline())
        result = workspace.describe()
        self.assertEqual(result["status"], "ok")
        inventory = {column["name"]: column for column in result["data"]["columns"]}
        self.assertEqual(inventory[IRON]["unit"], "percent")
        self.assertIsNone(inventory[FLOW]["unit"])
        self.assertEqual(inventory["date"]["type"], "datetime")
        source_rows = result["data"]["profile"]["source_rows"]
        self.assertEqual(self.values(result)[source_rows], 4)
        self.assertEqual(self.values(result)[result["data"]["profile"]["column_count"]], 4)
        self.assertEqual(self.values(result)[result["data"]["profile"]["timestamp_gap_count"]], 0)
        self.assertTrue(all(isinstance(ref, str) for ref in result["data"]["profile"].values()))
        self.assertEqual(result["sources"][0]["id"], "P01")
        self.assertNotIn("row_count", result["sources"][0])
        self.assertNotIn("summary", result["data"])

    def test_distribution_works_on_third_column_without_quality_direction_rules(self):
        result = ProcessWorkspace(self.baseline()).distribution([FLOW])
        row = result["data"]["distributions"][0]
        values = self.values(result)
        expected = {"n": 4, "min": 100, "q25": 175, "median": 250, "q75": 325, "max": 400, "mean": 250}
        self.assertEqual({name: values[ref] for name, ref in row["facts"].items()}, expected)
        self.assertTrue(all(fact["unit"] is None for fact in result["facts"] if fact["metric"] != "n"))

    def test_outside_both_readings_are_independently_identified_with_distances(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", Decimal("80")), user("U002", Decimal("10"))])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}, {"column": SILICA, "input_ref": "U002"}])
        values = self.values(result)
        comparisons = result["data"]["comparisons"]
        self.assertEqual([row["range_status"] for row in comparisons], ["above_max", "above_max"])
        self.assertEqual(values[comparisons[0]["facts"]["distance_above_max"]], 15)
        self.assertEqual(values[comparisons[1]["facts"]["distance_above_max"]], 4)
        self.assertEqual(values[comparisons[0]["facts"]["empirical_percentile"]], 100)
        distances = [fact for fact in result["facts"] if fact["metric"] == "distance_above_max"]
        self.assertTrue(all(fact["unit"] == "percentage_point" for fact in distances))
        self.assertNotIn("joint_comparison", result["data"])
        self.assertNotIn("conclusion", result["data"])

    def test_mixed_readings_return_marginal_evidence_not_a_canned_assessment(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 60), user("U002", 6)])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}, {"column": SILICA, "input_ref": "U002"}])
        values = self.values(result)
        iron, silica = result["data"]["comparisons"]
        self.assertEqual(values[iron["facts"]["empirical_percentile"]], 62.5)
        self.assertEqual(values[silica["facts"]["empirical_percentile"]], 87.5)
        self.assertEqual(iron["range_status"], "within_observed_range")
        self.assertEqual(silica["range_status"], "within_observed_range")
        self.assertEqual(iron["middle_50_status"], "within_middle_50")
        self.assertEqual(silica["middle_50_status"], "above_q75")
        self.assertNotIn("classification", result["data"])

    def test_middle_half_and_median_relationships_include_exact_boundaries(self):
        self.baseline()
        cases = [(100, "below_q25", "below"), (175, "within_middle_50", "below"),
                 (250, "within_middle_50", "equal"), (325, "within_middle_50", "above"),
                 (400, "above_q75", "above")]
        for reading, expected_middle, expected_median in cases:
            with self.subTest(reading=reading):
                workspace = ProcessWorkspace(self.path, [user("U001", Decimal(reading), "number")])
                result = workspace.compare([{"column": FLOW, "input_ref": "U001"}])
                compared = result["data"]["comparisons"][0]
                self.assertEqual(compared["middle_50_status"], expected_middle)
                self.assertEqual(compared["median_relation"], expected_median)
                self.assertEqual(compared["mean_relation"], expected_median)

    def test_mean_and_median_relationships_are_independently_computed(self):
        path = self.write([(50, 6, 1), (55, 5, 2), (60, 4, 4), (65, 3, 8)])
        workspace = ProcessWorkspace(path, [user("U001", Decimal("3.5"), "number")])
        result = workspace.compare([{"column": FLOW, "input_ref": "U001"}])
        compared = result["data"]["comparisons"][0]
        values = self.values(result)
        self.assertEqual(values[compared["facts"]["median"]], 3)
        self.assertEqual(values[compared["facts"]["mean"]], 3.75)
        self.assertEqual(compared["median_relation"], "above")
        self.assertEqual(compared["mean_relation"], "below")
        self.assertEqual(compared["middle_50_status"], "within_middle_50")

    def test_constant_column_equality_and_empty_column_relationships_are_explicit(self):
        path = self.write([(50, 6, 8), (55, 5, 8)])
        workspace = ProcessWorkspace(path, [user("U001", 8, "number")])
        compared = workspace.compare([{"column": FLOW, "input_ref": "U001"}])["data"]["comparisons"][0]
        self.assertEqual(compared["middle_50_status"], "within_middle_50")
        self.assertEqual(compared["median_relation"], "equal")
        self.assertEqual(compared["mean_relation"], "equal")
        self.write([(50, 6, ""), (55, 5, "")])
        unavailable = ProcessWorkspace(self.path, [user("U001", 8, "number")]).compare(
            [{"column": FLOW, "input_ref": "U001"}])["data"]["comparisons"][0]
        for field in ("range_status", "middle_50_status", "median_relation", "mean_relation"):
            self.assertEqual(unavailable[field], "unavailable")

    def test_compare_third_column_accepts_number_unit_without_inventing_flow_units(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 250, "number")])
        result = workspace.compare([{"column": FLOW, "input_ref": "U001"}])
        row = result["data"]["comparisons"][0]
        self.assertEqual(self.values(result)[row["facts"]["empirical_percentile"]], 50)
        self.assertEqual(result["status"], "ok")

    def test_predicate_directions_are_arbitrary_and_strictness_is_preserved(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 60), user("U002", 4)])
        conditions = [{"column": IRON, "operator": "lt", "input_ref": "U001"},
                      {"column": SILICA, "operator": "gt", "input_ref": "U002"}]
        result = workspace.count(conditions)
        values, refs = self.values(result), result["data"]["facts"]
        self.assertEqual(values[refs["population"]], 4)
        self.assertEqual(values[refs["count"]], 2)
        self.assertEqual(values[refs["percentage"]], 50)
        self.assertEqual([p["symbol"] for p in result["data"]["conditions"]], ["<", ">"])
        conditions[0]["operator"] = "le"
        conditions[1]["operator"] = "ge"
        inclusive = workspace.count(conditions)
        self.assertEqual(self.values(inclusive)[inclusive["data"]["facts"]["count"]], 3)

    def test_or_predicates_and_third_column_are_not_rewritten_as_quality_directions(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 150, "number"), user("U002", 3)])
        result = workspace.count([{"column": FLOW, "operator": "lt", "input_ref": "U001"},
                                  {"column": SILICA, "operator": "eq", "input_ref": "U002"}], combine="or")
        self.assertEqual(self.values(result)[result["data"]["facts"]["count"]], 2)
        self.assertEqual(result["data"]["combine"], "or")

    def test_missing_cells_define_complete_predicate_population_even_for_or(self):
        path = self.write([(50, 6, 100), (55, "", 200), (60, 4, ""), (65, 3, 400)])
        workspace = ProcessWorkspace(path, [user("U001", 150, "number"), user("U002", 5)])
        result = workspace.count([{"column": FLOW, "operator": "gt", "input_ref": "U001"},
                                  {"column": SILICA, "operator": "gt", "input_ref": "U002"}], "or")
        values, refs = self.values(result), result["data"]["facts"]
        self.assertEqual(values[refs["population"]], 2)
        self.assertEqual(values[refs["count"]], 2)
        distribution = workspace.distribution([FLOW])
        self.assertEqual(self.values(distribution)[distribution["data"]["distributions"][0]["facts"]["n"]], 3)

    def test_empty_population_has_no_invented_zero_percentage(self):
        path = self.write([(50, 6, ""), (55, 5, "")])
        workspace = ProcessWorkspace(path, [user("U001", 150, "number")])
        result = workspace.count([{"column": FLOW, "operator": "gt", "input_ref": "U001"}])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("percentage", result["data"]["facts"])
        self.assertEqual(self.values(result)[result["data"]["facts"]["population"]], 0)
        distribution = workspace.distribution([FLOW])
        self.assertEqual(distribution["status"], "insufficient_evidence")
        self.assertEqual(set(distribution["data"]["distributions"][0]["facts"]), {"n"})

    def test_source_change_invalidates_existing_workspace_and_new_facts_have_new_ids(self):
        workspace = ProcessWorkspace(self.baseline())
        original = workspace.distribution([FLOW])
        self.write([(50, 6, 1), (55, 5, 3)])
        stale = workspace.distribution([FLOW])
        self.assertEqual(stale["status"], "insufficient_evidence")
        self.assertEqual(stale["facts"], [])
        fresh = ProcessWorkspace(self.path).distribution([FLOW])
        self.assertNotEqual(original["sources"][0]["sha256"], fresh["sources"][0]["sha256"])
        self.assertTrue(set(self.values(original)).isdisjoint(self.values(fresh)))

    def test_source_change_during_operation_cannot_leave_partial_registry_facts(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 60)])
        original_fact = workspace._fact
        changed = False

        def changing_source(*args, **kwargs):
            nonlocal changed
            fact = original_fact(*args, **kwargs)
            if not changed:
                changed = True
                self.write([(50, 6, 1), (55, 5, 3)])
            return fact

        with patch.object(workspace, "_fact", side_effect=changing_source):
            result = workspace.distribution([FLOW])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["facts"], [])
        self.assertEqual(set(workspace.facts), {"U001"})

    def test_identical_source_and_input_produce_stable_facts_without_mutation(self):
        inputs = [user("U001", Decimal("250"), "number")]
        first = ProcessWorkspace(self.baseline(), inputs)
        second = ProcessWorkspace(self.path, inputs)
        readings = [{"column": FLOW, "input_ref": "U001"}]
        self.assertEqual(first.compare(readings), second.compare(readings))
        first.facts["U001"]["value"] = 99999
        self.assertEqual(first.compare(readings), second.compare(readings))
        self.assertEqual(inputs[0]["value"], Decimal("250"))

    def test_computed_results_cannot_be_promoted_to_user_thresholds(self):
        workspace = ProcessWorkspace(self.baseline())
        distribution = workspace.distribution([FLOW])
        median_ref = distribution["data"]["distributions"][0]["facts"]["median"]
        result = workspace.compare([{"column": FLOW, "input_ref": median_ref}])
        self.assertEqual(result["status"], "invalid_request")
        self.assertEqual(result["facts"], [])

    def test_free_model_constants_and_unknown_references_are_rejected(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 60)])
        invalid = [workspace.compare([{"column": IRON, "input_ref": "invented"}]),
                   workspace.compare([{"column": IRON, "input_ref": "U001", "value": 65}]),
                   workspace.count([{"column": IRON, "operator": "gt", "value": 65}]),
                   workspace.count([{"column": IRON, "operator": "approx", "input_ref": "U001"}]),
                   workspace.count([], combine="and"),
                   workspace.count([{"column": IRON, "operator": "gt", "input_ref": "U001"}], combine="xor")]
        self.assertTrue(all(result["status"] == "invalid_request" and not result["facts"] for result in invalid))

    def test_input_units_must_match_percent_or_unknown_header_unit(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 60, "hour"), user("U002", 250, "L/min"),
                                                      user("U003", 60, "number"), user("U004", 250, "percent")])
        for column, ref in [(IRON, "U001"), (FLOW, "U002"), (IRON, "U003"), (FLOW, "U004")]:
            result = workspace.compare([{"column": column, "input_ref": ref}])
            self.assertEqual(result["status"], "invalid_request", (column, ref))

    def test_unknown_and_datetime_columns_have_explicit_unavailable_signal(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 5)])
        for column in ("Invented Column", "date"):
            result = workspace.distribution([column])
            self.assertEqual(result["status"], "invalid_request")
            self.assertIn(FLOW, result["data"]["available_columns"])
            self.assertEqual(result["facts"], [])

    def test_duplicate_hours_and_conflicting_third_column_use_existing_cleaning(self):
        path = self.write([(50, 6, 100), (50, 6, 300), (55, 5, 200)],
                          timestamps=["2021-01-01 00:00:00", "2021-01-01 00:00:00", "2021-01-01 01:00:00"])
        workspace = ProcessWorkspace(path)
        result = workspace.distribution([FLOW])
        values, refs = self.values(result), result["data"]["distributions"][0]["facts"]
        self.assertEqual(values[refs["n"]], 1)
        self.assertEqual(values[refs["median"]], 200)
        described = workspace.describe()
        conflict = described["data"]["cell_issues"][FLOW]["conflicting_duplicate_hours"]
        self.assertEqual(self.values(described)[conflict], 1)

    def test_decimal_comma_and_zero_are_numeric_without_imputation(self):
        path = self.write([(50, 6, "0"), (55, 5, "1,5"), (60, 4, "")], delimiter=";")
        result = ProcessWorkspace(path).distribution([FLOW])
        refs, values = result["data"]["distributions"][0]["facts"], self.values(result)
        self.assertEqual(values[refs["n"]], 2)
        self.assertEqual(values[refs["min"]], 0)
        self.assertEqual(values[refs["mean"]], .75)

    def test_invalid_user_inputs_fail_before_any_calculation(self):
        self.baseline()
        for invalid in [user("U001", float("nan")), user("U001", True), user("U001", "60"),
                        {**user("U001", 60), "kind": "computed"}, {**user("U001", 60), "id": ""}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ProcessWorkspace(self.path, [invalid])
        with self.assertRaises(ValueError):
            ProcessWorkspace(self.path, [user("U001", 1), user("U001", 2)])

    def test_missing_or_unusable_source_returns_insufficient_evidence(self):
        self.assertEqual(ProcessWorkspace(self.path).describe()["status"], "insufficient_evidence")
        self.path.write_text("wrong,headers\n1,2\n", encoding="utf-8")
        self.assertEqual(ProcessWorkspace(self.path).describe()["status"], "insufficient_evidence")

    def test_every_returned_numeric_result_has_registry_identity_and_source(self):
        workspace = ProcessWorkspace(self.baseline(), [user("U001", 60)])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}])
        for fact in result["facts"]:
            self.assertEqual(fact, workspace.facts[fact["id"]])
            self.assertEqual(fact["source_ids"], ["P01"])
            self.assertEqual(fact["kind"], "computed")
            self.assertIsNone(fact["currency"])
            self.assertIn("parents", fact)
            self.assertIn("assumptions", fact)
        comparison_ids = [fact for fact in result["facts"] if fact["operation"] == "compare"]
        self.assertTrue(all("U001" in fact["parents"] for fact in comparison_ids))


if __name__ == "__main__":
    unittest.main()
