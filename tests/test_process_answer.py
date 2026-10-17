"""Presentation depends on selected computed operations, not canned questions."""
import csv
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from metso_assistant.process_answer import process_answer
from metso_assistant.process_ops import ProcessWorkspace
from tests.test_process_ops import user


IRON, SILICA, FLOW = "% Iron Concentrate", "% Silica Concentrate", "Invented Signal Flow"


class ProcessAnswerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "invented.csv"
        self.write([(50, 6, 100), (55, 5, 200), (60, 4, 300), (65, 3, 400)])

    def write(self, rows, timestamps=None):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", IRON, SILICA, FLOW])
            for index, row in enumerate(rows):
                stamp = timestamps[index] if timestamps else (datetime(2022, 4, 5) + timedelta(hours=index)).isoformat(sep=" ")
                writer.writerow([stamp, *row])

    def workspace(self, inputs):
        return ProcessWorkspace(self.path, inputs)

    @staticmethod
    def entry(operation, result):
        return {"action": {"action": operation}, "result": result}

    def render(self, workspace, history, message="A fresh question"):
        return process_answer(workspace, history, workspace.describe(), [message])

    def test_each_outside_reading_and_direction_precedes_evidence(self):
        workspace = self.workspace([user("U001", 70), user("U002", 1)])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}, {"column": SILICA, "input_ref": "U002"}])
        answer = self.render(workspace, [self.entry("compare", result)])
        text = answer["data"]["summary"]
        first = text.split("\n\n")[0]
        self.assertIn(IRON + " (70%) is above its recorded maximum", first)
        self.assertIn(SILICA + " (1%) is below its recorded minimum", first)
        self.assertIn("supplied 70%; recorded range 50%–65%", text)
        self.assertIn("supplied 1%; recorded range 3%–6%", text)
        self.assertEqual(answer["status"], "ok")

    def test_centrality_is_explicitly_marginal_not_joint_normality(self):
        workspace = self.workspace([user("U001", 250, "number")])
        result = workspace.compare([{"column": FLOW, "input_ref": "U001"}])
        text = self.render(workspace, [self.entry("compare", result)])["data"]["summary"]
        self.assertIn("individual readings are broadly typical", text)
        self.assertIn("does not establish that the combination is typical", text)
        self.assertIn("equal to the median 250", text)
        self.assertIn("units not stated", text)

    def test_mixed_quartile_positions_do_not_become_a_favorable_quality_score(self):
        workspace = self.workspace([user("U001", 60), user("U002", 6)])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}, {"column": SILICA, "input_ref": "U002"}])
        text = self.render(workspace, [self.entry("compare", result)], "Are these readings better?")["data"]["summary"]
        self.assertIn("Both readings fall within this plant's historical ranges", text)
        self.assertIn("Silica Concentrate is relatively high", text)
        self.assertIn("above the upper quartile", text)
        self.assertIn("does not establish customer targets, an overall improvement", text)
        self.assertNotIn("normal operation", text)
        self.assertNotIn("both more iron", text)

    def test_equal_sample_count_changes_only_the_requested_assessment_layout(self):
        workspace = self.workspace([user("U001", 6), user("U002", 60)])
        result = workspace.compare([{"column": SILICA, "input_ref": "U001"}, {"column": IRON, "input_ref": "U002"}])
        text = self.render(workspace, [self.entry("compare", result)])["data"]["summary"]
        old_summary = (
            "Both readings fall within this plant's historical ranges. Silica Concentrate is relatively high (above its upper quartile).\n\n"
            "- % Silica Concentrate: supplied 6%; above the upper quartile (quartiles 3.75%–5.25%); above the median 4.5%; 4 hourly observations.\n"
            "- % Iron Concentrate: supplied 60%; within the historical middle half (quartiles 53.75%–61.25%); above the median 57.5%; 4 hourly observations.\n\n"
            "Historical position does not establish customer targets, an overall improvement, causes or operating setpoints. Confirm units and plant comparability.\n\n"
            "Sources:\n- invented.csv")
        expected = old_summary.replace("\n\n- % Silica", "\n\nFrom 4 hourly historical observations:\n\n- % Silica", 1).replace("; 4 hourly observations.", ".")
        self.assertEqual(text, expected)
        self.assertEqual(text.count("4 hourly historical observations"), 1)
        self.assertNotIn("; 4 hourly observations", text)

    def test_unequal_populations_keep_their_actual_per_column_counts(self):
        self.write([(50, 6, 100), (55, "", 200), (60, 4, 300), (65, 3, 400)])
        workspace = self.workspace([user("U001", 60), user("U002", 6)])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}, {"column": SILICA, "input_ref": "U002"}])
        text = self.render(workspace, [self.entry("compare", result)])["data"]["summary"]
        self.assertIn("sample sizes differ by column", text)
        self.assertIn("; 4 hourly observations.", text)
        self.assertIn("; 3 hourly observations.", text)
        self.assertNotIn("From 4 hourly historical observations:", text)

    def test_requested_coverage_and_gaps_show_source_period_population_and_missing_hours(self):
        self.write([(50, 6, 100), (55, 5, 200), (60, 4, 300)],
                   timestamps=["2022-04-05 00:00:00", "2022-04-05 01:00:00", "2022-04-05 05:00:00"])
        workspace = self.workspace([])
        result = workspace.describe()
        result["data"]["requested_sections"] = ["coverage", "gaps"]
        answer = self.render(workspace, [self.entry("describe", result)])
        text = answer["data"]["summary"]
        self.assertEqual(answer["status"], "ok")
        self.assertIn("Recorded period: 2022-04-05 00:00:00 to 2022-04-05 05:00:00 (timezone not stated); 3 distinct valid hourly observations.", text)
        self.assertIn("3 missing hourly timestamps across 1 recorded gap", text)
        self.assertIn("After 2022-04-05 01:00:00 and before 2022-04-05 05:00:00: 3 missing hours", text)
        self.assertNotIn("Available CSV columns", text)
        self.assertNotRegex(text, r"PF[0-9a-f]{20}")

    def test_empty_timestamp_population_does_not_claim_complete_time_coverage(self):
        self.write([])
        workspace = self.workspace([])
        result = workspace.describe()
        result["data"]["requested_sections"] = ["coverage", "gaps"]
        text = self.render(workspace, [self.entry("describe", result)])["data"]["summary"]
        self.assertIn("Recorded period: unavailable", text)
        self.assertIn("Timestamp gaps cannot be assessed", text)
        self.assertNotIn("There are no missing hourly timestamps", text)

    def test_latest_comparison_for_each_column_preserves_other_selected_columns(self):
        workspace = self.workspace([user("U001", 53), user("U002", 60), user("U003", 6)])
        old = workspace.compare([{"column": IRON, "input_ref": "U001"}])
        silica = workspace.compare([{"column": SILICA, "input_ref": "U003"}])
        corrected = workspace.compare([{"column": IRON, "input_ref": "U002"}])
        history = [self.entry("compare", item) for item in (old, silica, corrected)]
        text = self.render(workspace, history)["data"]["summary"]
        self.assertIn("supplied 60%", text)
        self.assertIn("supplied 6%", text)
        self.assertNotIn("supplied 53%", text)
        self.assertEqual(text.count("- " + IRON + ":"), 1)

    def test_distribution_table_supports_arbitrary_column_and_explicit_unknown_units(self):
        workspace = self.workspace([])
        result = workspace.distribution([FLOW])
        text = self.render(workspace, [self.entry("distribution", result)], "Show the signal distribution.")["data"]["summary"]
        self.assertIn("| " + FLOW + " | not stated | 4 | 100 | 175 | 250 | 325 | 400 | 250 |", text)
        self.assertIn("Mean", text)
        self.assertNotIn("L/min", text)

    def test_count_uses_actual_user_thresholds_and_explicit_and_directions(self):
        workspace = self.workspace([user("U001", 200, "number"), user("U002", 4)])
        result = workspace.count([{"column": FLOW, "operator": "ge", "input_ref": "U001"},
                                  {"column": SILICA, "operator": "le", "input_ref": "U002"}], "and")
        text = self.render(workspace, [self.entry("count", result)])["data"]["summary"]
        self.assertIn(FLOW + " (units not stated) >= 200 AND " + SILICA + " <= 4%", text)
        self.assertIn("2 of 4 complete hourly observations (50%)", text)
        self.assertNotIn("175", text)
        self.assertNotIn("quartile", text)

    def test_or_count_keeps_its_own_complete_hour_population(self):
        self.write([(50, 6, 100), (55, "", 200), (60, 4, ""), (65, 3, 400)])
        workspace = self.workspace([user("U001", 150, "number"), user("U002", 5)])
        result = workspace.count([{"column": FLOW, "operator": "gt", "input_ref": "U001"},
                                  {"column": SILICA, "operator": "gt", "input_ref": "U002"}], "or")
        text = self.render(workspace, [self.entry("count", result)])["data"]["summary"]
        self.assertIn(" OR ", text)
        self.assertIn("2 of 2 complete hourly observations (100%)", text)
        self.assertIn("missing value in any selected predicate column", text)

    def test_empty_count_population_has_no_fake_zero_percentage(self):
        self.write([(50, 6, ""), (55, 5, "")])
        workspace = self.workspace([user("U001", 100, "number")])
        result = workspace.count([{"column": FLOW, "operator": "gt", "input_ref": "U001"}])
        answer = self.render(workspace, [self.entry("count", result)])
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertIn("a percentage is unavailable", answer["data"]["summary"])
        self.assertNotIn("0%", answer["data"]["summary"])

    def test_no_private_identifiers_or_json_are_visible_and_source_is_readable(self):
        workspace = self.workspace([user("U001", 60)])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}])
        text = self.render(workspace, [self.entry("compare", result)])["data"]["summary"]
        self.assertNotIn("U001", text)
        self.assertNotRegex(text, r"PF[0-9a-f]{20}|\[P01\]")
        self.assertNotIn("input_ref", text)
        self.assertIn("Sources:\n- invented.csv", text)

    def test_stale_source_and_invalid_selected_refs_fail_closed(self):
        workspace = self.workspace([user("U001", 60)])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}])
        result["data"]["comparisons"][0]["facts"]["median"] = "invented-private-id"
        answer = self.render(workspace, [self.entry("compare", result)])
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertNotIn("invented-private-id", answer["data"]["summary"])
        result = workspace.compare([{"column": IRON, "input_ref": "U001"}])
        self.write([(50, 6, 999)])
        answer = process_answer(workspace, [self.entry("compare", result)], None, [])
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertEqual(answer["sources"], [])

    def test_no_successful_operation_is_an_explicit_insufficient_result(self):
        answer = self.render(self.workspace([]), [])
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertEqual(answer["data"]["response"]["scope"], "historical_assessment")
        self.assertIsNone(answer["question"])
        self.assertEqual(answer["missing_fields"], [])


if __name__ == "__main__":
    unittest.main()
