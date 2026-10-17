"""Literal measurement binding and one-plan execution, using synthetic data."""
import csv
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from metso_assistant.adaptive import request_facts
from metso_assistant.process_ops import ProcessWorkspace
from metso_assistant.process_request import binding_candidates, execute_process_request, interpret_process_request


IRON, SILICA, FEED, FLOW = "% Iron Concentrate", "% Silica Concentrate", "% Iron Feed", "Starch Flow"
COLUMNS = [{"name": name, "type": "number", "unit": "percent" if name.startswith("%") else None}
           for name in (IRON, SILICA, FEED, FLOW)]


def plan(*, compare=(), distribution=(), count=(), describe=False):
    return {"status": "ready", "question": None, "compare": list(compare),
            "distribution": list(distribution), "count": list(count), "describe": describe}


class Model:
    def __init__(self, output):
        self.output, self.calls = output, []

    def invoke(self, system, messages, schema, **kwargs):
        self.calls.append((system, messages, schema))
        payload = json.loads(messages[0]["content"])
        return deepcopy(self.output(payload) if callable(self.output) else self.output), {"test": True}


class ProcessRequestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.csv"
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", IRON, SILICA, FEED, FLOW])
            for index, values in enumerate([(52, 6, 40, 100), (57, 5, 45, 200), (62, 4, 50, 300), (67, 3, 55, 400)]):
                writer.writerow([(datetime(2024, 1, 1) + timedelta(hours=index)).isoformat(sep=" "), *values])

    def prepare(self, messages, output):
        inputs = request_facts(messages, "flotation")
        model = Model(output)
        chosen, traces = interpret_process_request(model, messages, inputs, COLUMNS)
        return chosen, traces, ProcessWorkspace(self.path, inputs), model

    def bindings(self, messages):
        inputs = request_facts(messages, "flotation")
        pairs, audits, ambiguous = binding_candidates(messages, inputs, COLUMNS)
        return {pair["column"]: float(next(fact["value"] for fact in inputs if fact["id"] == pair["input_ref"])) for pair in pairs}, audits, ambiguous

    def test_complete_restatement_replaces_each_measurement_in_actual_order(self):
        messages = ["We get 3.1% silica and 64.5% iron in concentrate. Is this typical?",
                    "Now 2.2% silica and 65% iron in concentrate. What do you think?"]
        chosen, _, workspace, model = self.prepare(messages, lambda payload: plan(compare=payload["binding_candidates"]))
        self.assertEqual(chosen["status"], "ready")
        actual, _, ambiguous = self.bindings(messages)
        self.assertEqual(actual, {SILICA: 2.2, IRON: 65})
        self.assertEqual(ambiguous, [])
        answer, trace = execute_process_request(workspace, chosen)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(answer["status"], "ok")
        self.assertEqual(answer["data"]["operations_executed"], 1)
        executed = [entry for entry in trace if entry["operation"] == "adaptive_tool"]
        self.assertEqual(executed[0]["action"]["readings"], chosen["compare"])
        values = {row["column"]: float(workspace._input(row["input_ref"], row["column"])["value"])
                  for row in executed[0]["result"]["data"]["comparisons"]}
        self.assertEqual(values, {SILICA: 2.2, IRON: 65})

    def test_reversed_labels_and_outliers_never_use_magnitude_for_identity(self):
        for message, expected in [
            ("Concentrate iron is 2.2% and silica is 65%.", {IRON: 2.2, SILICA: 65}),
            ("Concentrate silica is 150% and iron is -8%.", {SILICA: 150, IRON: -8}),
            ("Concentrate has 65% silica and 2.2% iron.", {SILICA: 65, IRON: 2.2}),
        ]:
            with self.subTest(message=message):
                actual, _, ambiguous = self.bindings([message])
                self.assertEqual(actual, expected)
                self.assertEqual(ambiguous, [])

    def test_partial_update_keeps_other_prior_measurement_and_inherits_same_scope(self):
        messages = ["Concentrate has 6% silica and 57% iron.", "Iron is now 62%."]
        actual, audits, ambiguous = self.bindings(messages)
        self.assertEqual(actual, {SILICA: 6, IRON: 62})
        self.assertEqual(ambiguous, [])
        self.assertEqual(next(audit["turn"] for audit in audits if audit["column"] == IRON), 1)

    def test_sentence_ending_after_decimal_keeps_the_update_with_its_own_measurement(self):
        messages = ["Concentrate iron is 2.2% and concentrate silica is 65%. Are these readings typical?",
                    "Correction: concentrate iron is 66% and concentrate silica is 1.8%. Reassess those readings.",
                    "Only silica has now changed to 2.5%. Iron remains at its latest value."]
        chosen, _, workspace, _ = self.prepare(messages, lambda payload: plan(compare=payload["binding_candidates"]))
        answer, traces = execute_process_request(workspace, chosen)
        self.assertEqual(answer["status"], "ok")
        comparison = next(trace["result"] for trace in traces if trace.get("action", {}).get("action") == "compare")
        actual = {row["column"]: float(workspace._input(row["input_ref"], row["column"])["value"])
                  for row in comparison["data"]["comparisons"]}
        self.assertEqual(actual, {IRON: 66, SILICA: 2.5})
        latest = next(audit for audit in chosen["binding_evidence"] if audit["turn"] == 2)
        self.assertEqual(latest["column"], SILICA)
        self.assertEqual(latest["literal"], "2.5")
        self.assertIn("% Iron Concentrate: supplied 66%", answer["data"]["summary"])
        self.assertIn("% Silica Concentrate: supplied 2.5%", answer["data"]["summary"])

    def test_measurement_search_does_not_cross_sentences_or_newlines_in_either_direction(self):
        initial = "Concentrate has 5% silica and 57% iron."
        for latest, expected in [
            ("Only iron has now changed to 62.4%. Silica remains at its latest value.", {IRON: 62.4, SILICA: 5}),
            ("Only silica has now changed to 4.5%\nIron remains at its latest value.", {IRON: 57, SILICA: 4.5}),
            ("The concentrate silica is 6.2%. Concentrate iron has now changed to 63.5%.", {IRON: 63.5, SILICA: 6.2}),
        ]:
            with self.subTest(latest=latest):
                actual, _, ambiguous = self.bindings([initial, latest])
                self.assertEqual(actual, expected)
                self.assertEqual(ambiguous, [])

    def test_numeric_value_cannot_borrow_a_measurement_label_from_the_next_sentence(self):
        chosen, _, _, model = self.prepare(["The new reading is 4.5%. Concentrate silica needs checking."], plan(describe=True))
        self.assertEqual(chosen["status"], "clarify")
        self.assertEqual(model.calls, [])

    def test_scope_can_be_shared_within_a_sentence_but_not_borrowed_from_the_next_one(self):
        metadata = [*COLUMNS, {"name": "% Silica Feed", "type": "number", "unit": "percent"}]
        messages = ["Concentrate has 5% silica and 57% iron.",
                    "Silica has now changed to 4.5%. Feed iron will be reported later."]
        inputs = request_facts(messages, "flotation")
        pairs, _, ambiguous = binding_candidates(messages, inputs, metadata)
        values = {pair["column"]: float(next(fact["value"] for fact in inputs if fact["id"] == pair["input_ref"])) for pair in pairs}
        self.assertEqual(values, {SILICA: 4.5, IRON: 57})
        self.assertEqual(ambiguous, [])
        unknown = ["Silica is 4.5%. The concentrate sample will be checked later."]
        pairs, _, ambiguous = binding_candidates(unknown, request_facts(unknown, "flotation"), metadata)
        self.assertEqual(pairs, [])
        self.assertTrue(ambiguous)

    def test_newly_available_measurement_inherits_unique_prior_sample_scope(self):
        messages = ["The concentrate silica is 1.2%. I do not have the iron result yet.",
                    "The iron result is now available: 66.4%. Does that change the comparison?"]
        actual, _, ambiguous = self.bindings(messages)
        self.assertEqual(actual, {SILICA: 1.2, IRON: 66.4})
        self.assertEqual(ambiguous, [])
        chosen, _, workspace, _ = self.prepare(messages, lambda payload: plan(compare=payload["binding_candidates"]))
        self.assertEqual(chosen["status"], "ready")
        self.assertEqual(execute_process_request(workspace, chosen)[0]["status"], "ok")

    def test_literal_feed_and_concentrate_scope_can_differ_within_one_message(self):
        actual, _, ambiguous = self.bindings(["Feed iron is 45% and concentrate iron is 62%."])
        self.assertEqual(actual, {FEED: 45, IRON: 62})
        self.assertEqual(ambiguous, [])

    def test_missing_scope_or_unknown_measurement_clarifies_without_model(self):
        for message in ["Iron is 62%.", "Concentrate contains 4% unknownite and 62% iron."]:
            with self.subTest(message=message):
                chosen, _, workspace, model = self.prepare([message], plan(describe=True))
                self.assertEqual(chosen["status"], "clarify")
                self.assertEqual(len(model.calls), 0)
                answer, traces = execute_process_request(workspace, chosen)
                self.assertEqual(answer["status"], "clarify")
                self.assertEqual(answer["data"]["operations_executed"], 0)
                self.assertTrue(answer["question"])

    def test_model_swapped_inputs_are_rejected_instead_of_repaired(self):
        def swapped(payload):
            first, second = payload["binding_candidates"]
            return plan(compare=[{"column": first["column"], "input_ref": second["input_ref"]},
                                 {"column": second["column"], "input_ref": first["input_ref"]}])
        chosen, traces, workspace, model = self.prepare(["Concentrate has 4% silica and 62% iron."], swapped)
        self.assertEqual(chosen["status"], "clarify")
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(traces[0]["validation_errors"])
        self.assertEqual(execute_process_request(workspace, chosen)[0]["status"], "clarify")

    def test_requested_other_column_statistics_do_not_repeat_old_readings(self):
        messages = ["Concentrate has 4% silica and 62% iron.", "Show the mean and median of Starch Flow."]
        chosen, _, workspace, model = self.prepare(messages, plan(distribution=[FLOW]))
        answer, traces = execute_process_request(workspace, chosen)
        self.assertEqual(answer["status"], "ok")
        self.assertEqual(chosen["compare"], [])
        self.assertEqual(len(model.calls), 1)
        rows = [entry for entry in traces if entry["operation"] == "adaptive_tool"][0]["result"]["data"]["distributions"]
        self.assertEqual([row["column"] for row in rows], [FLOW])
        self.assertIn("250", answer["data"]["summary"])
        self.assertNotIn("supplied 62%", answer["data"]["summary"])

    def test_wrong_distribution_scope_is_rejected_even_when_column_exists(self):
        chosen, traces, _, _ = self.prepare(["Show the historical mean of feed iron."], plan(distribution=[IRON]))
        self.assertEqual(chosen["status"], "clarify")
        self.assertTrue(traces[0]["validation_errors"])

    def test_count_and_or_execute_exact_literal_predicates(self):
        for combine in ("and", "or"):
            message = f"How many hours had Starch Flow above 150 {combine} concentrate silica below 5%?"
            def output(payload):
                pairs = {entry["column"]: entry for entry in payload["binding_candidates"]}
                return plan(count=[{"conditions": [{**pairs[FLOW], "operator": "gt"},
                                                    {**pairs[SILICA], "operator": "lt"}], "combine": combine}])
            chosen, _, workspace, model = self.prepare([message], output)
            answer, traces = execute_process_request(workspace, chosen)
            self.assertEqual(chosen["status"], "ready")
            self.assertEqual(answer["status"], "ok")
            result = [entry for entry in traces if entry["operation"] == "adaptive_tool"][0]["result"]
            count = workspace.facts[result["data"]["facts"]["count"]]["value"]
            self.assertEqual(count, 2 if combine == "and" else 3)
            self.assertEqual([{key: row[key] for key in ("column", "input_ref", "operator")}
                              for row in result["data"]["conditions"]], chosen["count"][0]["conditions"])
            self.assertEqual(len(model.calls), 1)

    def test_wrong_predicate_direction_and_missing_current_measurement_reject(self):
        def wrong(payload):
            return plan(count=[{"conditions": [{**payload["binding_candidates"][0], "operator": "lt"}], "combine": "and"}])
        chosen, _, _, _ = self.prepare(["How often was Starch Flow above 150?"], wrong)
        self.assertEqual(chosen["status"], "clarify")

    def test_inclusive_predicate_cannot_be_replaced_with_strict_opposite(self):
        def output(payload):
            return plan(count=[{"conditions": [{**payload["binding_candidates"][0], "operator": "ge"}], "combine": "and"}])
        chosen, _, workspace, _ = self.prepare(["How often was Starch Flow at least 200?"], output)
        self.assertEqual(chosen["status"], "ready")
        self.assertEqual(execute_process_request(workspace, chosen)[0]["status"], "ok")
        for phrase, incorrect in [("no more than", "gt"), ("no less than", "lt")]:
            chosen, _, _, _ = self.prepare([f"How often was Starch Flow {phrase} 200?"],
                lambda payload: plan(count=[{"conditions": [{**payload["binding_candidates"][0], "operator": incorrect}], "combine": "and"}]))
            self.assertEqual(chosen["status"], "clarify")

    def test_frequency_followup_uses_its_labeled_directions_and_prior_readings(self):
        messages = ["Concentrate has 4% silica and 62% iron.", "What fraction had more iron and less silica?"]
        def output(payload):
            pairs = {entry["column"]: entry for entry in payload["binding_candidates"]}
            return plan(count=[{"conditions": [{**pairs[IRON], "operator": "gt"},
                                                {**pairs[SILICA], "operator": "lt"}], "combine": "and"}])
        chosen, _, workspace, _ = self.prepare(messages, output)
        self.assertEqual(chosen["status"], "ready")
        answer, _ = execute_process_request(workspace, chosen)
        self.assertEqual(answer["status"], "ok")
        self.assertIn("1 of 4", answer["data"]["summary"])

    def test_negated_numeric_direction_is_not_silently_used_as_positive_predicate(self):
        for text in ("How often was Starch Flow not above 200?", "Count hours without Starch Flow above 200."):
            chosen, _, _, _ = self.prepare([text], lambda payload: plan(count=[{
                "conditions": [{**payload["binding_candidates"][0], "operator": "gt"}], "combine": "and"}]))
            self.assertEqual(chosen["status"], "clarify")

    def test_non_percent_physical_units_cannot_be_assumed_from_unitless_header(self):
        chosen, _, _, model = self.prepare(["Starch Flow is 250 L/min."], plan(describe=True))
        self.assertEqual(chosen["status"], "clarify")
        self.assertEqual(len(model.calls), 0)
        chosen, _, _, _ = self.prepare(["Concentrate has 4% silica and 62% iron."],
                                      lambda payload: plan(compare=payload["binding_candidates"][:1]))
        self.assertEqual(chosen["status"], "clarify")

    def test_executor_revalidates_tampered_plan_and_exact_literal_values(self):
        chosen, _, workspace, _ = self.prepare(["Concentrate iron is 62%."],
                                              lambda payload: plan(compare=payload["binding_candidates"]))
        chosen["compare"][0]["column"] = FEED
        answer, _ = execute_process_request(workspace, chosen)
        self.assertEqual(answer["status"], "insufficient_evidence")
        chosen, _, workspace, _ = self.prepare(["Concentrate iron is 62%."],
                                              lambda payload: plan(compare=payload["binding_candidates"]))
        chosen["_request_context"]["inputs"][0]["value"] = 63
        answer, _ = execute_process_request(workspace, chosen)
        self.assertEqual(answer["status"], "insufficient_evidence")

    def test_describe_needs_no_numerical_inputs_or_answer_generation(self):
        chosen, _, workspace, model = self.prepare(["Which columns and units are available?"], plan(describe=True))
        answer, traces = execute_process_request(workspace, chosen)
        self.assertEqual(answer["status"], "ok")
        self.assertIn(FLOW, answer["data"]["summary"])
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(traces[-1]["numerical_generation"])

    def test_time_subsets_decline_without_substituting_full_population_statistics(self):
        for message in ["Show the mean concentrate silica for June only.",
                        "Show the mean concentrate silica in May.",
                        "Compare concentrate silica of 4% during 2024-01-01 only.",
                        "Count hours with Starch Flow above 200 in the last 7 days.",
                        "Show the median Starch Flow between 08:00 and 12:00."]:
            with self.subTest(message=message):
                chosen, traces, workspace, model = self.prepare([message], plan(distribution=[SILICA]))
                self.assertEqual(chosen["status"], "insufficient_evidence")
                self.assertEqual(model.calls, [])
                self.assertFalse(traces[0]["model_called"])
                answer, execution = execute_process_request(workspace, chosen)
                self.assertEqual(answer["status"], "insufficient_evidence")
                self.assertIn("Time-filtered process calculations are not supported", answer["data"]["summary"])
                self.assertEqual(answer["data"]["operations_executed"], 0)
                self.assertFalse(any(entry["operation"] == "adaptive_tool" for entry in execution))
                self.assertFalse(any(fact.get("operation") == "distribution" for fact in workspace.facts.values()))

    def test_unsupported_percentiles_cannot_be_answered_by_a_quartile_table(self):
        for message in ["What is the 90th percentile of concentrate silica?",
                        "Show percentile 90 for Starch Flow.",
                        "Show the ninth decile of concentrate silica.",
                        "What is P90 for Starch Flow?"]:
            with self.subTest(message=message):
                chosen, _, workspace, model = self.prepare([message], plan(distribution=[SILICA]))
                answer, _ = execute_process_request(workspace, chosen)
                self.assertEqual(model.calls, [])
                self.assertEqual(answer["status"], "insufficient_evidence")
                self.assertIn("Requested percentiles or quantiles are not supported", answer["data"]["summary"])
                self.assertEqual(answer["data"]["operations_executed"], 0)
                self.assertNotIn("| Column |", answer["data"]["summary"])

    def test_executor_rechecks_unsupported_scope_even_if_ready_plan_is_injected(self):
        chosen, _, workspace, _ = self.prepare(["Show the June mean concentrate silica."], plan(distribution=[SILICA]))
        chosen.update(plan(distribution=[SILICA]))
        answer, _ = execute_process_request(workspace, chosen)
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertEqual(answer["data"]["operations_executed"], 0)
        self.assertEqual(answer["data"]["analysis_plan"]["required_operations"], [])

    def test_full_period_statistics_and_polite_may_are_still_supported(self):
        chosen, _, workspace, model = self.prepare(["May I see the mean and quartiles of Starch Flow for the full recorded period?"],
                                                  plan(distribution=[FLOW], describe=True))
        answer, _ = execute_process_request(workspace, chosen)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(answer["status"], "ok")
        self.assertIn("| Starch Flow | not stated | 4 |", answer["data"]["summary"])
        self.assertIn("Recorded period: 2024-01-01 00:00:00 to 2024-01-01 03:00:00", answer["data"]["summary"])

    def test_requested_recorded_period_and_gaps_are_rendered_without_column_inventory(self):
        chosen, _, workspace, model = self.prepare(["What period is recorded, and are there gaps?"], plan(describe=True))
        answer, _ = execute_process_request(workspace, chosen)
        self.assertEqual(chosen["status"], "ready")
        self.assertEqual(len(model.calls), 1)
        self.assertIn("4 distinct valid hourly observations", answer["data"]["summary"])
        self.assertIn("no missing hourly timestamps", answer["data"]["summary"])
        self.assertNotIn("Available CSV columns", answer["data"]["summary"])

    def test_normal_assessment_schema_cannot_add_frequency_inventory_or_distribution(self):
        chosen, _, _, model = self.prepare(["Concentrate has 4% silica and 62% iron. Are these normal?"],
            lambda payload: plan(compare=payload["binding_candidates"]))
        self.assertEqual(chosen["status"], "ready")
        properties = model.calls[0][2]["properties"]
        self.assertEqual(properties["count"]["maxItems"], 0)
        self.assertEqual(properties["distribution"]["maxItems"], 0)
        self.assertEqual(properties["describe"]["enum"], [False])

    def test_frequency_schema_exposes_only_supported_column_operator_reference(self):
        def output(payload):
            available = payload["available_operations"]
            self.assertEqual(available["count_predicates"], [{"column": FLOW, "input_ref": "U001", "operator": "gt"}])
            return plan(count=[{"conditions": available["count_predicates"], "combine": "and"}])
        chosen, _, _, model = self.prepare(["How often was Starch Flow above 200?"], output)
        self.assertEqual(chosen["status"], "ready")
        properties = model.calls[0][2]["properties"]
        self.assertEqual(properties["compare"]["maxItems"], 0)
        predicate = properties["count"]["items"]["properties"]["conditions"]["items"]["oneOf"][0]
        self.assertEqual(predicate["properties"]["operator"]["enum"], ["gt"])


if __name__ == "__main__":
    unittest.main()
