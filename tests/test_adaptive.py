"""Model-selected operations and literal-input boundaries, using synthetic data."""
import csv
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from metso_assistant.adaptive import AdaptiveRunner, request_facts, active_duration_inputs

class OperandTests(unittest.TestCase):
    def test_signed_literals_and_word_numbers_preserve_value(self):
        result = request_facts(["Concentrate silica is -3%, iron is +80%, and another reading is −4%."], "flotation")
        self.assertEqual([x["value"] for x in result], [Decimal(-3), Decimal(80), Decimal(-4)])
        self.assertEqual(request_facts(["We need negative five cables."], "service_cost")[0]["value"], -5)

    def test_literal_package_units_remain_distinct_and_model_numbers_are_not_counts(self):
        result = request_facts(["For VSP-750, price 4 pairs of gloves, 2 tubes of grease and 300 ml of lubricant."], "service_cost")
        self.assertEqual([(f["value"], f["unit"]) for f in result], [(4, "pair"), (2, "tube"), (300, "ml")])

    def test_ambiguous_trip_is_not_an_hour_operand_but_each_way_is_converted(self):
        ambiguous = request_facts(["One trip takes 6 hours."], "service_cost")
        self.assertFalse(any(f["unit"] == "hour" for f in ambiguous))
        explicit = request_facts(["Travel takes 6 hours each way."], "service_cost")
        self.assertEqual([f["value"] for f in explicit if f["unit"] == "hour"], [12])

    def test_date_is_not_a_requested_quantity(self):
        result = request_facts(["On 2025-08-12 we need three replacement valves."], "service_cost")
        self.assertEqual([x["value"] for x in result], [3])

    def test_grouped_counts_and_explicit_labour_hours(self):
        result = request_facts(["Price 1,000 cables. Labour: 4 hours, travel 6 hours."], "service_cost")
        self.assertEqual(result[0]["value"], 1000)
        self.assertEqual((result[1]["unit"], result[1]["scope"]["role"]), ("hour", "labour_hours"))
        self.assertEqual(result[2]["scope"]["role"], "ambiguous_duration")

    def test_explicit_unknown_travel_retires_prior_duration_operands(self):
        result = request_facts(["Round-trip travel is 4 hours.", "I do not know travel hours after all."], "service_cost", {"round_trip_travel_hours": None})
        self.assertFalse(any(f["unit"] == "hour" for f in result))
        self.assertEqual(len({f["id"] for f in result}), len(result))

    def test_analytical_counts_and_column_identifiers_are_not_operands(self):
        result = request_facts(["Give the middle 50% for Flotation Column 03 Air Flow."], "flotation")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["scope"]["role"], "analytical_parameter")
        result = request_facts(["Combine these into one USD total."], "service_cost")
        self.assertEqual(result[0]["unit"], "number")
        self.assertEqual(result[0]["scope"]["role"], "analytical_parameter")


class DurationInputTests(unittest.TestCase):
    def test_supplied_duration_uses_latest_billing_role_and_requested_categories(self):
        facts = {
            "U1": {"id": "U1", "kind": "user_input", "unit": "hour", "value": 4, "scope": {"role": "travel_hours", "turn": 0}},
            "U2": {"id": "U2", "kind": "user_input", "unit": "hour", "value": 6, "scope": {"role": "travel_hours", "turn": 1}},
            "U3": {"id": "U3", "kind": "user_input", "unit": "hour", "value": 3, "scope": {"role": "labour_hours", "turn": 0}},
        }
        self.assertEqual({fact["id"] for fact in active_duration_inputs(facts)}, {"U2", "U3"})
        self.assertEqual([fact["id"] for fact in active_duration_inputs(facts, ["travel"])], ["U2"])
        self.assertEqual(active_duration_inputs(facts, ["material"]), [])


class FakeModel:
    def __init__(self, outputs): self.outputs, self.calls = iter(outputs), []
    def invoke(self, *args, **kwargs):
        self.calls.append(args)
        return next(self.outputs), {"synthetic": True}


class PlannerTests(unittest.TestCase):
    def run_plan(self, outputs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "% Iron Concentrate", "% Silica Concentrate"])
                writer.writerows([["2021-02-03 00:00:00", 50, 6], ["2021-02-03 01:00:00", 55, 5], ["2021-02-03 02:00:00", 60, 4], ["2021-02-03 03:00:00", 65, 3]])
            model = FakeModel(outputs)
            result, traces = AdaptiveRunner(model, store_path="unused", csv_path=path).run("flotation", ["Concentrate iron is 80%."])
            return result, traces, model

    def test_model_chooses_operation_and_code_presents_computed_relationship(self):
        result, traces, model = self.run_plan([{"status": "ready", "question": None,
            "compare": [{"column": "% Iron Concentrate", "input_ref": "U001"}], "distribution": [], "count": [], "describe": False}])
        self.assertEqual(result["status"], "ok")
        self.assertIn("80%", result["data"]["summary"])
        self.assertIn("65%", result["data"]["summary"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(traces[-1]["operation"], "deterministic_evidence_rendering")
        self.assertFalse(traces[-1]["numerical_generation"])

    def test_invalid_interpretation_clarifies_without_replanning_or_calculation(self):
        result, traces, model = self.run_plan([{"action": []}])
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["data"]["operations_executed"], 0)


if __name__ == "__main__":
    unittest.main()
