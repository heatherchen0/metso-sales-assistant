"""Synthetic intent-selection contracts; no source data or model execution."""
from copy import deepcopy
from decimal import Decimal
import json
import unittest

from metso_assistant.analysis_plan import select_analysis
from metso_assistant.model import ModelError


class MockModel:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def invoke(self, system, messages, schema, **options):
        self.calls.append((system, deepcopy(messages), deepcopy(schema), options))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output, {"synthetic": True}


def literal(identifier, message, token, value, entity, *, unit="each", role="requested_quantity", turn=0):
    """Synthetic parser-shaped input with an exact original-message span."""
    start = message.index(token)
    return {"id": identifier, "label": token + " " + entity, "value": Decimal(str(value)),
            "unit": unit, "kind": "user_input", "source_ids": [], "parents": [],
            "user_text": message, "scope": {"entity_text": entity, "role": role,
                                             "turn": turn, "span": [start, start + len(token)]}}


class AnalysisPlanTests(unittest.TestCase):
    def test_source_values_and_assistant_answers_are_not_sent(self):
        selected = {"purpose": "requested_work", "requested_categories": ["labour"], "quantity_structure": "direct", "equipment_count_ref": None}
        model = MockModel([selected])
        messages = [{"role": "user", "content": "What labour should I budget?"},
                    {"role": "assistant", "content": "Private invented amount SECRET"},
                    {"role": "tool", "content": "Source values SECRET"},
                    {"role": "user", "content": "I do not know the hours yet."}]
        inputs = [{"id": "R1", "label": "SECRET SOURCE RATE", "value": 987654, "unit": "money/hour", "kind": "hourly_rate"},
                  {"id": "SP", "label": "SECRET SOURCE-DERIVED INPUT", "value": 987654, "unit": "hour", "kind": "user_input", "source_ids": ["S1"]},
                  {"id": "CP", "label": "SECRET COMPUTED INPUT", "value": 987654, "unit": "hour", "kind": "user_input", "parents": ["U1"]}]
        select_analysis(model, "service_cost", messages, inputs=inputs)
        payload = json.loads(model.calls[0][1][0]["content"])
        self.assertEqual(payload["inputs"], [])
        self.assertEqual(payload["user_messages"], [messages[0]["content"], messages[-1]["content"]])
        self.assertNotIn("SECRET", str(model.calls))
        self.assertNotIn("987654", str(model.calls))
        self.assertEqual(payload["current_question"], "I do not know the hours yet.")

    def test_only_last_six_actual_user_messages_are_in_context(self):
        model = MockModel([{"purpose": "requested_work", "requested_categories": ["labour"], "quantity_structure": "direct", "equipment_count_ref": None}])
        messages = [{"role": "user", "content": f"request {i}"} for i in range(9)]
        messages.insert(5, {"role": "assistant", "content": "old material result"})
        messages[-1]["content"] = "Add labour; I do not know the hours."
        plan, _ = select_analysis(model, "service_cost", messages)
        payload = json.loads(model.calls[0][1][0]["content"])
        self.assertEqual(len(payload["user_messages"]), 6)
        self.assertEqual(payload["user_messages"][0], "request 3")
        self.assertEqual(plan["requested_categories"], ["labour"])
        self.assertIn("Missing hours do not", model.calls[0][0])

    def test_recorded_invoice_conversion_remains_historical(self):
        selected = {"purpose": "original_visit_totals", "requested_categories": ["material", "labour", "travel"], "quantity_structure": "direct", "equipment_count_ref": None}
        model = MockModel([selected])
        plan, _ = select_analysis(model, "service_cost", ["Compare recorded visits in Asterland and Borealia, then express them in one currency."])
        self.assertEqual(plan, selected)
        self.assertNotIn("columns", model.calls[0][2]["properties"])

    def test_category_restricted_historical_subtotal_repairs_purpose_without_expanding_scope(self):
        message = "Give a historical materials subtotal for the requested replacement guards. Exclude labour and travel."
        good = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "direct", "equipment_count_ref": None}
        bad = dict(good, purpose="original_visit_totals")
        model = MockModel([bad, good])
        plan, traces = select_analysis(model, "service_cost", [message])
        self.assertEqual(plan, good)
        self.assertEqual(plan["requested_categories"], ["material"])
        self.assertTrue(any("category-restricted subtotal requires requested_work" in error for error in traces[0]["validation_errors"]))
        repair = model.calls[1][1][-1]["content"]
        self.assertIn("Preserve the user's explicit exclusions", repair)
        self.assertIn("do not add excluded categories", repair)

    def test_original_invoice_scope_requires_every_category_independent_of_array_order(self):
        selected = {"purpose": "original_visit_totals", "requested_categories": ["travel", "material", "labour"], "quantity_structure": "direct", "equipment_count_ref": None}
        plan, traces = select_analysis(MockModel([selected]), "service_cost", ["Compare the reports' original whole-visit invoices without repricing components."])
        self.assertEqual(plan, selected)
        self.assertFalse(traces[0]["validation_errors"])

    def test_original_invoice_cannot_keep_a_category_restriction_after_repair(self):
        for categories in (["material"], ["labour"], ["travel"], ["material", "travel"]):
            bad = {"purpose": "original_visit_totals", "requested_categories": categories, "quantity_structure": "direct", "equipment_count_ref": None}
            with self.subTest(categories=categories):
                with self.assertRaises(ModelError) as caught:
                    select_analysis(MockModel([bad, bad]), "service_cost", ["Give only selected service components, with the other categories excluded."])
                self.assertTrue(any("unrestricted original whole-invoice" in error for error in caught.exception.traces[-1]["validation_errors"]))

    def test_unknown_category_and_extra_constants_require_repair(self):
        good = {"purpose": "requested_work", "requested_categories": ["labour"], "quantity_structure": "direct", "equipment_count_ref": None}
        bad = dict(good, requested_categories=["installation"], hourly_rate=1234)
        model = MockModel([bad, good])
        plan, trace = select_analysis(model, "service_cost", ["Price on-site labour only."])
        self.assertEqual(plan, good)
        self.assertEqual(len(trace), 2)
        self.assertTrue(trace[0]["validation_errors"])
        self.assertTrue(any("keys must be exactly" in error for error in trace[0]["validation_errors"]))
        self.assertIn("requested_categories contains an unavailable value", trace[0]["validation_errors"])
        self.assertIn("Correct the analysis plan", model.calls[1][1][-1]["content"])

    def test_invalid_service_shape_does_not_fall_back_to_materials(self):
        for bad in ({"purpose": "requested_work", "requested_categories": [], "quantity_structure": "direct", "equipment_count_ref": None},
                    {"purpose": "prediction", "requested_categories": ["labour"], "quantity_structure": "direct", "equipment_count_ref": None},
                    {"purpose": "requested_work", "requested_categories": ["labour", "labour"], "quantity_structure": "direct", "equipment_count_ref": None},
                    {"purpose": "requested_work", "requested_categories": "travel", "quantity_structure": "direct", "equipment_count_ref": None},
                    {"purpose": "requested_work", "requested_categories": ["installation"], "quantity_structure": "direct", "equipment_count_ref": None},
                    ["requested_work"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ModelError):
                    select_analysis(MockModel([bad, bad]), "service_cost", ["Add billed travel."])

    def test_service_purpose_names_and_descriptions_separate_target_from_provenance(self):
        expected = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "direct", "equipment_count_ref": None}
        messages = ["Give a historical cost example for 6 replacement guards. Materials only; exclude labour and travel.",
                    "Add 2 retaining kits. How does the subtotal change?"]
        model = MockModel([dict(expected, purpose="historical_visits"), expected])
        plan, traces = select_analysis(model, "service_cost", messages)
        self.assertEqual(plan, expected)
        self.assertIn("purpose is unavailable", traces[0]["validation_errors"])
        schema = model.calls[0][2]
        self.assertEqual(schema["properties"]["purpose"]["enum"], ["requested_work", "original_visit_totals"])
        self.assertIn("not evidence age", schema["properties"]["purpose"]["description"])
        self.assertIn("exclusions", schema["properties"]["requested_categories"]["description"])
        self.assertIn("Missing hours do not", model.calls[0][0])
        self.assertNotIn("Choose required process operations", model.calls[0][0])

    def test_supply_and_duration_reply_retains_active_visit_scope_and_owned_material_flag(self):
        messages = ["We need 6 guards replaced. What should we budget for the service visit?",
                    "The customer already supplies the guards. Round-trip travel is 4 hours."]
        material = literal("U1", messages[0], "6", 6, "guards", turn=0)
        material["scope"]["customer_supplied"] = True
        travel = literal("U2", messages[1], "4", 4, "travel", unit="hour", role="travel_hours", turn=1)
        selected = {"purpose": "requested_work", "requested_categories": ["labour", "travel"], "quantity_structure": "direct", "equipment_count_ref": None}
        model = MockModel([selected])
        plan, _ = select_analysis(model, "service_cost", messages, inputs=[material, travel])
        payload = json.loads(model.calls[0][1][0]["content"])
        self.assertEqual(plan["requested_categories"], ["labour", "travel"])
        self.assertIs(payload["inputs"][0]["scope"]["customer_supplied"], True)
        self.assertEqual(payload["user_messages"], messages)
        self.assertIn("whole service-visit budget", model.calls[0][0])
        self.assertIn("continues that active visit scope", model.calls[0][0])
        description = model.calls[0][2]["properties"]["requested_categories"]["description"]
        self.assertIn("Whole visit includes on-site labour and billed travel", description)
        self.assertIn("not customer_supplied items", description)

    def test_literal_customer_supply_metadata_is_boolean_only_and_source_free(self):
        message = "Price 3 guards and 2 fastener kits."
        guards = literal("U1", message, "3", 3, "guards")
        guards["scope"]["customer_supplied"] = False
        kits = literal("U2", message, "2", 2, "fastener kits")
        kits["scope"]["customer_supplied"] = "true"
        source = {"id": "R1", "label": "PRIVATE MATERIAL SOURCE", "value": 123456, "unit": "money/each", "kind": "unit_price",
                  "scope": {"customer_supplied": True}}
        selected = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "direct", "equipment_count_ref": None}
        model = MockModel([selected])
        select_analysis(model, "service_cost", [message], inputs=[guards, kits, source])
        payload = json.loads(model.calls[0][1][0]["content"])
        self.assertIs(payload["inputs"][0]["scope"]["customer_supplied"], False)
        self.assertNotIn("customer_supplied", payload["inputs"][1]["scope"])
        self.assertNotIn("PRIVATE MATERIAL SOURCE", str(model.calls))
        self.assertNotIn("123456", str(model.calls))
        self.assertIn("any other", model.calls[0][0])
        self.assertIn("Equipment ownership alone", model.calls[0][0])

    def test_model_decode_failure_is_retried_once_with_retained_trace(self):
        failure = ModelError("malformed JSON")
        model = MockModel([failure, {"purpose": "requested_work", "requested_categories": ["travel"], "quantity_structure": "direct", "equipment_count_ref": None}])
        plan, traces = select_analysis(model, "service_cost", ["What is billed travel?"])
        self.assertEqual(plan["requested_categories"], ["travel"])
        self.assertEqual(traces[0]["validation_errors"], ["malformed JSON"])
        self.assertEqual(len(model.calls), 2)

    def test_per_equipment_structure_is_a_declared_relation_without_arithmetic(self):
        selected = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "per_equipment", "equipment_count_ref": "U1"}
        message = "We have 4 machines and need 2 seals per machine."
        inputs = [literal("U1", message, "4", 4, "machines"), literal("U2", message, "2", 2, "seals")]
        model = MockModel([selected])
        plan, _ = select_analysis(model, "service_cost", [message], inputs=inputs)
        self.assertEqual(plan, selected)
        self.assertEqual(set(plan), {"purpose", "requested_categories", "quantity_structure", "equipment_count_ref"})
        self.assertIn("actual equipment-count", model.calls[0][0])
        self.assertIn("semantic choice", model.calls[0][0])
        self.assertEqual(model.calls[0][2]["properties"]["quantity_structure"]["enum"], ["direct", "per_equipment"])
        self.assertEqual(plan["equipment_count_ref"], "U1")

    def test_equipment_reference_context_contains_only_literal_user_fields(self):
        message = "We have 4 machines and need 1 component per machine. Installation is 2 hours."
        inputs = [literal("U1", message, "4", 4, "machines"),
                  literal("U2", message, "1", 1, "component"),
                  literal("UH", message, "2", 2, "installation", unit="hour", role="labour_hours"),
                  {"id": "R1", "label": "PRIVATE SOURCE RATE", "value": 765432, "unit": "money/each", "kind": "unit_price"},
                  {"id": "C1", "label": "PRIVATE COMPUTED AMOUNT", "value": 765432, "unit": "money", "kind": "calculation"}]
        inputs[0]["extra"] = "do not send"
        inputs[0]["scope"]["secret"] = "do not send"
        selected = {"purpose": "requested_work", "requested_categories": ["material"],
                    "quantity_structure": "per_equipment", "equipment_count_ref": "U1"}
        model = MockModel([selected])
        select_analysis(model, "service_cost", [message], inputs=inputs)
        payload = json.loads(model.calls[0][1][0]["content"])
        self.assertEqual(payload["inputs"][0], {"id": "U1", "label": "4 machines", "value": "4", "unit": "each",
                                               "scope": {"entity_text": "machines", "turn": 0}})
        self.assertNotIn("765432", str(model.calls))
        self.assertNotIn("PRIVATE", str(model.calls))
        self.assertNotIn("do not send", str(model.calls))
        reference = model.calls[0][2]["properties"]["equipment_count_ref"]["anyOf"]
        self.assertEqual(reference, [{"type": "null"}, {"type": "string", "enum": ["U1"]}])
        self.assertEqual(payload["proven_quantity_relations"], [{"per_item_ref": "U2", "equipment_count_ref": "U1"}])

    def test_unknown_machine_count_remains_unresolved_without_inventing_a_reference(self):
        selected = {"purpose": "requested_work", "requested_categories": ["labour"],
                    "quantity_structure": "per_equipment", "equipment_count_ref": None}
        model = MockModel([selected])
        plan, _ = select_analysis(model, "service_cost", ["There are several machines; allow installation time per machine, but their count is unknown."])
        self.assertIsNone(plan["equipment_count_ref"])
        self.assertEqual(model.calls[0][2]["properties"]["equipment_count_ref"]["anyOf"], [{"type": "null"}])

    def test_spurious_repetition_is_rejected_and_unproven_count_ids_are_unavailable(self):
        message = "Price 4 guards in total."
        inputs = [literal("U1", message, "4", 4, "guards")]
        bad = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "per_equipment", "equipment_count_ref": None}
        good = dict(bad, quantity_structure="direct")
        model = MockModel([bad, good])
        plan, traces = select_analysis(model, "service_cost", [message], inputs=inputs)
        self.assertEqual(plan, good)
        self.assertTrue(any("repetition relationship" in error for error in traces[0]["validation_errors"]))
        self.assertEqual(model.calls[0][2]["properties"]["quantity_structure"]["enum"], ["direct"])
        self.assertEqual(model.calls[0][2]["properties"]["equipment_count_ref"]["anyOf"], [{"type": "null"}])

    def test_per_item_id_cannot_replace_proven_equipment_count_anchor(self):
        message = "We have 4 machines and need 1 guard per machine."
        inputs = [literal("U1", message, "4", 4, "machines"), literal("U2", message, "1", 1, "guard")]
        good = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "per_equipment", "equipment_count_ref": "U1"}
        bad = dict(good, equipment_count_ref="U2")
        plan, traces = select_analysis(MockModel([bad, good]), "service_cost", [message], inputs=inputs)
        self.assertEqual(plan, good)
        self.assertTrue(any("eligible actual user-input" in error for error in traces[0]["validation_errors"]))

    def test_reference_must_be_a_compatible_literal_input_and_direct_must_be_null(self):
        inputs = [{"id": "U1", "label": "machines", "value": 3, "unit": "each", "kind": "user_input", "scope": {"role": "requested_quantity"}},
                  {"id": "UH", "label": "hours", "value": 2, "unit": "hour", "kind": "user_input", "scope": {"role": "labour_hours"}},
                  {"id": "R1", "label": "source rate", "value": 500, "unit": "money/each", "kind": "unit_price"},
                  {"id": "SP", "label": "source-derived pretended input", "value": 3, "unit": "each", "kind": "user_input", "source_ids": ["S1"], "scope": {"role": "requested_quantity"}},
                  {"id": "CP", "label": "derived pretended input", "value": 3, "unit": "each", "kind": "user_input", "parents": ["U1"], "scope": {"role": "requested_quantity"}}]
        base = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "per_equipment"}
        for reference in ("unknown", "UH", "R1", "SP", "CP", 3, True, [], {}):
            with self.subTest(reference=reference):
                bad = dict(base, equipment_count_ref=reference)
                with self.assertRaises(ModelError):
                    select_analysis(MockModel([bad, bad]), "service_cost", ["Price per machine."], inputs=inputs)
        bad = dict(base, quantity_structure="direct", equipment_count_ref="U1")
        with self.assertRaises(ModelError) as caught:
            select_analysis(MockModel([bad, bad]), "service_cost", ["Price three components in total."], inputs=inputs)
        self.assertTrue(any("direct quantities" in error for error in caught.exception.traces[0]["validation_errors"]))

    def test_missing_equipment_reference_key_and_invalid_literal_metadata_fail_closed(self):
        bad = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "direct"}
        with self.assertRaises(ModelError):
            select_analysis(MockModel([bad, bad]), "service_cost", ["Price the materials."])
        model = MockModel([])
        for value in (True, Decimal("NaN"), float("inf"), {"invented": 5}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    select_analysis(model, "service_cost", ["Price the materials."], inputs=[{"id": "U1", "label": "quantity", "unit": "each", "kind": "user_input", "value": value}])
        self.assertEqual(model.calls, [])

    def test_direct_quantity_structure_preserves_total_or_unknown_basis(self):
        selected = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "direct", "equipment_count_ref": None}
        for question in ("Price eight seals in total for the machines.", "I do not know how many seals are needed."):
            with self.subTest(question=question):
                plan, _ = select_analysis(MockModel([selected]), "service_cost", [question])
                self.assertEqual(plan["quantity_structure"], "direct")

    def test_invalid_or_missing_quantity_structure_requires_repair(self):
        good = {"purpose": "requested_work", "requested_categories": ["material"], "quantity_structure": "direct", "equipment_count_ref": None}
        for value in (None, [], True, "per_part", "", {"relation": "per_equipment"}):
            bad = dict(good, quantity_structure=value)
            with self.subTest(value=value):
                plan, traces = select_analysis(MockModel([bad, good]), "service_cost", ["Price the requested materials."])
                self.assertEqual(plan, good)
                self.assertIn("quantity_structure", traces[0]["validation_errors"][0])
        bad = {key: value for key, value in good.items() if key != "quantity_structure"}
        with self.assertRaises(ModelError) as caught:
            select_analysis(MockModel([bad, bad]), "service_cost", ["Price the requested materials."])
        self.assertTrue(any("quantity_structure" in error for error in caught.exception.traces[0]["validation_errors"]))

    def test_bad_caller_inputs_do_not_invoke_model(self):
        model = MockModel([])
        for route, messages in [("product", ["help"]), ("flotation", ["help"]),
                                ("service_cost", []), ("service_cost", [{"role": "assistant", "content": "invented request"}]),
                                ("service_cost", [{"role": "user", "content": 123}])]:
            with self.subTest(route=route, messages=messages):
                with self.assertRaises(ValueError):
                    select_analysis(model, route, messages)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
