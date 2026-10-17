"""Explicit user ownership annotations do not invent procurement assumptions."""
from copy import deepcopy
from decimal import Decimal
import unittest

from metso_assistant.request_scope import annotate_material_supply, annotate_quantity_supersession


def operands():
    return [
        {"id": "U001", "kind": "user_input", "value": Decimal(6), "unit": "each", "label": "Broad irrelevant label",
         "user_text": "Replace 6 locking pins and 2 seal kits.", "scope": {"span": [8, 9], "entity_text": "locking pins"}},
        {"id": "U002", "kind": "user_input", "value": Decimal(2), "unit": "kit",
         "user_text": "Replace 6 locking pins and 2 seal kits.", "scope": {"span": [27, 28], "entity_text": "seal kits"}},
    ]


class RequestScopeTests(unittest.TestCase):
    def test_explicit_customer_ownership_binds_only_matching_literal_material_and_keeps_provenance(self):
        inputs = operands(); before = deepcopy(inputs)
        messages = ["Replace 6 locking pins and 2 seal kits.", "The customer already has the locking pins."]
        result = annotate_material_supply(inputs, messages)
        self.assertIs(result[0]["scope"]["customer_supplied"], True)
        self.assertEqual(result[0]["scope"]["supply_provenance"], {"turn": 1, "quote": "The customer already has the locking pins"})
        self.assertNotIn("customer_supplied", result[1]["scope"])
        self.assertEqual(inputs, before)

    def test_explicit_we_client_and_predicate_phrases_are_supported_without_part_aliases(self):
        for message in ("We already have the locking pins.", "The client supplies locking pins.",
                        "The locking pins are customer-supplied.", "Locking pins are client supplied."):
            with self.subTest(message=message):
                result = annotate_material_supply(operands(), [message])
                self.assertIs(result[0]["scope"]["customer_supplied"], True)

    def test_negated_ownership_and_later_service_procurement_reverse_previous_ownership(self):
        for message in ("The customer does not have locking pins.", "The customer doesn't own locking pins.",
                        "The locking pins are not customer-supplied.", "Please supply the locking pins.",
                        "The technician will supply locking pins."):
            with self.subTest(message=message):
                result = annotate_material_supply(operands(), ["The customer already has locking pins.", message])
                self.assertIs(result[0]["scope"]["customer_supplied"], False)
                self.assertEqual(result[0]["scope"]["supply_provenance"]["turn"], 1)

    def test_uncertainty_partial_stock_questions_and_unrelated_materials_establish_no_ownership(self):
        for message in ("If the customer already has locking pins, what happens?",
                        "I don't know whether the customer has locking pins.",
                        "Does the customer have locking pins?", "The customer has some locking pins.",
                        "The customer has 3 locking pins.", "The customer already has spare rotors.",
                        "The technician will not supply locking pins."):
            with self.subTest(message=message):
                result = annotate_material_supply(operands(), [message])
                self.assertNotIn("customer_supplied", result[0]["scope"])

    def test_exact_stated_stock_count_binds_but_unbound_or_source_facts_cannot_gain_flags(self):
        result = annotate_material_supply(operands(), ["The customer already has 6 locking pins."])
        self.assertIs(result[0]["scope"]["customer_supplied"], True)
        for change in ({"kind": "quantity"}, {"unit": "hour"}, {"user_text": "6", "scope": {}},
                       {"user_text": "Replace 6 rotors", "scope": {}}):
            inputs = operands(); inputs[0].update(change)
            self.assertNotIn("customer_supplied", annotate_material_supply(inputs, ["We own locking pins."])[0]["scope"])

    def test_unproven_existing_flags_are_not_trusted_and_silence_does_not_overwrite_an_explicit_statement(self):
        inputs = operands(); inputs[0]["scope"]["customer_supplied"] = True
        self.assertNotIn("customer_supplied", annotate_material_supply(inputs, ["Travel will take four hours."])[0]["scope"])
        result = annotate_material_supply(operands(), ["The customer already has locking pins.", "Travel is four hours."])
        self.assertIs(result[0]["scope"]["customer_supplied"], True)

    def test_equipment_ownership_does_not_establish_ownership_of_its_replacement_parts(self):
        inputs = [{"id": "U001", "kind": "user_input", "unit": "each", "value": 3,
                   "user_text": "Replace 3 pump seals.", "scope": {"entity_text": "pump seals"}}]
        result = annotate_material_supply(inputs, ["The customer has 2 pumps; replace 3 pump seals."])
        self.assertNotIn("customer_supplied", result[0]["scope"])
        self.assertIs(annotate_material_supply(inputs, ["The customer supplies pump seals."])[0]["scope"]["customer_supplied"], True)
        inputs[0].update(user_text="Replace 3 filter cloths.", scope={"entity_text": "filter cloths"})
        self.assertIs(annotate_material_supply(inputs, ["The customer already has the cloths."])[0]["scope"]["customer_supplied"], True)

    def test_ambiguous_we_supply_is_unresolved_and_partial_count_does_not_cover_the_larger_request(self):
        for message in ("We will supply the locking pins.", "We provide locking pins."):
            self.assertNotIn("customer_supplied", annotate_material_supply(operands(), [message])[0]["scope"])
        inputs = [{"id": "U001", "kind": "user_input", "unit": "each", "value": 4,
                   "user_text": "Replace 4 seals.", "scope": {"entity_text": "seals"}}]
        self.assertNotIn("customer_supplied", annotate_material_supply(inputs, ["The customer has 2 seals; replace 4 seals."])[0]["scope"])


class QuantitySupersessionTests(unittest.TestCase):
    def fact(self, identifier, turn, message, number, entity, unit="each", role="requested_quantity"):
        start = message.index(str(number))
        return {"id": identifier, "kind": "user_input", "value": Decimal(number), "unit": unit,
                "user_text": message, "label": "A broad label is not quantity identity",
                "scope": {"turn": turn, "span": [start, start+len(str(number))], "entity_text": entity, "role": role}}

    def test_explicit_shorter_noun_correction_retires_original_count_and_preserves_other_and_additive_items(self):
        messages = ["Need 5 braided cords and 2 retaining clips.", "Actually make that 7 cords, not 5.", "Also add 3 guard kits."]
        inputs = [self.fact("U001",0,messages[0],5,"braided cords"), self.fact("U002",0,messages[0],2,"retaining clips"),
                  self.fact("U003",1,messages[1],7,"cords"), self.fact("U004",1,messages[1],5,"",role="superseded_input"),
                  self.fact("U005",2,messages[2],3,"guard kits")]
        before = deepcopy(inputs)
        result = annotate_quantity_supersession(inputs,messages)
        self.assertEqual(result[0]["scope"]["role"], "superseded_input")
        self.assertEqual(result[0]["scope"]["supersession_provenance"], {"turn":1,"quote":messages[1],"replaced_by_ref":"U003"})
        self.assertEqual(result[0]["value"], Decimal(5))
        self.assertTrue(all(result[i]["scope"]["role"] == "requested_quantity" for i in (1,2,4)))
        self.assertEqual(inputs,before)

    def test_additive_uncertain_and_unmarked_counts_do_not_retire_earlier_requests(self):
        first = "Need 5 braided cords."
        for later in ("Actually add 7 cords.", "Actually make that 7 additional cords.",
                      "Could we make that 7 cords?", "If we need 7 cords instead of 5, what happens?", "Need 7 cords."):
            with self.subTest(later=later):
                inputs=[self.fact("U001",0,first,5,"braided cords"),self.fact("U002",1,later,7,"cords")]
                self.assertEqual(annotate_quantity_supersession(inputs,[first,later])[0]["scope"]["role"],"requested_quantity")

    def test_generic_correction_with_two_distinct_matching_entities_remains_ambiguous(self):
        first="Need 5 braided cords and 2 optical cords.";later="Actually make that 7 cords."
        inputs=[self.fact("U001",0,first,5,"braided cords"),self.fact("U002",0,first,2,"optical cords"),
                self.fact("U003",1,later,7,"cords")]
        result=annotate_quantity_supersession(inputs,[first,later])
        self.assertTrue(all(f["scope"]["role"]=="requested_quantity" for f in result))

    def test_distinct_modifiers_units_source_facts_and_nonliteral_claims_cannot_establish_supersession(self):
        first="Need 5 braided cords.";later="Actually make that 7 optical cords."
        inputs=[self.fact("U001",0,first,5,"braided cords"),self.fact("U002",1,later,7,"optical cords")]
        self.assertEqual(annotate_quantity_supersession(inputs,[first,later])[0]["scope"]["role"],"requested_quantity")
        later="Actually make that 7 cords."
        for change in ({"unit":"pair"},{"kind":"quantity"},{"value":Decimal(9)}, {"user_text":"Actually make that some cords."}):
            values=[self.fact("U001",0,first,5,"braided cords"),self.fact("U002",1,later,7,"cords")]
            values[1].update(change)
            self.assertEqual(annotate_quantity_supersession(values,[first,later])[0]["scope"]["role"],"requested_quantity")

    def test_correction_chain_keeps_the_newest_literal_active_with_original_ids_and_provenance(self):
        messages=["Need 5 braided cords.","Make that 7 cords.","8 cords, not 7."]
        inputs=[self.fact("U001",0,messages[0],5,"braided cords"),self.fact("U002",1,messages[1],7,"cords"),
                self.fact("U003",2,messages[2],8,"cords")]
        result=annotate_quantity_supersession(inputs,messages)
        self.assertEqual([f["scope"]["role"] for f in result],["superseded_input","superseded_input","requested_quantity"])
        self.assertEqual(result[1]["scope"]["supersession_provenance"]["replaced_by_ref"],"U003")


if __name__ == "__main__":
    unittest.main()
