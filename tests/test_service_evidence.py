"""Invented equipment and components exercise the generic evidence boundary."""
from copy import deepcopy
from contextlib import closing
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from metso_assistant.service_evidence import ServiceWorkspace, duration_scaling_matches, has_quantity_repetition, material_quantity_matches, quantity_relations
from metso_assistant.services import json_numbers, parse_text
from metso_assistant.store import build_store


class ServiceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.serial = 0

    def tearDown(self):
        self.temp.cleanup()

    def record(self, identifier="FSR-9001", country="Chile", currency="USD"):
        text = f"""Reference: {identifier}
Site: Invented coupler facility, {country}
Equipment: Helix Separator
On-site hours: 3
Travel hours: 1.5
| Task | Time Spent (hrs) |
|------|------------------|
| Replace Helix coupler cartridges | 2 |
| Preparation and testing | 1 |

| Item | Quantity | Unit Rate ({currency}) | Total ({currency}) |
|------|----------|-----------------------|------------------|
| Helix coupler cartridges | 3 | 7.15/ea | 21.45 |
| Labour | 3 hrs | 50/hr | 150 |
| Travel | 1.5 hrs | 20/hr | 30 |

Total Cost: {currency} 201.45
"""
        path = self.root / f"{identifier}.txt"
        path.write_bytes(text.encode("utf-8"))
        return parse_text(text, source_path=str(path))

    def workspace(self, records=None, inputs=None):
        self.serial += 1
        store = self.root / f"store-{self.serial}.sqlite"
        build_store(store, [], [], [], records or [self.record()])
        return ServiceWorkspace(store, self.root / f"index-{self.serial}.sqlite", inputs=inputs)

    def inputs(self, **updates):
        fact = {"id": "U001", "label": "Requested coupler cartridges", "value": 6,
                "unit": "each", "kind": "user_input", "user_text": "replace 6 coupler cartridges",
                "scope": {"role": "request_quantity"}}
        fact.update(updates)
        return [fact]

    def facts(self, workspace):
        workspace.read_report("FSR-9001")
        return list(workspace.facts.values())

    def choose(self, facts, kind, text=None, unit=None):
        return next(f for f in facts if f["kind"] == kind and (text is None or text in f["label"])
                    and (unit is None or f["unit"] == unit))

    def test_whole_question_retrieves_unseen_component_with_foreign_context_as_soft_ranking(self):
        workspace = self.workspace()
        answer = workspace.search("A customer needs 6 Helix coupler cartridges replaced; what should I budget?",
                                  country="Finland", equipment="Different equipment")
        price = self.choose(answer["facts"], "unit_price", "coupler")
        self.assertEqual(price["value"], Decimal("7.15"))
        self.assertEqual(price["scope"]["country"], "Chile")
        self.assertTrue(price["scope"]["validated"])
        self.assertLessEqual(len(answer["facts"]), 12)
        self.assertTrue(all(source["lines"] and source["source_sha256"] for source in answer["sources"]))
        self.assertTrue(all(type(ref["line_start"]) is int and type(ref["line_end"]) is int
                            for source in answer["sources"] for ref in source["lines"]))

    def test_explicit_source_directory_rechecks_relocated_originals_by_hash(self):
        record = self.record()
        original = Path(record['source']['path'])
        relocated = self.root / 'new-data' / 'HistoricalServices'
        relocated.mkdir(parents=True)
        replacement = relocated / original.name
        replacement.write_bytes(original.read_bytes())
        original.unlink()
        store = self.root / 'relocated.sqlite'
        build_store(store, [], [], [], [record])
        workspace = ServiceWorkspace(store, source_dir=relocated)
        self.assertEqual(workspace.index_path, store.with_name('service-evidence.sqlite'))
        self.assertFalse(any('unavailable' in warning for warning in workspace.read_report('FSR-9001')['warnings']))
        replacement.write_text('changed source', encoding='utf-8')
        changed = workspace.search('coupler')
        self.assertIn('Original service source changed', changed['error'])
        self.assertEqual(changed['facts'], [])
        with self.assertRaisesRegex(ValueError, 'Original service source changed'):
            ServiceWorkspace(store, source_dir=relocated)

    def test_exact_decimal_material_calculation_keeps_parents_assumptions_and_sources(self):
        workspace = self.workspace(inputs=self.inputs())
        facts = self.facts(workspace)
        price = self.choose(facts, "unit_price", "coupler")
        result = workspace.calculate("multiply", [price["id"], "U001"])
        calculation = result["facts"][0]
        self.assertEqual(calculation["value"], Decimal("42.90"))
        self.assertEqual(calculation["parents"], [price["id"], "U001"])
        self.assertTrue(calculation["assumptions"])
        self.assertEqual(calculation["source_ids"], price["source_ids"])
        self.assertEqual(json.loads(json_numbers(result), parse_float=Decimal)["facts"][0]["value"], Decimal("42.90"))
        result["facts"][0]["value"] = Decimal(999)
        self.assertEqual(workspace.facts[calculation["id"]]["value"], Decimal("42.90"))

    def duration_record(self, identifier="FSR-9101", *, description="Travel (2.5 hours)", quantity="2.5", rate="8", amount="20", total_currency="USD"):
        text=f"""Reference: {identifier}
Site: Invented coupler facility, Chile
Equipment: Helix Separator
| Item | Quantity | Unit Rate (USD) | Total ({total_currency}) |
|------|----------|-----------------|-------------------------|
| {description} | {quantity} | {rate} | {amount} |
"""
        path=self.root/f"{identifier}.txt";path.write_bytes(text.encode("utf-8"))
        return parse_text(text,source_path=str(path))

    def test_matching_same_row_duration_establishes_hourly_rate_and_propagates_assumption(self):
        for description in ("Travel (2.5 hours)","Labour (2.5 hrs)"):
            with self.subTest(description=description):
                workspace=self.workspace([self.duration_record(description=description)],inputs=[{
                    "id":"U_HOURS","value":Decimal(4),"unit":"hour","user_text":"Total travel and labour are each 4 hours.",
                    "scope":{"role":"travel_hours" if description.startswith("Travel") else "labour_hours"}}])
                result=workspace.read_report("FSR-9101")
                rate=self.choose(result["facts"],"hourly_rate")
                self.assertTrue(rate["scope"]["validated"])
                self.assertEqual((rate["unit"],rate["value"],rate["currency"]),("money/hour",Decimal(8),"USD"))
                self.assertEqual(rate["scope"]["denominator_basis"],"matching_same_row_hour_quantity")
                self.assertIsNone(rate["scope"]["quantity_unit_reported"])
                quantity=workspace.facts[rate["scope"]["quantity_ref"]]
                self.assertEqual((quantity["unit"],quantity["value"]),("hour",Decimal("2.5")))
                priced=workspace.calculate("multiply",[rate["id"],"U_HOURS"])["facts"][0]
                self.assertEqual(priced["value"],Decimal(32))
                self.assertTrue(any("same row" in text for text in priced["assumptions"]))

    def test_description_hour_inference_rejects_conflicting_quantities_units_currency_and_totals(self):
        variants=[{"quantity":"3"},{"quantity":"2.5 ml"},{"amount":"21"},{"total_currency":"EUR"},
                  {"description":"Travel (2.5 days)"},{"description":"Travel (2.5 hours plus 2.5 hours)"},
                  {"description":"Materials (2.5 hours)"}]
        for variant in variants:
            with self.subTest(variant=variant):
                workspace=self.workspace([self.duration_record(**variant)])
                result=workspace.read_report("FSR-9101")
                self.assertFalse(any(f["kind"]=="hourly_rate" and f["scope"].get("validated") for f in result["facts"]))
        record=self.duration_record()
        record["cost"]["line_items"][0]["header_evidence"]=[]
        workspace=self.workspace([record])
        self.assertFalse(any(f["kind"]=="hourly_rate" for f in workspace.read_report("FSR-9101")["facts"]))

    def test_country_travel_cohort_is_complete_exact_casefolded_and_loads_only_eligible_rates(self):
        records=[self.record(f"FSR-{9200+i}",country="Chile",currency="EUR" if i==17 else "USD") for i in range(18)]
        records.extend([self.record("FSR-9291",country="Canada"),self.duration_record("FSR-9292",amount="21")])
        workspace=self.workspace(records)
        result=workspace.travel_rates(" cHiLe ")
        self.assertEqual(result["status"],"ok")
        self.assertEqual(len(result["facts"]),18)
        self.assertEqual(set(result["data"]["rate_ids"]),{f["id"] for f in result["facts"]})
        self.assertEqual(result["data"]["selection"],"country_metadata")
        self.assertEqual({f["currency"] for f in result["facts"]},{"USD","EUR"})
        self.assertEqual({f["report_id"] for f in result["facts"]},{f"FSR-{9200+i}" for i in range(18)})
        self.assertEqual(set(workspace.facts),set(result["data"]["rate_ids"]))
        self.assertTrue(all(f["kind"]=="hourly_rate" and f["scope"]["category"]=="travel" for f in workspace.facts.values()))
        self.assertEqual(len(result["sources"]),18)
        self.assertTrue(all(type(span["line_start"]) is int for source in result["sources"] for span in source["lines"]))
        self.assertEqual(workspace.travel_rates("Chil")["status"],"insufficient_evidence")

    def cross_row_duration_record(self, declaration="Total travel hours: 2.5"):
        text=f"""Reference: FSR-9188
Site: Synthetic facility, Chile
{declaration}

| Item | Quantity/Hours | Unit Rate (USD) | Total (USD) |
|------|----------------|-----------------|-------------|
| Travel | 2.5 | 8 | 20 |
"""
        path=self.root/"FSR-9188.txt";path.write_bytes(text.encode("utf-8"))
        return parse_text(text,source_path=str(path))

    def test_independent_exact_visit_hour_statement_can_establish_missing_billing_unit_with_its_source(self):
        workspace=self.workspace([self.cross_row_duration_record()])
        result=workspace.travel_rates("Chile")
        self.assertEqual(result["status"],"ok")
        rate=result["facts"][0]
        self.assertEqual((rate["value"],rate["unit"]),(Decimal(8),"money/hour"))
        self.assertEqual(rate["scope"]["denominator_basis"],"matching_reported_hour_quantity")
        self.assertTrue(any("independent" in item for item in rate["assumptions"]))
        cited={span["line_start"] for source in result["sources"] for span in source["lines"]}
        self.assertEqual(cited,{3,5,7})

    def test_cross_row_hour_proof_rejects_approximate_conflicting_billed_only_and_overlapping_evidence(self):
        for declaration in ("", "Total travel hours: about 2.5", "Total travel hours: 3", "Total travel hours: 2 to 3"):
            with self.subTest(declaration=declaration):
                workspace=self.workspace([self.cross_row_duration_record(declaration)])
                self.assertFalse(workspace.travel_rates("Chile")["facts"])
        for change in ("overlap","billed_only","missing_header","conflicting_unit"):
            with self.subTest(change=change):
                record=self.cross_row_duration_record()
                row=record["cost"]["line_items"][0]
                if change=="overlap":
                    for candidate in record["hours"]["travel"]["candidates"]:
                        if candidate["scope"]=="visit_total":candidate["evidence"]=row["evidence"]
                elif change=="billed_only":record["hours"]["travel"]["candidates"]=[c for c in record["hours"]["travel"]["candidates"] if c["scope"]=="billed_hours"]
                elif change=="missing_header":row["rate_basis"]="not_stated"
                else:row["quantity_unit"]="ml"
                workspace=self.workspace([record])
                self.assertFalse(workspace.travel_rates("Chile")["facts"])

    def test_country_travel_missing_unknown_and_source_change_have_explicit_failure_states(self):
        record=self.record();workspace=self.workspace([record])
        for country in (None,"",[],42):
            with self.subTest(country=country):
                result=workspace.travel_rates(country)
                self.assertEqual(result["status"],"error");self.assertTrue(result["error"])
                self.assertFalse(result["facts"])
        self.assertEqual(workspace.travel_rates("Finland")["status"],"insufficient_evidence")
        Path(record["source"]["path"]).write_text("Changed original source",encoding="utf-8")
        changed=workspace.travel_rates("Chile")
        self.assertFalse(changed["facts"]);self.assertIn("source changed",changed["error"])

    def test_whole_visit_total_and_duration_do_not_create_an_hourly_travel_rate(self):
        text="Reference: FSR-9199\nSite: Invented site, Chile\nTravel hours: 4\nOn-site hours: 7\nTotal Cost: USD 800\n"
        path=self.root/"FSR-9199.txt";path.write_bytes(text.encode("utf-8"))
        workspace=self.workspace([parse_text(text,source_path=str(path))])
        result=workspace.travel_rates("Chile")
        self.assertFalse(result["facts"]);self.assertEqual(result["status"],"insufficient_evidence")

    def test_same_denominator_does_not_let_an_unrelated_material_use_the_requested_count(self):
        workspace = self.workspace(inputs=self.inputs(user_text="replace 6 Zircon locking pins"))
        price = self.choose(self.facts(workspace), "unit_price", "coupler")
        self.assertFalse(material_quantity_matches(price, workspace.facts["U001"]))
        result = workspace.calculate("multiply", [price["id"], "U001"])
        self.assertIn("component", result["error"])
        self.assertFalse(result["facts"])

    def test_material_count_needs_a_narrow_literal_component_not_a_broad_label_units_or_equipment(self):
        for text in ("6", "replace 6 parts", "replace 6 units of Helix Separator", "replace 6 new components"):
            with self.subTest(text=text):
                record = self.record()
                # Synthetic prepared metadata retains the literal Equipment
                # header even though the baseline classifier has no such family.
                record["equipment"]["description"] = "Helix Separator"
                workspace = self.workspace([record], inputs=self.inputs(user_text=text, label="Requested coupler cartridges"))
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertFalse(material_quantity_matches(price, workspace.facts["U001"]))
                self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))

    def test_shared_model_identifier_cannot_match_different_material_descriptors(self):
        for code in ("ZX700", "R5X", "AC42"):
            with self.subTest(code=code):
                requested = {"id": "U001", "value": Decimal(2), "unit": "each", "kind": "user_input",
                             "user_text": f"2 {code} rotors", "scope": {"span": [0, 1]}}
                rotor = {"kind": "unit_price", "unit": "money/each", "scope": {
                    "equipment": "Invented transfer pump", "entity_tokens": [code.casefold(), "rotor"]}}
                gasket = deepcopy(rotor); gasket["scope"]["entity_tokens"] = [code.casefold(), "gasket"]
                self.assertTrue(material_quantity_matches(rotor, requested))
                self.assertFalse(material_quantity_matches(gasket, requested))

    def test_identifier_only_material_request_stays_ambiguous(self):
        requested = {"id": "U001", "value": Decimal(2), "unit": "each", "kind": "user_input",
                     "user_text": "2 ZX700", "scope": {"span": [0, 1]}}
        for descriptor in ("rotor", "gasket"):
            with self.subTest(descriptor=descriptor):
                rate = {"kind": "unit_price", "unit": "money/each", "scope": {
                    "equipment": "Invented transfer pump", "entity_tokens": ["zx700", descriptor]}}
                self.assertFalse(material_quantity_matches(rate, requested))

    def test_material_match_uses_the_actual_operand_span_and_narrow_entity_extraction(self):
        text = "Previously 3 coupler cartridges were discussed; now replace 6 rotors on Helix Separator."
        start = text.index("6")
        for entity in ("rotors on Helix Separator", "coupler cartridges"):
            with self.subTest(entity=entity):
                workspace = self.workspace(inputs=self.inputs(user_text=text, scope={"span": [start, start+1], "entity_text": entity}))
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertFalse(material_quantity_matches(price, workspace.facts["U001"]))
                self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))
        text = "Replace 6 coupler cartridges on Helix Separator, then inspect the rotors."
        start = text.index("6")
        workspace = self.workspace(inputs=self.inputs(user_text=text, scope={"span": [start,start+1], "entity_text": "coupler cartridges on Helix Separator"}))
        price = self.choose(self.facts(workspace), "unit_price", "coupler")
        self.assertTrue(material_quantity_matches(price, workspace.facts["U001"]))
        self.assertEqual(workspace.calculate("multiply", [price["id"], "U001"])["facts"][0]["value"], Decimal("42.90"))

    def test_repeated_material_quantity_follows_the_immutable_literal_per_item_operand(self):
        inputs = self.repeated_inputs("2 machines need 3 sets of coupler cartridges per machine.")
        record = self.record()
        record["cost"]["line_items"][0].update(quantity_unit="set", rate_unit="set")
        workspace = self.workspace([record], inputs)
        price = self.choose(self.facts(workspace), "unit_price", "coupler")
        quantity = workspace.calculate("quantity_product", ["U001", "U002"])["facts"][0]
        self.assertFalse(material_quantity_matches(price, quantity))
        self.assertTrue(material_quantity_matches(price, quantity, workspace.facts))
        workspace.facts["U001"]["user_text"] = "2 machines need 3 sets of rotors per machine."
        result = workspace.calculate("multiply", [price["id"], quantity["id"]])
        self.assertIn("modified", result["error"])
        self.assertFalse(result["facts"])

    def test_customer_owned_material_cannot_be_purchased_but_its_count_still_supports_labour(self):
        from metso_assistant.request_scope import annotate_material_supply
        inputs = annotate_material_supply(self.inputs(), ["Replace 6 coupler cartridges.", "We already have the coupler cartridges."])
        workspace = self.workspace(inputs=inputs)
        facts = self.facts(workspace)
        price = self.choose(facts, "unit_price", "coupler")
        self.assertFalse(material_quantity_matches(price, workspace.facts["U001"]))
        self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))
        task = self.choose(facts, "task_duration", "coupler")
        observed = self.choose(facts, "quantity", "coupler", "each")
        self.assertTrue(duration_scaling_matches(task, workspace.facts["U001"], observed, workspace.facts))
        duration = workspace.calculate("scale_duration", [task["id"], "U001", observed["id"]])["facts"][0]
        self.assertEqual(duration["value"], Decimal(4))
        labour = self.choose(facts, "hourly_rate", "Labour")
        amount = workspace.calculate("multiply", [labour["id"], duration["id"]])["facts"][0]
        self.assertEqual(amount["value"], Decimal("200.00"))

    def test_duration_scaling_rejects_an_unrelated_requested_component_even_when_task_and_source_count_match(self):
        workspace = self.workspace(inputs=self.inputs(user_text="Replace 6 Zircon locking pins."))
        facts = self.facts(workspace)
        task = self.choose(facts, "task_duration", "coupler")
        observed = self.choose(facts, "quantity", "coupler", "each")
        self.assertFalse(duration_scaling_matches(task, workspace.facts["U001"], observed, workspace.facts))
        result = workspace.calculate("scale_duration", [task["id"], "U001", observed["id"]])
        self.assertIn("matching literal requested component", result["error"])
        self.assertFalse(result["facts"])

    def test_explicit_task_row_quantity_takes_precedence_over_another_same_component_bill_count(self):
        record = self.record()
        task_row = record["tasks"][0]["evidence"]
        record["service_quantities"] = [{"item": "Helix coupler cartridges", "quantity": Decimal(9),
                                          "quantity_unit": "each", "activity": "replacement", "evidence": task_row}]
        workspace = self.workspace([record], self.inputs())
        facts = self.facts(workspace)
        task = self.choose(facts, "task_duration", "coupler")
        row_count = next(f for f in facts if f["kind"] == "quantity" and f["value"] == 9)
        bill_count = self.choose(facts, "quantity", "coupler", "each")
        self.assertTrue(duration_scaling_matches(task, workspace.facts["U001"], row_count, workspace.facts))
        self.assertFalse(duration_scaling_matches(task, workspace.facts["U001"], bill_count, workspace.facts))
        self.assertIn("error", workspace.calculate("scale_duration", [task["id"], "U001", bill_count["id"]]))
        scaled = workspace.calculate("scale_duration", [task["id"], "U001", row_count["id"]])["facts"][0]
        self.assertFalse(any("No task-row" in item for item in scaled["assumptions"]))

    def test_explicit_replacement_and_inspection_requests_do_not_exchange_task_scenarios(self):
        record = self.record()
        path = Path(record["source"]["path"])
        text = path.read_text(encoding="utf-8").replace("| Replace Helix coupler cartridges | 2 |",
            "| Inspect 9 Helix coupler cartridges | 1 |\n| Replace 3 Helix coupler cartridges | 1 |")
        path.write_bytes(text.encode("utf-8"))
        record = parse_text(text, source_path=str(path))
        record["service_quantities"] = [
            {"item": "Helix coupler cartridges", "quantity": Decimal(quantity), "quantity_unit": "each",
             "activity": activity, "evidence": record["tasks"][index]["evidence"]}
            for index, quantity, activity in ((0, 9, "inspection"), (1, 3, "replacement"))]
        for verb, expected in (("Replace", "replacement"), ("Inspect", "inspection")):
            with self.subTest(verb=verb):
                workspace = self.workspace([record], self.inputs(user_text=f"{verb} 6 coupler cartridges."))
                facts = self.facts(workspace)
                for activity, label in (("inspection", "Inspect"), ("replacement", "Replace")):
                    task = next(f for f in facts if f["kind"] == "task_duration" and f["label"].startswith(label))
                    quantity = next(f for f in facts if f["kind"] == "quantity" and f["scope"].get("activity") == activity)
                    self.assertEqual(duration_scaling_matches(task, workspace.facts["U001"], quantity, workspace.facts), activity == expected)
                    result = workspace.calculate("scale_duration", [task["id"], "U001", quantity["id"]])
                    self.assertEqual("error" not in result, activity == expected)

    def test_two_component_denominators_of_the_same_combined_task_cannot_double_bill_its_hours(self):
        record = self.record()
        record["tasks"][0]["description"] = "Replace Helix coupler cartridges and locking pins"
        row = deepcopy(record["cost"]["line_items"][0]); row.update(description="Helix locking pins", quantity=Decimal(2), amount=Decimal("14.30"))
        record["cost"]["line_items"].append(row)
        inputs = self.inputs() + [{"id": "U002", "kind": "user_input", "value": 4, "unit": "each", "user_text": "Replace 4 locking pins."}]
        workspace = self.workspace([record], inputs)
        facts = self.facts(workspace); task = self.choose(facts, "task_duration", "coupler")
        couplers = self.choose(facts, "quantity", "coupler", "each")
        pins = self.choose(facts, "quantity", "locking pins", "each")
        durations = [workspace.calculate("scale_duration", [task["id"], user, observed["id"]])["facts"][0]
                     for user, observed in (("U001", couplers), ("U002", pins))]
        self.assertTrue(all(any("No task-row" in a for a in duration["assumptions"]) for duration in durations))
        self.assertIn("error", workspace.calculate("sum", [duration["id"] for duration in durations]))
        labour = self.choose(facts, "hourly_rate", "Labour")
        amounts = [workspace.calculate("multiply", [labour["id"], duration["id"]])["facts"][0] for duration in durations]
        self.assertIn("error", workspace.calculate("sum", [amount["id"] for amount in amounts]))

    def test_actual_user_billing_duration_supersedes_source_hours_only_for_its_own_role(self):
        inputs = [{"id": "U001", "kind": "user_input", "value": 4, "unit": "hour",
                   "user_text": "Round-trip travel is 4 hours.", "scope": {"role": "travel_hours", "turn": 1}}]
        workspace = self.workspace(inputs=inputs)
        facts = self.facts(workspace)
        travel = self.choose(facts, "hourly_rate", "Travel")
        source_travel = next(f for f in facts if f["unit"] == "hour" and f["scope"].get("role") == "travel_hours" and f.get("source_ids"))
        rejected = workspace.calculate("multiply", [travel["id"], source_travel["id"]])
        self.assertIn("supersede", rejected["error"])
        current = workspace.calculate("multiply", [travel["id"], "U001"])["facts"][0]
        self.assertEqual(current["value"], Decimal("80.00"))
        labour = self.choose(facts, "hourly_rate", "Labour")
        source_labour = next(f for f in facts if f["unit"] == "hour" and f["scope"].get("role") == "labour_hours")
        self.assertTrue(workspace.calculate("multiply", [labour["id"], source_labour["id"]])["facts"])
        no_user_duration = self.workspace()
        original = self.facts(no_user_duration)
        original_travel = self.choose(original, "hourly_rate", "Travel")
        original_hours = next(f for f in original if f["unit"] == "hour" and f["scope"].get("role") == "travel_hours")
        self.assertTrue(no_user_duration.calculate("multiply", [original_travel["id"], original_hours["id"]])["facts"])

    def test_partial_inconsistent_unknown_currency_prices_remain_observations_not_calculation_bases(self):
        for mutation in ("amount_missing", "amount_wrong", "currency_unknown"):
            with self.subTest(mutation=mutation):
                record = self.record()
                line = record["cost"]["line_items"][0]
                if mutation == "amount_missing": line["amount"] = None
                elif mutation == "amount_wrong": line["amount"] = Decimal(1)
                else: line["currency"] = line["unit_rate_currency"] = None
                workspace = self.workspace([record], self.inputs())
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertFalse(price["scope"]["validated"])
                self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))

    def test_duplicate_conflicting_prices_are_not_cherry_picked(self):
        record = self.record()
        duplicate = deepcopy(record["cost"]["line_items"][0])
        duplicate.update(unit_rate=Decimal(8), amount=Decimal(24))
        record["materials"].append(duplicate)
        workspace = self.workspace([record], self.inputs())
        prices = [f for f in self.facts(workspace) if f["kind"] == "unit_price" and "coupler" in f["label"]]
        self.assertEqual(len(prices), 2)
        self.assertTrue(all(not f["scope"]["validated"] for f in prices))
        self.assertTrue(all(workspace.calculate("multiply", [f["id"], "U001"])["facts"] == [] for f in prices))

    def test_dimensioned_package_rates_match_only_explicit_same_units(self):
        record = self.record()
        row = record["cost"]["line_items"][0]
        row.update(description="Helix protective sleeves", quantity=Decimal(3), quantity_unit="pair",
                   unit_rate=Decimal("7.15"), rate_unit="pair")
        workspace = self.workspace([record], self.inputs(unit="pair", user_text="replace 6 pairs of protective sleeves"))
        price = self.choose(self.facts(workspace), "unit_price", "sleeves")
        self.assertEqual(price["unit"], "money/pair")
        self.assertEqual(workspace.calculate("multiply", [price["id"], "U001"])["facts"][0]["value"], Decimal("42.90"))
        wrong = self.workspace([record], self.inputs(unit="each"))
        price = self.choose(self.facts(wrong), "unit_price", "sleeves")
        self.assertIn("error", wrong.calculate("multiply", [price["id"], "U001"]))

    def test_explicit_each_rate_for_a_whole_named_package_does_not_expand_package_contents(self):
        record = self.record()
        record["cost"]["line_items"][0].update(description="Helix coupler sets", quantity_unit="set", rate_unit="each")
        workspace = self.workspace([record], self.inputs(unit="set"))
        price = self.choose(self.facts(workspace), "unit_price", "sets")
        self.assertEqual(price["unit"], "money/set")
        self.assertTrue(workspace.calculate("multiply", [price["id"], "U001"])["facts"])

    def test_ambiguous_volume_lump_sum_does_not_become_per_volume_price(self):
        record = self.record()
        record["cost"]["line_items"][0].update(description="Helix protective fluid", quantity=Decimal(150),
            quantity_unit="ml", unit_rate=Decimal(24), rate_unit=None, amount=Decimal(24), rate_basis="unit_rate_header")
        workspace = self.workspace([record], self.inputs(unit="ml", value=300))
        price = self.choose(self.facts(workspace), "unit_price", "fluid")
        self.assertFalse(price["scope"]["validated"])
        self.assertEqual(price["unit"], "money")
        self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))

    def test_literal_quantity_units_and_coherent_rate_columns_support_generic_denominators(self):
        for literal, unit in (("units", "each"), ("pairs", "pair"), ("ml", "ml"), ("L", "litre")):
            with self.subTest(literal=literal):
                record = self.record()
                path = Path(record["source"]["path"])
                text = path.read_bytes().decode("utf-8").replace("Unit Rate (USD)", "Rate (USD)")
                text = text.replace("cartridges | 3 |", f"cartridges | 3 {literal} |")
                text = text.replace("7.15/ea", "7.15")
                path.write_bytes(text.encode("utf-8"))
                record = parse_text(text, source_path=str(path))
                self.assertEqual(record["cost"]["line_items"][0]["rate_basis"], "not_stated")
                workspace = self.workspace([record], self.inputs(unit=unit))
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertEqual(price["unit"], "money/" + unit)
                self.assertTrue(price["scope"]["validated"])
                self.assertEqual(workspace.calculate("multiply", [price["id"], "U001"])["facts"][0]["value"], Decimal("42.90"))

    def test_quantity_unit_alone_cannot_promote_a_missing_or_incoherent_rate_basis(self):
        for mutation in ("mismatched_amount", "missing_header", "unknown_currency"):
            with self.subTest(mutation=mutation):
                record = self.record()
                row = record["cost"]["line_items"][0]
                row.update(quantity_unit="unit", rate_unit=None, rate_basis="not_stated")
                if mutation == "mismatched_amount": row["amount"] = Decimal(20)
                elif mutation == "missing_header": row["header_evidence"] = []
                else: row["currency"] = row["unit_rate_currency"] = None
                workspace = self.workspace([record], self.inputs())
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertFalse(price["scope"]["validated"])
                self.assertEqual(price["unit"], "money")
                self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))

    def test_unit_rate_headers_can_count_each_listed_part_with_an_explicit_assumption(self):
        for is_material in (False, True):
            with self.subTest(is_material=is_material):
                record = self.record()
                row = record["cost"]["line_items"][0]
                row.update(quantity_unit=None, rate_unit=None, rate_basis="unit_rate_header")
                if is_material:
                    record["cost"]["line_items"].remove(row)
                    row.pop("category")
                    record["materials"].append(row)
                workspace = self.workspace([record], self.inputs())
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertEqual(price["unit"], "money/each")
                self.assertTrue(price["scope"]["validated"])
                self.assertTrue(any("each listed item" in a for a in price["assumptions"]))
                calculation = workspace.calculate("multiply", [price["id"], "U001"])["facts"][0]
                self.assertEqual(calculation["value"], Decimal("42.90"))
                self.assertTrue(set(price["assumptions"]).issubset(calculation["assumptions"]))

    def test_implied_whole_item_units_require_positive_coherent_counts_prices_and_no_unknown_literal_unit(self):
        for mutation in ("fraction", "plain_rate_header", "unknown_unit", "unknown_currency", "labour_without_hours"):
            with self.subTest(mutation=mutation):
                record = self.record()
                row = record["cost"]["line_items"][0]
                row.update(quantity_unit=None, rate_unit=None, rate_basis="unit_rate_header")
                if mutation == "fraction": row.update(quantity=Decimal("1.5"), amount=Decimal("10.73"))
                elif mutation == "plain_rate_header": row["rate_basis"] = "not_stated"
                elif mutation == "unknown_unit": row["quantity_unit"] = "kg"
                elif mutation == "unknown_currency": row["currency"] = row["unit_rate_currency"] = None
                else:
                    row["category"] = "labour"
                    record["hours"]["on_site"].update(status="not_stated",value=None,candidates=[])
                workspace = self.workspace([record], self.inputs())
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertFalse(price["scope"]["validated"])
                self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))

    def test_zero_or_customer_supplied_material_rates_are_not_new_purchase_price_bases(self):
        for supplied, price_value in ((False, Decimal(0)), (True, Decimal("7.15"))):
            with self.subTest(supplied=supplied):
                record = self.record()
                row = record["cost"]["line_items"][0]
                row.update(unit_rate=price_value, amount=price_value * 3, customer_supplied=supplied)
                workspace = self.workspace([record], self.inputs())
                price = self.choose(self.facts(workspace), "unit_price", "coupler")
                self.assertEqual(price["value"], price_value)
                self.assertFalse(price["scope"]["validated"])
                self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))

    def repeated_inputs(self, text, per_value=3, per_unit="set", equipment_value=2):
        return [{"id": "U001", "label": "Per-machine requested material", "value": per_value, "unit": per_unit,
                 "user_text": text, "scope": {"role": "requested_quantity", "turn": 2}},
                {"id": "U002", "label": "Equipment count", "value": equipment_value, "unit": "each",
                 "user_text": text, "scope": {"role": "equipment_count", "turn": 1}}]

    def test_literal_repeated_counts_compose_units_parents_and_support_checked_pricing_and_task_scaling(self):
        record = self.record()
        record["cost"]["line_items"][0].update(quantity_unit="set", rate_unit="set")
        workspace = self.workspace([record], self.repeated_inputs("The customer has 2 machines and needs 3 sets of coupler cartridges per machine."))
        facts = self.facts(workspace)
        combined = workspace.calculate("quantity_product", ["U001", "U002"])["facts"][0]
        self.assertEqual((combined["value"], combined["unit"], combined["kind"]), (Decimal(6), "set", "request_quantity"))
        self.assertEqual(combined["parents"], ["U001", "U002"])
        self.assertFalse(combined["source_ids"])
        self.assertEqual(combined["scope"]["turn"], 2)
        price = self.choose(facts, "unit_price", "coupler")
        amount = workspace.calculate("multiply", [price["id"], combined["id"]])["facts"][0]
        self.assertEqual(amount["value"], Decimal("42.90"))
        self.assertTrue(set(combined["assumptions"]).issubset(amount["assumptions"]))
        task = self.choose(facts, "task_duration", "coupler")
        observed = self.choose(facts, "quantity", "coupler", "set")
        scaled = workspace.calculate("scale_duration", [task["id"], combined["id"], observed["id"]])["facts"][0]
        self.assertEqual(scaled["value"], Decimal(4))

    def test_one_item_each_for_two_machines_uses_actual_word_counts(self):
        workspace = self.workspace(inputs=self.repeated_inputs("Two pumps need one coupler each.", per_value=1, per_unit="each"))
        combined = workspace.calculate("quantity_product", ["U001", "U002"])["facts"][0]
        self.assertEqual(combined["value"], Decimal(2))
        self.assertEqual(combined["unit"], "each")

    def test_repeated_count_requires_a_real_relation_not_an_existing_total_or_a_unit_price(self):
        for text in ("The customer has 2 machines and needs 3 sets in total.",
                     "Price 3 sets at USD 7.15 each for 2 machines.",
                     "We have 2 machines and need 3 sets at USD 7.15 per set."):
            with self.subTest(text=text):
                workspace = self.workspace(inputs=self.repeated_inputs(text))
                self.assertIn("error", workspace.calculate("quantity_product", ["U001", "U002"]))

    def test_repeated_counts_reject_fractional_zero_made_up_and_reused_operands(self):
        with self.assertRaises(ValueError):
            self.workspace(inputs=self.repeated_inputs("2 machines need 1.5 sets per machine.", per_value=Decimal("1.5")))
        workspace = self.workspace(inputs=self.repeated_inputs("2 machines need 3 sets per machine."))
        self.assertIn("error", workspace.calculate("quantity_product", ["U001", "made-up"]))
        self.assertIn("error", workspace.calculate("quantity_product", ["U001", "U001"]))
        zero = self.workspace(inputs=self.repeated_inputs("0 machines need 3 sets per machine.", equipment_value=0))
        self.assertIn("error", zero.calculate("quantity_product", ["U001", "U002"]))

    def test_quantity_product_requires_literal_counts_not_a_model_provided_label(self):
        workspace = self.workspace(inputs=self.repeated_inputs("The number of machines and sets per machine are unknown."))
        self.assertIn("error", workspace.calculate("quantity_product", ["U001", "U002"]))

    def test_quantity_product_accepts_prefix_each_but_rejects_an_unrelated_repetition_phrase(self):
        for text in ("Two machines each need 3 sets.", "For every machine, provide 3 sets; there are two machines."):
            workspace = self.workspace(inputs=self.repeated_inputs(text))
            self.assertEqual(workspace.calculate("quantity_product", ["U001", "U002"])["facts"][0]["value"], Decimal(6))
        for text in ("For each machine, inspect a panel; two machines need 3 sets overall.",
                     "Two machines need 3 sets, plus 5 clips per machine.",
                     "Two machines need 3 sets; charge USD 7 each."):
            workspace = self.workspace(inputs=self.repeated_inputs(text))
            self.assertIn("error", workspace.calculate("quantity_product", ["U001", "U002"]))

    def literal_inputs(self, text, specifications):
        inputs, cursor = [], 0
        for index, (token, value, unit) in enumerate(specifications, 1):
            start = text.index(token, cursor); cursor = start + len(token)
            inputs.append({"id": f"U{index:03}", "kind": "user_input", "label": text[start:cursor+35],
                           "value": value, "unit": unit, "user_text": text,
                           "scope": {"role": "requested_quantity", "span": [start, cursor], "turn": 0}})
        return inputs

    def test_shared_trailing_each_covers_every_conjunctive_component_and_only_the_subject_anchor(self):
        text = "A site has 2 machines and needs one cartridge and one locking set for each."
        inputs = self.literal_inputs(text, [("2", 2, "each"), ("one", 1, "each"), ("one", 1, "set")])
        expected = [{"per_item_ref": "U002", "equipment_count_ref": "U001"},
                    {"per_item_ref": "U003", "equipment_count_ref": "U001"}]
        self.assertEqual(quantity_relations(inputs), expected)
        before = deepcopy(inputs)
        workspace = self.workspace(inputs=inputs)
        for identifier, unit in (("U002", "each"), ("U003", "set")):
            result = workspace.calculate("quantity_product", [identifier, "U001"])
            self.assertEqual((result["facts"][0]["value"], result["facts"][0]["unit"]), (Decimal(2), unit))
            self.assertEqual(result["facts"][0]["parents"], [identifier, "U001"])
        self.assertEqual(inputs, before)

    def test_leading_each_in_the_next_sentence_governs_the_following_component_list(self):
        text = "We have 3 machines. Each needs 2 seals and 1 rotor."
        inputs = self.literal_inputs(text, [("3", 3, "each"), ("2", 2, "each"), ("1", 1, "each")])
        self.assertEqual(quantity_relations(inputs), [
            {"per_item_ref": "U002", "equipment_count_ref": "U001"},
            {"per_item_ref": "U003", "equipment_count_ref": "U001"}])
        workspace = self.workspace(inputs=inputs)
        self.assertEqual(workspace.calculate("quantity_product", ["U002", "U001"])["facts"][0]["value"], Decimal(6))
        self.assertEqual(workspace.calculate("quantity_product", ["U003", "U001"])["facts"][0]["value"], Decimal(3))

    def test_shared_quantity_proof_rejects_alternatives_price_each_totals_and_unrelated_clauses(self):
        texts = [
            "We have 3 machines. Each needs 2 seals or 1 rotor.",
            "We have 3 machines and need 2 seals or 1 rotor for each.",
            "We have 3 machines and need 2 seals and 1 rotor in total for each.",
            "We have 3 machines and need 2 seals and 1 rotor per machine overall.",
            "We have 3 machines and need 2 seals; buy 1 rotor at USD 7 each.",
            "We have 3 machines and need 2 seals. Another workshop wants 1 rotor for each.",
        ]
        for text in texts:
            with self.subTest(text=text):
                inputs = self.literal_inputs(text, [("3", 3, "each"), ("2", 2, "each"), ("1", 1, "each")])
                self.assertFalse(quantity_relations(inputs))

    def test_public_relations_use_only_finite_positive_whole_literal_user_counts(self):
        text = "We have 3 machines. Each needs 2 seals."
        inputs = self.literal_inputs(text, [("3", 3, "each"), ("2", 2, "each")])
        for change in ({"kind": "quantity"}, {"kind": "unit_price"}, {"unit": "hour"},
                       {"value": Decimal("1.5")}, {"value": float("inf")}, {"value": True},
                       {"value": 0}, {"value": 9}, {"scope": {"role": "analytical_parameter"}}):
            altered = deepcopy(inputs); altered[1].update(change)
            with self.subTest(change=str(change)):
                self.assertFalse(quantity_relations(altered))
        for malformed in (None, {}, [None], [inputs[0], {"id": [], "kind": "user_input"}]):
            self.assertFalse(quantity_relations(malformed))

    def test_literal_repetition_intent_can_be_reported_without_an_available_machine_count(self):
        self.assertTrue(has_quantity_repetition([], ["The machine count is unknown. Each machine needs one new seal."]))
        self.assertTrue(has_quantity_repetition([], ["What is the repair duration per machine? The count is unknown."]))
        self.assertFalse(quantity_relations([]))
        for message in ("The seals cost USD 7 each.", "Use 2 seals or 1 rotor for each.", "We need 3 seals in total."):
            self.assertFalse(has_quantity_repetition([], [message]))

    def test_no_implicit_litres_to_millilitres_or_currency_conversion(self):
        record = self.record()
        record["cost"]["line_items"][0].update(description="Helix protective fluid", quantity=Decimal(3),
            quantity_unit="litre", rate_unit="litre")
        workspace = self.workspace([record], self.inputs(unit="ml", value=300))
        price = self.choose(self.facts(workspace), "unit_price", "fluid")
        self.assertIn("error", workspace.calculate("multiply", [price["id"], "U001"]))
        self.assertIn("error", workspace.calculate("convert", [price["id"]]))

    def test_generic_task_scaling_has_explicit_linear_assumption_and_rejects_visit_or_unrelated_quantity(self):
        workspace = self.workspace(inputs=self.inputs())
        facts = self.facts(workspace)
        task = self.choose(facts, "task_duration", "coupler")
        observed = self.choose(facts, "quantity", "coupler", "each")
        result = workspace.calculate("scale_duration", [task["id"], "U001", observed["id"]])
        self.assertEqual(result["facts"][0]["value"], Decimal(4))
        self.assertTrue(any("Linear scaling" in assumption for assumption in result["facts"][0]["assumptions"]))
        visit = next(f for f in facts if f["scope"].get("type") == "visit_duration" and f["unit"] == "hour")
        self.assertIn("error", workspace.calculate("scale_duration", [visit["id"], "U001", observed["id"]]))
        preparation = self.choose(facts, "task_duration", "Preparation")
        self.assertIn("error", workspace.calculate("scale_duration", [preparation["id"], "U001", observed["id"]]))

    def test_scaling_needs_same_report_and_no_ambiguous_quantity_conversion(self):
        workspace = self.workspace([self.record(), self.record("FSR-9002", country="Finland")], self.inputs())
        facts = self.facts(workspace)
        workspace.read_report("FSR-9002")
        task = self.choose(facts, "task_duration", "coupler")
        observed = next(f for f in workspace.facts.values() if f["report_id"] == "FSR-9002" and f["unit"] == "each")
        self.assertIn("error", workspace.calculate("scale_duration", [task["id"], "U001", observed["id"]]))

    def test_task_and_material_costs_can_be_added_but_duplicate_alternatives_cannot(self):
        workspace = self.workspace(inputs=self.inputs())
        facts = self.facts(workspace)
        price = self.choose(facts, "unit_price", "coupler")
        rate = self.choose(facts, "hourly_rate", "Labour")
        task = self.choose(facts, "task_duration", "coupler")
        observed = self.choose(facts, "quantity", "coupler", "each")
        hours = workspace.calculate("scale_duration", [task["id"], "U001", observed["id"]])["facts"][0]
        labour = workspace.calculate("multiply", [rate["id"], hours["id"]])["facts"][0]
        parts = workspace.calculate("multiply", [price["id"], "U001"])["facts"][0]
        subtotal = workspace.calculate("sum", [parts["id"], labour["id"]])["facts"][0]
        self.assertEqual(subtotal["value"], Decimal("242.90"))
        self.assertIn("error", workspace.calculate("sum", [parts["id"], parts["id"]]))
        self.assertIn("error", workspace.calculate("sum", [parts["id"], subtotal["id"]]))
        original_amount = next(f for f in facts if f["scope"].get("type") == "reported_component_amount" and "coupler" in f["label"])
        self.assertIn("error", workspace.calculate("sum", [parts["id"], original_amount["id"]]))
        whole = self.choose(facts, "visit_total")
        self.assertIn("error", workspace.calculate("sum", [whole["id"], parts["id"]]))

    def test_foreign_currency_components_remain_separate_and_reported_total_never_added_to_parts(self):
        workspace = self.workspace([self.record(), self.record("FSR-9002", currency="EUR")], self.inputs())
        self.facts(workspace); workspace.read_report("FSR-9002")
        components = [f for f in workspace.facts.values() if f["kind"] == "unit_price" and "coupler" in f["label"]]
        amounts = [workspace.calculate("multiply", [f["id"], "U001"])["facts"][0] for f in components]
        self.assertIn("error", workspace.calculate("sum", [f["id"] for f in amounts]))

    def test_travel_rate_cannot_be_used_as_labour_rate_and_ambiguous_hours_stay_unusable(self):
        workspace = self.workspace(inputs=self.inputs(unit="number", value=4, label="Ambiguous trip hours"))
        facts = self.facts(workspace)
        travel = self.choose(facts, "hourly_rate", "Travel")
        task = self.choose(facts, "task_duration", "coupler")
        self.assertIn("error", workspace.calculate("multiply", [travel["id"], task["id"]]))
        self.assertIn("error", workspace.calculate("multiply", [travel["id"], "U001"]))

    def test_shared_rate_is_reusable_for_distinct_tasks_but_overlapping_duration_charges_are_rejected(self):
        workspace = self.workspace()
        facts = self.facts(workspace)
        rate = self.choose(facts, "hourly_rate", "Labour")
        tasks = [f for f in facts if f["kind"] == "task_duration"]
        costs = [workspace.calculate("multiply", [rate["id"], t["id"]])["facts"][0] for t in tasks]
        self.assertEqual(workspace.calculate("sum", [f["id"] for f in costs])["facts"][0]["value"], Decimal(150))
        total_hours = workspace.calculate("sum", [t["id"] for t in tasks])["facts"][0]
        all_cost = workspace.calculate("multiply", [rate["id"], total_hours["id"]])["facts"][0]
        self.assertIn("error", workspace.calculate("sum", [all_cost["id"], costs[0]["id"]]))

    def test_made_up_modified_nonfinite_and_fractional_count_inputs_are_rejected(self):
        for value in (float("nan"), float("inf"), Decimal("NaN"), True, "6", Decimal("1.5"), -6):
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError): self.workspace(inputs=self.inputs(value=value))
        workspace = self.workspace(inputs=self.inputs())
        self.assertIn("error", workspace.calculate("sum", ["invented", "U001"]))
        workspace.facts["U001"]["value"] = Decimal(999)
        self.assertIn("error", workspace.calculate("sum", ["U001", "U002"]))

    def test_sidecar_binding_reuse_tamper_rebuild_and_store_readonly(self):
        workspace = self.workspace()
        original = workspace.store_path.read_bytes()
        first = workspace.index_path.read_bytes()
        rebuilt = ServiceWorkspace(workspace.store_path, workspace.index_path)
        self.assertEqual(rebuilt.index_path.read_bytes(), first)
        with closing(sqlite3.connect(workspace.index_path)) as connection:
            connection.execute("UPDATE observations_fts SET text='tampered search words'")
            connection.commit()
        repaired = ServiceWorkspace(workspace.store_path, workspace.index_path)
        self.assertTrue(repaired.search("Helix coupler cartridges")["facts"])
        self.assertEqual(repaired.store_path.read_bytes(), original)
        self.assertFalse(any(workspace.index_path.parent.glob("*.tmp")))

    def test_changed_original_or_changed_prepared_store_invalidates_live_workspace(self):
        record = self.record()
        workspace = self.workspace([record])
        Path(record["source"]["path"]).write_text("changed source", encoding="utf-8")
        self.assertIn("error", workspace.search("coupler"))
        with self.assertRaises(ValueError): ServiceWorkspace(workspace.store_path, workspace.index_path)
        other = self.workspace()
        with closing(sqlite3.connect(other.store_path)) as connection:
            connection.execute("UPDATE metadata SET value='changed' WHERE key='schema_version'")
            connection.commit()
        self.assertIn("error", other.search("coupler"))

    def test_original_missing_has_explicit_warning_but_prepared_provenance_survives(self):
        record = self.record()
        Path(record["source"]["path"]).unlink()
        workspace = self.workspace([record])
        answer = workspace.search("coupler")
        self.assertTrue(answer["facts"])
        self.assertTrue(any("Original sources unavailable" in warning for warning in answer["warnings"]))
        self.assertTrue(all(source["source_sha256"] == record["source"]["sha256"] for source in answer["sources"]))

    def test_every_report_is_indexed_without_the_old_thousand_record_cutoff(self):
        base = self.record()
        records = []
        for index in range(1003):
            record = deepcopy(base)
            record["report_id"] = f"FSR-{10000+index}"
            records.append(record)
        workspace = self.workspace(records)
        result = workspace.read_report("FSR-11002")
        self.assertTrue(result["facts"])
        with closing(sqlite3.connect(workspace.index_path)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM reports").fetchone()[0], len(records))
        self.assertLessEqual(len(result["facts"]), 20)


if __name__ == "__main__":
    unittest.main()
