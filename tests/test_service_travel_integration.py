"""Country travel cohorts integrate with separately scoped service examples."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from metso_assistant.adaptive import request_facts
from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_execution import execute_service_request
from metso_assistant.service_request import resolve_duration_inputs
from metso_assistant.services import parse_report
from metso_assistant.store import build_store


class CountryTravelIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.serial = 0

    def report(self, identifier, *, country="Chile", currency="USD", equipment="Invented Crusher CR100",
               rate="11", main=False, component_model="HX900", clips_rate=None):
        if main:
            onsite, material, task = "3", f"| {component_model} coupler cartridges | 3 | 15/ea | 45 |", f"| Replace {component_model} coupler cartridges (3 pcs) | 2 |\n| Preparation and testing | 1 |"
            if clips_rate is not None:
                material += f"\n| {component_model} retaining clips | 3 | {clips_rate}/ea | {Decimal(clips_rate)*3} |"
        else:
            onsite, material, task = "1", "| Generic spare | 1 | 4/ea | 4 |", "| Inspect invented equipment | 1 |"
        travel_amount = Decimal(rate) * 2
        text = f"""Reference: {identifier}
Site: Invented facility, {country}
Equipment: {equipment}
On-site hours: {onsite}
Travel hours: 2
| Task | Time Spent (hrs) |
|------|------------------|
{task}

| Item | Quantity | Unit Rate ({currency}) | Total ({currency}) |
|------|----------|-----------------------|------------------|
{material}
| Labour | {onsite} hrs | 50/hr | {Decimal(onsite)*50} |
| Travel | 2 hrs | {rate}/hr | {travel_amount} |
"""
        path = self.root / (identifier + ".txt")
        path.write_text(text, encoding="utf-8")
        return parse_report(path)

    def records(self, *, currency="USD", rates=("11", "17", "23")):
        records = [self.report("FSR-9800", country="Sweden", currency="EUR",
                               equipment="Helix Separator HX900", rate="99", main=True)]
        records += [self.report(f"FSR-{9810+i}", currency=currency, rate=rate,
                                equipment=f"Unrelated Machine CR{100+i}") for i, rate in enumerate(rates)]
        return records

    def execute(self, *, records=None, messages=None, country="Chile", categories=None):
        self.serial += 1
        records = records or self.records()
        store = self.root / f"store-{self.serial}.sqlite"
        build_store(store, [], [], [], records)
        messages = messages or ["A customer in Chile needs 6 HX900 coupler cartridges replaced on a Helix Separator HX900. Total billed travel is 8 hours. What should I budget?"]
        durations = resolve_duration_inputs(messages)
        inputs = [fact for fact in request_facts(messages, "service_cost")
                  if fact.get("scope", {}).get("role") not in {"labour_hours", "travel_hours", "ambiguous_duration"}]
        inputs += [fact for duration in durations if (fact := duration.to_fact("U_duration_" + duration.role))]
        workspace = ServiceWorkspace(store, self.root / f"index-{self.serial}.sqlite", inputs=inputs)
        plan = {"purpose": "requested_work", "requested_categories": categories or ["material", "labour", "travel"],
                "quantity_structure": "direct", "equipment_count_ref": None}
        result, trace = execute_service_request(workspace, plan, messages,
            {"country": country, "equipment": "Helix Separator HX900"}, duration_inputs=durations)
        return result, trace, workspace

    def cohort(self, result, currency="USD"):
        return next(cohort for cohort in result["data"]["travel_context"]["cohorts"] if cohort["currency"] == currency)

    def fact(self, workspace, reference):
        self.assertIsNotNone(reference)
        return workspace.facts[reference]

    def bounds(self, result, *, currency="USD", category=None):
        groups = (result["data"]["category_subtotal_ranges_by_currency"][category] if category else
                  result["data"]["subtotal_ranges_by_currency"])
        return groups[currency]

    def test_all_exact_country_rates_cross_equipment_bound_known_travel(self):
        result, trace, workspace = self.execute()
        context = result["data"]["travel_context"]
        self.assertEqual((context["status"], context["country"]), ("available", "Chile"))
        cohort = self.cohort(result)
        rates = [self.fact(workspace, reference) for reference in cohort["fact_ids"]]
        self.assertEqual({fact["report_id"] for fact in rates}, {"FSR-9810", "FSR-9811", "FSR-9812"})
        self.assertEqual({fact["value"] for fact in rates}, {Decimal(11), Decimal(17), Decimal(23)})
        self.assertTrue(all(fact["scope"]["country"] == "Chile" and fact["scope"]["category"] == "travel" for fact in rates))
        self.assertEqual((cohort["minimum"], cohort["maximum"]), (Decimal(11), Decimal(23)))
        self.assertEqual(self.fact(workspace, cohort["amount_min_ref"])["value"], Decimal("88.00"))
        self.assertEqual(self.fact(workspace, cohort["amount_max_ref"])["value"], Decimal("184.00"))
        quantity = self.fact(workspace, cohort["quantity_ref"])
        self.assertEqual((quantity["value"], quantity["unit"]), (Decimal(8), "hour"))
        self.assertEqual(len(set(context["endpoint_fact_ids"])), 2)
        self.assertFalse(any(step.get("operation") == "adaptive_action_selection" for step in trace))

    def test_foreign_material_and_labour_currency_stays_separate_from_local_travel(self):
        result, _, workspace = self.execute()
        foreign = self.bounds(result, currency="EUR")
        local = self.bounds(result, currency="USD")
        self.assertEqual((foreign["minimum"], foreign["maximum"]), (Decimal("290.00"), Decimal("290.00")))
        self.assertEqual((local["minimum"], local["maximum"]), (Decimal("88.00"), Decimal("184.00")))
        self.assertEqual((self.bounds(result, currency="EUR", category="material")["minimum"],
                          self.bounds(result, currency="EUR", category="service")["minimum"]),
                         (Decimal("90.00"), Decimal("200.00")))
        self.assertNotIn("FSR-9800", self.cohort(result)["reports"])
        self.assertIn("Currencies stay separate", result["data"]["summary"])

    def test_same_currency_combines_each_endpoint_once_with_service_and_material(self):
        result, _, workspace = self.execute(records=self.records(currency="EUR"))
        total, service = self.bounds(result, currency="EUR"), self.bounds(result, currency="EUR", category="service")
        self.assertEqual((total["minimum"], total["maximum"]), (Decimal("378.00"), Decimal("474.00")))
        self.assertEqual((service["minimum"], service["maximum"]), (Decimal("288.00"), Decimal("384.00")))
        low = self.fact(workspace, total["minimum_ref"])
        high = self.fact(workspace, total["maximum_ref"])
        cohort = self.cohort(result, "EUR")
        self.assertNotIn(cohort["amount_max_ref"], low.get("parents", []))
        self.assertNotIn(cohort["amount_min_ref"], high.get("parents", []))

    def test_single_rate_and_duplicate_equal_rates_produce_a_point_not_a_double_charge(self):
        for rates in (("17",), ("17", "17")):
            with self.subTest(rates=rates):
                result, _, workspace = self.execute(records=self.records(rates=rates))
                cohort = self.cohort(result)
                self.assertEqual((cohort["minimum"], cohort["maximum"]), (Decimal(17), Decimal(17)))
                self.assertEqual((self.bounds(result)["minimum"], self.bounds(result)["maximum"]), (Decimal("136.00"), Decimal("136.00")))
                self.assertEqual(len(cohort["fact_ids"]), len(rates))
                self.assertEqual(len(set(result["data"]["travel_context"]["endpoint_fact_ids"])), 1)

    def test_unknown_hours_show_local_rate_context_without_inventing_travel_cost(self):
        message = "A customer in Chile needs 6 HX900 coupler cartridges replaced on a Helix Separator HX900. I do not know travel hours. What can you estimate?"
        result, _, workspace = self.execute(messages=[message])
        cohort = self.cohort(result)
        self.assertIsNone(cohort["quantity_ref"])
        self.assertIsNone(cohort["amount_min_ref"])
        self.assertIsNone(cohort["amount_max_ref"])
        self.assertEqual(result["data"]["travel_context"]["endpoint_fact_ids"], [])
        self.assertNotIn("USD", result["data"]["subtotal_ranges_by_currency"])
        self.assertIn("travel", result["data"]["unpriced_categories"])
        self.assertIn("11.00", result["data"]["summary"])
        self.assertIn("23.00", result["data"]["summary"])
        self.assertIsNone(result["question"])

    def test_missing_or_no_eligible_country_does_not_fall_back_to_foreign_travel(self):
        for country, status in ((None, "missing_country"), ("Russia", "unavailable")):
            with self.subTest(country=country):
                result, _, workspace = self.execute(country=country)
                context = result["data"]["travel_context"]
                self.assertEqual(context["status"], status)
                self.assertEqual(context["cohorts"], [])
                self.assertEqual(context["endpoint_fact_ids"], [])
                self.assertIn("travel", result["data"]["unpriced_categories"])
                self.assertNotIn("USD", result["data"]["subtotal_ranges_by_currency"])
                self.assertFalse(any(fact.get("scope", {}).get("operation") == "multiply" and
                    any(str(token).startswith("billed:travel_hours:") for token in fact.get("scope", {}).get("coverage", []))
                    for fact in workspace.facts.values()))

    def test_country_cohorts_preserve_separate_currencies(self):
        records = self.records(rates=("11", "17"))
        records.append(self.report("FSR-9820", rate="23", currency="CAD", equipment="Other Machine YY700"))
        result, _, _ = self.execute(records=records)
        self.assertEqual({cohort["currency"] for cohort in result["data"]["travel_context"]["cohorts"]}, {"USD", "CAD"})
        self.assertEqual((self.bounds(result, currency="USD")["minimum"], self.bounds(result, currency="USD")["maximum"]),
                         (Decimal("88.00"), Decimal("136.00")))
        self.assertEqual((self.bounds(result, currency="CAD")["minimum"], self.bounds(result, currency="CAD")["maximum"]),
                         (Decimal("184.00"), Decimal("184.00")))

    def test_quantity_correction_keeps_model_source_and_discloses_estimated_labour(self):
        first = "A customer in Chile needs 6 HX900 coupler cartridges replaced on a Helix Separator HX900. Total billed travel is 8 hours. What should I budget?"
        result, _, workspace = self.execute(messages=[first, "Actually make that 9 cartridges, not 6."])
        material = self.bounds(result, currency="EUR", category="material")
        labour = self.bounds(result, currency="EUR", category="service")
        self.assertEqual((material["minimum"], material["maximum"]), (Decimal("135.00"), Decimal("135.00")))
        self.assertEqual((labour["minimum"], labour["maximum"]), (Decimal("300.00"), Decimal("300.00")))
        self.assertIn("estimated", result["data"]["summary"].casefold())
        selected = [workspace.facts[reference] for reference in result["data"]["selected_fact_ids"]]
        self.assertTrue(any("FSR-9800" in {workspace.sources[source]["report_id"] for source in fact.get("source_ids", [])}
                            for fact in selected if fact.get("currency") == "EUR"))

    def test_new_material_keeps_literal_model_source_while_local_travel_uses_other_machines(self):
        records = self.records()
        records[0] = self.report("FSR-9800", country="Sweden", currency="EUR", equipment="Helix Separator HX900",
                                  rate="99", main=True, clips_rate="4")
        records.append(self.report("FSR-9805", country="Chile", currency="USD", equipment="Helix Separator HX800",
                                   rate="31", main=True, component_model="HX800", clips_rate="2"))
        first = "A customer in Chile needs 6 HX900 coupler cartridges replaced on a Helix Separator HX900. Total billed travel is 8 hours. What should I budget?"
        result, _, workspace = self.execute(records=records, messages=[first, "Also include 2 retaining clips. Give the material subtotal separately."])
        material = self.bounds(result, currency="EUR", category="material")
        self.assertEqual((material["minimum"], material["maximum"]), (Decimal("98.00"), Decimal("98.00")))
        self.assertNotIn("USD", result["data"]["category_subtotal_ranges_by_currency"]["material"])
        # The extra unrelated Chilean machine legitimately contributes travel.
        cohort = self.cohort(result)
        self.assertIn("FSR-9805", cohort["reports"])
        self.assertEqual(cohort["maximum"], Decimal(31))


if __name__ == "__main__":
    unittest.main()
