"""Checked service presentation uses the real fact/calculation registry."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from metso_assistant.service_answer import service_answer
from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_execution import price_items
from metso_assistant.service_reports import find_reports
from metso_assistant.services import parse_text
from metso_assistant.store import build_store


class ServiceAnswerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def workspace(self, *, second_currency="USD", same_item=False, ambiguous=False, alternative=False):
        records = []
        rows = [("FSR-9961", "Quartz injectors", Decimal("10.25"), "USD", "Chile", "Orion Experimental Separator"),
                ("FSR-9962", "Quartz injectors" if same_item else "Zircon locking pins", Decimal("7.50"), second_currency, "Finland", "SALA VSP-750 experimental pump")]
        if alternative:
            rows.append(("FSR-9963", "Quartz injectors", Decimal("8.50"), "USD", "Canada", "Orion Experimental Separator"))
        if ambiguous:
            rows = [("FSR-9961", "Synthetic protective fluid", Decimal(24), "AUD", "Chile", "Orion Experimental Separator")]
        for identifier, component, rate, currency, country, equipment in rows:
            quantity, amount = ("150 ml", Decimal(24)) if ambiguous else ("2", rate * 2)
            rate_text = str(rate) if ambiguous else str(rate) + "/ea"
            total = amount + Decimal(165)
            records.append(parse_text(f"""Reference: {identifier}
Site: Invented test facility, {country}
Equipment: {equipment}
| Item | Quantity | Unit Rate ({currency}) | Total ({currency}) |
|------|----------|-----------------------|------------------|
| {component} | {quantity} | {rate_text} | {amount} |
| Labour | 3 hrs | 50/hr | 150 |
| Travel | 1 hr | 15/hr | 15 |

Total Cost: {currency} {total}
""", source_path=f"invented/{identifier}.txt"))
        store = self.root / "prepared.sqlite"
        build_store(store, [], [], [], records)
        workspace = ServiceWorkspace(store, self.root / "observations.sqlite", inputs=[
            {"id": "U001", "label": "Requested Quartz injectors", "value": 5, "unit": "each", "user_text": "5 Quartz injectors"},
            {"id": "U002", "label": "Requested Zircon locking pins", "value": 4, "unit": "each", "user_text": "4 Zircon locking pins"},
            {"id": "U003", "label": "Requested protective fluid", "value": 300, "unit": "ml", "user_text": "300 ml of protective fluid"}])
        for record in records:
            workspace.read_report(record["report_id"])
        return workspace

    def rate(self, workspace, report):
        return next(f for f in workspace.facts.values() if f.get("report_id") == report
                    and f["kind"] == "unit_price")

    def multiply(self, workspace, report, quantity):
        return workspace.calculate("multiply", [self.rate(workspace, report)["id"], quantity])

    def render(self, workspace, history=None):
        return service_answer(workspace, history or [], ["Price the requested materials only."], {})

    def test_one_component_uses_checked_material_amount_and_excludes_original_invoice(self):
        workspace = self.workspace()
        amount = self.multiply(workspace, "FSR-9961", "U001")["facts"][0]
        result, extra = self.render(workspace)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["response"]["scope"], "partial_components")
        self.assertEqual(result["data"]["subtotal_refs_by_currency"], {"USD": amount["id"]})
        self.assertEqual(workspace.facts[amount["id"]]["value"], Decimal("51.25"))
        self.assertFalse(any(workspace.facts[i]["kind"] == "visit_total" for i in result["data"]["selected_fact_ids"]))
        self.assertNotIn("185.50", result["data"]["summary"])
        self.assertTrue(result["sources"])
        self.assertFalse(extra)

    def test_two_distinct_materials_sum_exactly_without_using_a_historical_invoice(self):
        workspace = self.workspace()
        first = self.multiply(workspace, "FSR-9961", "U001")["facts"][0]
        second = self.multiply(workspace, "FSR-9962", "U002")["facts"][0]
        result, operations = self.render(workspace)
        subtotal = workspace.facts[result["data"]["subtotal_refs_by_currency"]["USD"]]
        self.assertEqual(subtotal["value"], Decimal("81.25"))
        self.assertEqual(set(subtotal["parents"]), {first["id"], second["id"]})
        self.assertEqual(len(operations), 1)
        self.assertFalse(operations[0]["result"].get("error"))
        self.assertEqual({s["report_id"] for s in result["sources"]}, {"FSR-9961", "FSR-9962"})
        self.assertTrue(all(workspace.facts[i]["kind"] == "calculation" for i in result["data"]["selected_fact_ids"]))

    def test_different_currencies_have_independent_subtotals_and_no_conversion(self):
        workspace = self.workspace(second_currency="EUR")
        self.multiply(workspace, "FSR-9961", "U001"); self.multiply(workspace, "FSR-9962", "U002")
        result, operations = self.render(workspace)
        self.assertEqual(set(result["data"]["subtotal_refs_by_currency"]), {"USD", "EUR"})
        values = {currency: workspace.facts[identifier]["value"] for currency, identifier in result["data"]["subtotal_refs_by_currency"].items()}
        self.assertEqual(values, {"USD": Decimal("51.25"), "EUR": Decimal("30.00")})
        self.assertFalse(operations)
        self.assertIn("exchange rate", result["data"]["summary"])

    def test_ambiguous_volume_price_stays_insufficient_and_never_becomes_per_ml(self):
        workspace = self.workspace(ambiguous=True)
        calculation = self.multiply(workspace, "FSR-9961", "U003")
        self.assertTrue(calculation.get("error"))
        result, operations = self.render(workspace, [{"action": {"action": "calculate"}, "result": calculation}])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["data"]["subtotal_refs_by_currency"])
        self.assertNotIn("/ml", result["data"]["summary"])
        self.assertNotIn("per ml", result["data"]["summary"])
        self.assertNotIn("7,200", result["data"]["summary"])
        self.assertFalse(operations)

    def test_original_visit_comparison_retains_actual_country_equipment_and_totals(self):
        workspace = self.workspace(second_currency="EUR")
        discovery = find_reports(workspace, "Compare Quartz and Zircon reports from Chile and Finland")
        history = [{"action": {"action": "find_reports"}, "result": discovery}]
        for candidate in discovery["data"]["report_candidates"]:
            report = candidate["report_id"]
            history.append({"action": {"action": "read_report", "report_id": report}, "result": workspace.read_report(report)})
        result, operations = self.render(workspace, history)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["response"]["scope"], "historical_comparison")
        selected = [workspace.facts[i] for i in result["data"]["selected_fact_ids"]]
        self.assertEqual({f["scope"]["country"] for f in selected}, {"Chile", "Finland"})
        self.assertTrue(all(f["scope"]["type"] == "reported_total" for f in selected))
        self.assertIn("Orion Experimental Separator", result["data"]["summary"])
        self.assertIn("SALA VSP-750 experimental pump", result["data"]["summary"])
        self.assertEqual({f["currency"]: f["value"] for f in selected}, {"USD": Decimal("185.50"), "EUR": Decimal("180.00")})
        self.assertFalse(result["data"]["subtotal_refs_by_currency"])
        self.assertFalse(operations)

    def test_discovery_does_not_replace_requested_component_pricing_with_invoice(self):
        workspace = self.workspace()
        discovery = find_reports(workspace, "Find Quartz injectors in Chile")
        history = [{"action": {"action": "find_reports"}, "result": discovery},
                   {"action": {"action": "read_report", "report_id": "FSR-9961"}, "result": workspace.read_report("FSR-9961")}]
        result, _ = service_answer(workspace, history, ["Price the requested materials."], plan={
            "purpose": "requested_work", "requested_categories": ["material"]})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["data"]["response"]["scope"], "insufficient")
        self.assertNotIn("185.50", result["data"]["summary"])
        self.assertTrue(all(workspace.facts[ref]["kind"] != "visit_total" for ref in result["data"]["selected_fact_ids"]))

    def test_alternative_prices_for_one_requested_component_are_not_an_overall_total(self):
        workspace = self.workspace(same_item=True)
        self.multiply(workspace, "FSR-9961", "U001"); self.multiply(workspace, "FSR-9962", "U001")
        result, operations = self.render(workspace)
        self.assertFalse(result["data"]["subtotal_refs_by_currency"])
        self.assertTrue(operations[0]["result"].get("error"))
        self.assertEqual(len(result["data"]["selected_fact_ids"]), 2)
        self.assertIn("alternatives", result["data"]["summary"])

    def test_old_subtotal_is_not_reused_as_an_overall_total_after_an_alternative_price(self):
        workspace = self.workspace(alternative=True)
        batch = price_items(workspace, [{"rate_ref": self.rate(workspace, "FSR-9961")["id"], "quantity_ref": "U001"},
                                        {"rate_ref": self.rate(workspace, "FSR-9962")["id"], "quantity_ref": "U002"}])
        self.assertTrue(batch["data"]["complete_for_selected_items"])
        self.multiply(workspace, "FSR-9963", "U001")
        result, _ = self.render(workspace)
        self.assertFalse(result["data"]["subtotal_refs_by_currency"])
        self.assertIn("alternatives", result["data"]["summary"])


if __name__ == "__main__":
    unittest.main()
