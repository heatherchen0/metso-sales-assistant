"""Batch pricing composes the real dimensional/provenance arithmetic guard."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_execution import price_items
from metso_assistant.services import parse_text
from metso_assistant.store import build_store


class ServicePricingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def workspace(self, *, second_currency="USD", invalid_second=False, same_item=False):
        records = []
        for identifier, component, rate, amount, currency in (
            ("FSR-9971", "Quartz injectors", "10.25", "20.50", "USD"),
            ("FSR-9972", "Quartz injectors" if same_item else "Zircon locking pins", "7.50",
             "1.00" if invalid_second else "15.00", second_currency),
        ):
            records.append(parse_text(f"""Reference: {identifier}
Site: Invented component facility, Chile
Equipment: Synthetic separator
| Item | Quantity | Unit Rate ({currency}) | Total ({currency}) |
|------|----------|-----------------------|------------------|
| {component} | 2 | {rate}/ea | {amount} |
""", source_path=f"invented/{identifier}.txt"))
        store = self.root / "prepared.sqlite"
        build_store(store, [], [], [], records)
        workspace = ServiceWorkspace(store, self.root / "observations.sqlite", inputs=[
            {"id": "U001", "label": "Requested Quartz injectors", "value": 5, "unit": "each", "user_text": "5 Quartz injectors"},
            {"id": "U002", "label": "Requested Zircon locking pins", "value": 4, "unit": "each", "user_text": "4 Zircon locking pins"}])
        workspace.read_report("FSR-9971"); workspace.read_report("FSR-9972")
        rates = [next(f for f in workspace.facts.values() if f.get("report_id") == identifier and f["kind"] == "unit_price")
                 for identifier in ("FSR-9971", "FSR-9972")]
        return workspace, [{"rate_ref": rates[0]["id"], "quantity_ref": "U001"},
                           {"rate_ref": rates[1]["id"], "quantity_ref": "U001" if same_item else "U002"}]

    def test_distinct_same_currency_components_have_an_exact_supported_subtotal(self):
        workspace, items = self.workspace()
        before = workspace.store_path.read_bytes()
        result = price_items(workspace, items)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["data"]["complete_for_selected_items"])
        subtotal = workspace.facts[result["data"]["subtotal_refs_by_currency"]["USD"]]
        self.assertEqual(subtotal["value"], Decimal("81.25"))
        self.assertEqual(len(subtotal["parents"]), 2)
        self.assertEqual({source["report_id"] for source in result["sources"]}, {"FSR-9971", "FSR-9972"})
        self.assertEqual(workspace.store_path.read_bytes(), before)
        self.assertTrue(all(fact["kind"] == "calculation" and fact["assumptions"] for fact in result["facts"]))

    def test_currency_groups_remain_separate_and_are_never_converted(self):
        workspace, items = self.workspace(second_currency="EUR")
        result = price_items(workspace, items)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["data"]["subtotal_refs_by_currency"]), {"USD", "EUR"})
        amounts = {currency: workspace.facts[identifier]["value"] for currency, identifier
                   in result["data"]["subtotal_refs_by_currency"].items()}
        self.assertEqual(amounts, {"USD": Decimal("51.25"), "EUR": Decimal("30.00")})
        self.assertTrue(all(f["scope"]["operation"] == "multiply" for f in result["facts"]))

    def test_invalid_price_preserves_supported_partial_item_without_claiming_a_complete_subtotal(self):
        workspace, items = self.workspace(invalid_second=True)
        result = price_items(workspace, items)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["data"]["complete_for_selected_items"])
        self.assertEqual(len([item for item in result["data"]["items"] if "error" in item]), 1)
        self.assertTrue(all(f["scope"]["operation"] != "sum" for f in result["facts"]))
        partial = workspace.facts[result["data"]["subtotal_refs_by_currency"]["USD"]]
        self.assertEqual(partial["value"], Decimal("51.25"))
        self.assertIn("excluded", result["data"]["exclusions"])

    def test_repeated_pairs_are_rejected_before_mutating_the_calculation_ledger(self):
        workspace, items = self.workspace()
        with patch.object(workspace, "calculate", wraps=workspace.calculate) as calculate:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                price_items(workspace, [items[0], items[0]])
        calculate.assert_not_called()

    def test_alternative_prices_for_the_same_requested_component_cannot_be_added(self):
        workspace, items = self.workspace(same_item=True)
        result = price_items(workspace, items)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["data"]["complete_for_selected_items"])
        self.assertTrue(result["data"]["sum_errors"])
        self.assertEqual(result["data"]["subtotal_refs_by_currency"], {})
        self.assertEqual(len(result["facts"]), 2)
        self.assertTrue(all(f["scope"]["operation"] == "multiply" for f in result["facts"]))

    def test_malformed_or_unknown_references_do_not_create_prices(self):
        workspace, items = self.workspace()
        for bad in (None, [], [{"rate_ref": 10, "quantity_ref": "U001"}],
                    [{"rate_ref": items[0]["rate_ref"], "literal_quantity": 5}]):
            with self.subTest(bad=str(bad)):
                with self.assertRaises(ValueError): price_items(workspace, bad)
        result = price_items(workspace, [{"rate_ref": "made-up", "quantity_ref": "U001"}])
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["facts"])
        self.assertFalse(result["data"]["subtotal_refs_by_currency"])

    def repeated_workspace(self, text="Two machines need 3 sets of quartz injectors per machine."):
        record = parse_text("""Reference: FSR-9981
Site: Invented component facility, Chile
Equipment: Synthetic separator
| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Quartz injector sets | 2 sets | 7.15/set | 14.30 |
""", source_path="invented/FSR-9981.txt")
        store = self.root / "repeated-prepared.sqlite"
        build_store(store, [], [], [], [record])
        workspace = ServiceWorkspace(store, self.root / "repeated-observations.sqlite", inputs=[
            {"id": "U001", "label": "Quartz injector sets per machine", "value": 3, "unit": "set", "user_text": text,
             "scope": {"role": "requested_quantity", "turn": 1}},
            {"id": "U002", "label": "Machines", "value": 2, "unit": "each", "user_text": text,
             "scope": {"role": "requested_quantity", "turn": 1}}])
        workspace.read_report("FSR-9981")
        rate = next(f for f in workspace.facts.values() if f["kind"] == "unit_price")
        return workspace, {"rate_ref": rate["id"], "quantity_ref": "U001", "repeat_ref": "U002"}

    def test_repeated_quantity_composition_preserves_literal_lineage_and_dimensional_price(self):
        workspace, item = self.repeated_workspace()
        result = price_items(workspace, [item])
        self.assertEqual(result["status"], "ok")
        quantity = next(f for f in result["facts"] if f["kind"] == "request_quantity")
        self.assertEqual((quantity["value"], quantity["unit"], quantity["parents"]), (Decimal(6), "set", ["U001", "U002"]))
        self.assertEqual(quantity["source_ids"], [])
        self.assertEqual(result["data"]["items"][0]["effective_quantity_ref"], quantity["id"])
        subtotal = workspace.facts[result["data"]["subtotal_refs_by_currency"]["USD"]]
        self.assertEqual(subtotal["value"], Decimal("42.90"))
        self.assertEqual(subtotal["parents"], [item["rate_ref"], quantity["id"]])
        self.assertTrue(set(quantity["assumptions"]) <= set(subtotal["assumptions"]))
        self.assertEqual({s["report_id"] for s in result["sources"]}, {"FSR-9981"})

    def test_optional_null_repeat_preserves_legacy_pricing_and_duplicate_detection(self):
        workspace, items = self.workspace()
        result = price_items(workspace, [{**items[0], "repeat_ref": None}])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(workspace.facts[result["data"]["subtotal_refs_by_currency"]["USD"]]["value"], Decimal("51.25"))
        with patch.object(workspace, "calculate", wraps=workspace.calculate) as calculate:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                price_items(workspace, [items[0], {**items[0], "repeat_ref": None}])
        calculate.assert_not_called()

    def test_repeat_without_literal_relation_or_with_unknown_reference_creates_no_price(self):
        workspace, item = self.repeated_workspace("Two machines need 3 sets of quartz injectors.")
        for changed in (item, {**item, "repeat_ref": "invented"}):
            with self.subTest(item=str(changed)):
                result = price_items(workspace, [changed])
                self.assertEqual(result["status"], "partial")
                self.assertFalse(result["facts"])
                self.assertFalse(result["data"]["subtotal_refs_by_currency"])
        for repeat in (1, [], {}, ""):
            with self.subTest(repeat=str(repeat)):
                with self.assertRaises(ValueError):
                    price_items(workspace, [{**item, "repeat_ref": repeat}])

    def test_per_item_price_and_its_repeated_aggregate_are_alternatives_not_additional_components(self):
        workspace, item = self.repeated_workspace()
        result = price_items(workspace, [{"rate_ref": item["rate_ref"], "quantity_ref": "U001"}, item])
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["data"]["subtotal_refs_by_currency"])
        self.assertTrue(any("Overlapping" in error for error in result["data"]["sum_errors"]))


if __name__ == "__main__":
    unittest.main()
