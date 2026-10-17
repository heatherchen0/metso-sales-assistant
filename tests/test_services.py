"""Independent invented fixtures; no historical report prices or answers."""
import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from metso_assistant.services import ingest_services, json_numbers, number, parse_report, parse_text, validate_record


class ServiceNumericTests(unittest.TestCase):
    def test_exact_decimal_json_numbers_and_missing_values(self):
        value = {"amount": Decimal("9000000000000000.1234"), "quantity": Decimal("0.3"), "missing": None}
        payload = json_numbers(value)
        loaded = json.loads(payload, parse_float=Decimal)
        self.assertEqual(loaded, value)
        self.assertNotIn('"9000000000000000.1234"', payload)
        self.assertIsNone(number("1,23"))
        self.assertEqual(number("1,234.50"), Decimal("1234.50"))

    def test_currency_is_not_inferred_from_country(self):
        report = parse_text("Customer: Invented Mine\nSite: Fictional Site, Australia\nTotal cost for this visit: $43.70\n")
        self.assertEqual(report["country"], "Australia")
        self.assertEqual(report["cost"]["reported_total"]["amount"], Decimal("43.70"))
        self.assertIsNone(report["cost"]["reported_total"]["currency"])
        self.assertIn("currency_unspecified", {w["code"] for w in report["warnings"]})

    def test_explicit_codes_and_euro_symbol_are_resolved(self):
        for raw, code in [("CAD $43.70", "CAD"), ("$43.70 USD", "USD"), ("43.70 EUR", "EUR"), ("€43.70", "EUR")]:
            with self.subTest(raw=raw):
                report = parse_text(f"Total cost for this visit: {raw}\n")
                self.assertEqual(report["cost"]["reported_total"]["currency"], code)

    def test_exact_line_sum_and_vat_are_separate_from_printed_total(self):
        report = parse_text("""Reference: FSR-9001
| Item | Quantity | Unit Rate (EUR) | Total (EUR) |
|------|----------|-----------------|-------------|
| Labour | 2.35 | 11.20/hr | 26.32 |
| Travel | 0.2 | 7.15/hr | 1.43 |
| Filter cloths | 7 | 0.35/ea | 2.45 |
| Subtotal | | | 30.20 |
| VAT (10%) | | | 3.02 |
| Total Amount Due | | | 33.22 |
""")
        self.assertEqual(report["cost"]["reported_subtotal"]["amount"], Decimal("30.20"))
        self.assertEqual(report["cost"]["reported_total"]["amount"], Decimal("33.22"))
        self.assertEqual(report["cost"]["tax_basis"], "inclusive_vat")
        self.assertTrue(all(c["status"] == "match" for c in report["cost"]["checks"]))
        self.assertEqual(report["typed_validation_errors"], [])

    def test_bad_line_and_printed_total_are_reported_without_repairs(self):
        report = parse_text("""| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Labour | 2 | 6/hr | 11 |
| Total | | | 10 |
""")
        self.assertEqual(report["cost"]["line_items"][0]["amount"], Decimal(11))
        self.assertEqual(report["cost"]["reported_total"]["amount"], Decimal(10))
        codes = {w["code"] for w in report["warnings"]}
        self.assertTrue({"line_amount_mismatch", "printed_total_mismatch"} <= codes)

    def test_zero_charge_is_distinct_from_missing_rate_and_amount(self):
        report = parse_text("""Parts supplied from customer's on-site inventory; no charge.
| Item | Quantity | Rate (USD) | Total (USD) |
|------|----------|------------|-------------|
| Gasket | 3 | N/A | 0 |
| Spare clip | 2 | N/A | - |
""")
        zero, missing = report["cost"]["line_items"]
        self.assertEqual(zero["amount"], Decimal(0))
        self.assertIsNone(zero["unit_rate"])
        self.assertTrue(zero["customer_supplied"])
        self.assertIsNone(missing["amount"])
        self.assertIsNone(report["cost"]["reported_total"])

    def test_truncated_cost_row_keeps_computed_amount_separate(self):
        report = parse_text("""| Item | Quantity | Unit Rate (EUR) | Total (EUR) |
|------|----------|-----------------|-------------|
| Clip | 3 | €0.17/ea | €
""")
        item = report["cost"]["line_items"][0]
        self.assertIsNone(item["amount"])
        self.assertEqual(report["cost"]["checks"][0]["computed_amount"], Decimal("0.51"))
        self.assertIsNone(report["cost"]["reported_total"])
        self.assertFalse(report["cost"]["line_item_sum"]["complete"] if report["cost"]["line_item_sum"] else False)

    def test_missing_physical_rate_denominator_does_not_make_unit_price(self):
        report = parse_text("""| Item | Quantity | Rate (USD) | Amount (USD) |
|------|----------|------------|--------------|
| Oil | 125 ml | $13 | $13 |
""")
        self.assertIsNone(report["cost"]["line_items"][0]["rate_unit"])
        self.assertEqual(report["cost"]["checks"][0]["status"], "ambiguous_rate_basis")
        self.assertEqual(report["cost"]["line_items"][0]["amount"], Decimal(13))

    def test_rate_currency_mismatch_is_not_converted(self):
        report = parse_text("""| Item | Quantity | Unit Rate | Total (CAD) |
|------|----------|-----------|-------------|
| Labour | 2 | USD 6/hr | CAD 12 |
""")
        self.assertEqual(report["cost"]["checks"][0]["status"], "currency_mismatch")


class ServiceScopeTests(unittest.TestCase):
    def test_ranges_and_activity_times_do_not_become_visit_totals(self):
        report = parse_text("Job took 7-8h on site.\nTravel: 4h each way.\n")
        self.assertIsNone(report["hours"]["on_site"]["value"])
        self.assertEqual(report["hours"]["on_site"]["range"], {"lower": Decimal(7), "upper": Decimal(8)})
        self.assertIsNone(report["hours"]["travel"]["value"])
        self.assertEqual(report["hours"]["travel"]["status"], "ambiguous")
        activity = parse_text("Total commissioning 3h, plus half a day calibrating.\n")
        self.assertIsNone(activity["hours"]["on_site"]["value"])
        self.assertTrue(activity["hours"]["context_notes"])

    def test_conflicting_hour_values_are_retained_and_not_selected(self):
        report = parse_text("On-site hours: 4.1\nTotal on-site hours: 5.3\n")
        self.assertEqual(report["hours"]["on_site"]["status"], "ambiguous")
        self.assertEqual(len(report["hours"]["on_site"]["candidates"]), 2)

    def test_vague_each_way_context_is_flagged_beside_declared_total(self):
        report = parse_text("Travel was like 2, maybe 2.5h each way, but hotel nearby this time.\nTotal time: 5.7 onsite, 1.9 travel.\n")
        self.assertEqual(report["hours"]["travel"]["value"], Decimal("1.9"))
        self.assertIn("travel_vague_each_way_alternative", {w["code"] for w in report["warnings"]})

    def test_approximation_does_not_bleed_into_neighboring_hours(self):
        report = parse_text("Total onsite ~7.7h, travel 3.3h.\n")
        self.assertTrue(report["hours"]["on_site"]["approximate"])
        self.assertFalse(report["hours"]["travel"]["approximate"])

    def test_cloth_replacement_on_plates_does_not_create_cloth_count(self):
        report = parse_text("Changed out filter cloths on 7 plates.\nReplaced cloths w/ spares (they had).\n")
        self.assertEqual(report["replacement_quantities"], [])
        count = report["service_quantities"][0]
        self.assertEqual((count["item"], count["quantity"], count["quantity_unit"]), ("filter_plate", Decimal(7), "plate"))
        self.assertTrue(count["customer_supplied"])
        self.assertEqual(count["customer_supply_scope"], "replacement_cloths")
        self.assertIn("cloth_count_not_stated", {w["code"] for w in report["warnings"]})

    def test_inspected_and_replaced_cloth_counts_keep_distinct_activity_scope(self):
        report = parse_text("Filter cloth inspection (all chambers, 99 pcs).\nFilter cloth replacement (3 pcs).\nReplaced 2/9 filter cloths; only partial cloth change.\n")
        self.assertEqual([q["quantity"] for q in report["replacement_quantities"]], [Decimal(3), Decimal(2)])
        inspection = next(q for q in report["service_quantities"] if q["activity"] == "inspection")
        self.assertEqual(inspection["quantity"], Decimal(99))
        partial = report["replacement_quantities"][-1]
        self.assertEqual(partial["planned_or_inspected_total"], Decimal(9))

    def test_task_total_mismatch_preserves_reported_hours(self):
        report = parse_text("""| Task | Time Spent (hrs) |
|------|------------------|
| Inspection | 0.2 |
| Reassembly | 0.3 |
| Total On-site Hours | 0.6 |
""")
        self.assertEqual(report["hours"]["on_site"]["value"], Decimal("0.6"))
        self.assertEqual(report["hours"]["task_sum_check"]["status"], "mismatch")

    def test_model_digits_dates_and_clock_times_are_not_quantities(self):
        report = parse_text("Equip: MD901 pump\nDate: 3/4/24\nArrived 07:30, left 17:20.\nMaterials/Parts Used:\n- Hydraulic oil (Product M 83)\n")
        self.assertIsNone(report["hours"]["on_site"]["value"])
        self.assertIsNone(report["visit_date"]["value"])
        self.assertIsNone(report["materials"][0]["quantity"])


class ServiceArtifactTests(unittest.TestCase):
    def test_utf8_provenance_hash_reproducibility_and_public_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "data" / "HistoricalServices"; data.mkdir(parents=True)
            report = data / "FSR-9002.txt"
            source = "Reference: FSR-9002\nCustomer: Élodie Synthetic\nSite: Fictional, Chile\nTotal Cost: EUR 0.30\n"
            report.write_bytes(source.encode("utf-8"))
            parsed = parse_report(report)
            self.assertEqual(parsed["source"]["sha256"], hashlib.sha256(source.encode("utf-8")).hexdigest())
            self.assertEqual(parsed["customer"], "Élodie Synthetic")
            self.assertEqual(parsed["cost"]["reported_total"]["evidence"], [{"line_start": 4, "line_end": 4}])
            first = ingest_services(root / "data", root / "output")
            payload = (root / "output" / "records.jsonl").read_bytes()
            second = ingest_services(root / "data", root / "output")
            self.assertEqual(payload, (root / "output" / "records.jsonl").read_bytes())
            self.assertEqual(first["manifest"]["records_sha256"], second["manifest"]["records_sha256"])
            aggregate = json.dumps(first["quality"])
            self.assertNotIn("Élodie", aggregate)
            self.assertNotIn("0.30", aggregate)
            self.assertEqual(first["quality"]["invalid_typed_records"], 0)

    def test_type_audit_rejects_numeric_strings_boolean_quantity_and_bad_refs(self):
        record = parse_text("On-site hours: 1.1\n")
        record["hours"]["on_site"]["value"] = "1.1"
        record["service_quantities"] = [{"quantity": True, "evidence": [{"line_start": 0, "line_end": 3}]}]
        errors = validate_record(record)
        self.assertTrue(any("numeric field" in e for e in errors))
        self.assertTrue(any("boolean" in e for e in errors))
        self.assertTrue(any("line range" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
