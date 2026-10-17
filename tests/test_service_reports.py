"""Report-level discovery with invented countries, components and tasks."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest

from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_reports import find_reports
from metso_assistant.services import parse_text
from metso_assistant.store import build_store


class ServiceReportDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def report(self, number, country="Asterland", tasks=("Software algorithm update",), equipment="Invented control rack"):
        identifier = f"FSR-{number:04}"
        task_rows = "\n".join(f"| {task} | 1 |" for task in tasks)
        text = f"""Reference: {identifier}
Site: Imaginary facility, {country}
Equipment: {equipment}
On-site hours: {len(tasks)}
| Task | Time Spent (hrs) |
|------|------------------|
{task_rows}
| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Imaginary rack widgets | 2 | 7.15/ea | 14.30 |
Total Cost: USD 14.30
"""
        path = self.root / f"{identifier}.txt"
        path.write_bytes(text.encode("utf-8"))
        record = parse_text(text, source_path=str(path))
        # Invented countries are literal test-source metadata; the legacy
        # parser recognizes only its supplied corpus's country vocabulary.
        record["country"] = country
        record["field_provenance"]["country"] = record["field_provenance"]["site"]
        return record

    def workspace(self, records):
        path = self.root / "prepared.sqlite"
        build_store(path, [], [], [], records)
        return ServiceWorkspace(path, self.root / "sidecar.sqlite")

    def test_named_country_comparison_finds_distinct_reports_and_excludes_others(self):
        workspace = self.workspace([self.report(9001), self.report(9002, "Borealia"), self.report(9003, "Cedarland")])
        result = find_reports(workspace, "Compare algorithm updates in Asterland and Borealia")
        self.assertNotIn("error", result)
        candidates = result["data"]["report_candidates"]
        self.assertEqual({item["country"] for item in candidates}, {"Asterland", "Borealia"})
        self.assertEqual({item["report_id"] for item in candidates}, {"FSR-9001", "FSR-9002"})
        self.assertEqual(result["data"]["requested_countries"], ["Asterland", "Borealia"])

    def test_report_slots_are_not_consumed_by_one_countrys_many_observations(self):
        records = [self.report(number, "Asterland") for number in range(9001, 9011)]
        records.append(self.report(9011, "Borealia"))
        result = find_reports(self.workspace(records), "Algorithm updates Asterland Borealia")
        candidates = result["data"]["report_candidates"]
        self.assertLessEqual(len(candidates), 8)
        self.assertEqual(len({item["report_id"] for item in candidates}), len(candidates))
        self.assertIn("Borealia", {item["country"] for item in candidates})
        self.assertFalse(result["data"]["complete"])
        self.assertTrue(any("incomplete" in warning for warning in result["warnings"]))

    def test_coverage_accumulates_across_each_reports_task_labels(self):
        workspace = self.workspace([self.report(9001, tasks=("Firmware inspection",)),
                                    self.report(9002, tasks=("Firmware inspection", "Sensor tuning"))])
        result = find_reports(workspace, "Firmware sensor tuning")
        self.assertEqual(result["data"]["report_candidates"][0]["report_id"], "FSR-9002")

    def test_discovery_suppresses_label_numbers_and_never_registers_price_facts(self):
        workspace = self.workspace([self.report(9001, tasks=("Update rack 42 module at 12.5 volts",))])
        before_facts, before_sources = deepcopy(workspace.facts), deepcopy(workspace.sources)
        result = find_reports(workspace, "rack module update")
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["sources"], [])
        self.assertEqual(workspace.facts, before_facts)
        self.assertEqual(workspace.sources, before_sources)
        labels = result["data"]["report_candidates"][0]["matching_labels"]
        self.assertTrue(labels)
        self.assertTrue(all(not re.search(r"\d", label) for label in labels))
        self.assertNotIn("7.15", json.dumps(result))
        self.assertNotIn("14.30", json.dumps(result))

    def test_report_inspection_after_discovery_still_owns_numeric_authority(self):
        workspace = self.workspace([self.report(9001)])
        result = find_reports(workspace, "algorithm update")
        identifier = result["data"]["report_candidates"][0]["report_id"]
        self.assertEqual(workspace.facts, {})
        inspected = workspace.read_report(identifier)
        self.assertTrue(inspected["facts"])
        self.assertTrue(workspace.facts)

    def test_requested_country_without_task_matches_is_explicit_not_replaced(self):
        workspace = self.workspace([self.report(9001, "Asterland", tasks=("Algorithm update",)),
                                    self.report(9002, "Borealia", tasks=("Bearing inspection",)),
                                    self.report(9003, "Cedarland", tasks=("Algorithm update",))])
        result = find_reports(workspace, "Algorithm updates Asterland Borealia")
        self.assertEqual({item["country"] for item in result["data"]["report_candidates"]}, {"Asterland"})
        self.assertEqual(result["data"]["missing_requested_countries"], ["Borealia"])

    def test_longer_catalogue_country_name_does_not_select_its_substring_country(self):
        workspace = self.workspace([self.report(9001, "Asterland"), self.report(9002, "North Asterland")])
        result = find_reports(workspace, "Algorithm update North Asterland")
        self.assertEqual(result["data"]["requested_countries"], ["North Asterland"])
        self.assertEqual({item["country"] for item in result["data"]["report_candidates"]}, {"North Asterland"})

    def test_country_matching_is_case_insensitive_and_requires_whole_name(self):
        workspace = self.workspace([self.report(9001, "Asterland"), self.report(9002, "Borealia")])
        result = find_reports(workspace, "Algorithm update ASTERLAND")
        self.assertEqual(result["data"]["requested_countries"], ["Asterland"])
        unrecognized = find_reports(workspace, "Algorithm update NotAsterland")
        self.assertEqual(unrecognized["data"]["requested_countries"], [])

    def test_source_and_store_bytes_remain_unchanged_and_order_is_stable(self):
        workspace = self.workspace([self.report(9002, "Borealia"), self.report(9001)])
        paths = [workspace.store_path, workspace.index_path, *self.root.glob("*.txt")]
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        first = find_reports(workspace, "algorithm updates")
        second = find_reports(workspace, "algorithm updates")
        self.assertEqual(first, second)
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})

    def test_tampered_sidecar_metadata_or_labels_cannot_supply_candidates(self):
        for mutation in ("metadata", "labels"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                old_root, self.root = self.root, Path(temporary)
                workspace = self.workspace([self.report(9001)])
                with closing(sqlite3.connect(workspace.index_path)) as connection:
                    if mutation == "metadata":
                        connection.execute("UPDATE metadata SET value='wrong' WHERE key='content_sha256'")
                    else:
                        row = connection.execute("SELECT id,json FROM observations LIMIT 1").fetchone()
                        fact = json.loads(row[1]); fact["label"] = "Invented algorithm price"
                        connection.execute("UPDATE observations SET json=? WHERE id=?", (json.dumps(fact), row[0]))
                    connection.commit()
                result = find_reports(workspace, "algorithm")
                self.assertIn("error", result)
                self.assertEqual(result["data"]["report_candidates"], [])
                self.assertEqual(workspace.facts, {})
                self.root = old_root

    def test_changed_original_source_fails_closed_before_discovery(self):
        workspace = self.workspace([self.report(9001)])
        source = self.root / "FSR-9001.txt"
        source.write_text(source.read_text(encoding="utf-8") + "Changed source", encoding="utf-8")
        result = find_reports(workspace, "algorithm")
        self.assertIn("error", result)
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["data"]["report_candidates"], [])

    def test_invalid_and_no_match_queries_have_no_numeric_authority(self):
        workspace = self.workspace([self.report(9001)])
        for query in ("", None, "x" * 6001):
            self.assertIn("error", find_reports(workspace, query))
        absent = find_reports(workspace, "Nonexistent turbine blades")
        self.assertEqual(absent["data"]["report_candidates"], [])
        self.assertEqual(absent["facts"], [])
        self.assertFalse(absent["data"]["complete"])


if __name__ == "__main__":
    unittest.main()
