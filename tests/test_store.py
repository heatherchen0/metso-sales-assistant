"""Offline store tests using independently invented content and quantities."""
from copy import deepcopy
from contextlib import closing
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from metso_assistant.services import parse_text
from metso_assistant.store import KnowledgeStore, build_store, query_facts, query_services, search_text, validate_store


def fixtures():
    docs = [{"id": "synthetic-a", "title": "Invented control product", "source_file": "invented-a.pdf", "source_sha256": "a" * 64, "page_count": 3},
            {"id": "synthetic-b", "title": "Invented pump product", "source_file": "invented-b.pdf", "source_sha256": "b" * 64, "page_count": 3}]
    chunks = [{"id": "context-a", "doc_id": "synthetic-a", "page": 1,
               "text": "Moisture control responds to changing feed composition.", "kind": "narrative",
               "source_sha256": "a" * 64, "citation": {"source_file": "invented-a.pdf", "page": 1}, "evidence_status": "source_text"},
              {"id": "caption-a", "doc_id": "synthetic-a", "page": 2,
               "text": "Moisture feedback control schematic.", "kind": "figure_caption",
               "source_sha256": "a" * 64, "citation": {"source_file": "invented-a.pdf", "page": 2}, "evidence_status": "source_text"},
              {"id": "raw-a", "doc_id": "synthetic-a", "page": 2,
               "text": "SecretDimension width 37.19 mm.", "kind": "table_context",
               "source_sha256": "a" * 64, "citation": {"source_file": "invented-a.pdf", "page": 2}, "evidence_status": "raw_table"},
              {"id": "context-b", "doc_id": "synthetic-b", "page": 1,
               "text": "Pump bearing lubrication and seal inspection.", "kind": "narrative",
               "source_sha256": "b" * 64, "citation": {"source_file": "invented-b.pdf", "page": 1}, "evidence_status": "source_text"}]
    base = {"product_family": "InventedPress", "configuration": "P3", "parameter": "height", "value": 37.19,
            "value_kind": "scalar", "unit": "mm", "status": "reported", "review_status": "verified", "issues": [],
            "source_refs": [{"file": "invented-a.pdf", "sha": "a" * 64, "page": 2, "table_id": "table-01", "locator": {"row": 1, "column": 2}}]}
    facts = [dict(deepcopy(base), fact_id="height"),
             dict(deepcopy(base), fact_id="pending-width", parameter="width", review_status="pending"),
             dict(deepcopy(base), fact_id="ambiguous", status="ambiguous", value=None),
             dict(deepcopy(base), fact_id="invalid", status="invalid", value=None),
             dict(deepcopy(base), fact_id="missing", status="missing", value=None),
             dict(deepcopy(base), fact_id="zero", parameter="zero-component", value=0),
             dict(deepcopy(base), fact_id="vector", parameter="plate_dimensions", value=None, value_kind="vector", value_components=[17, 29]),
             dict(deepcopy(base), fact_id="range", parameter="operating_range", value=None, value_kind="range", value_range={"lower": 0.7, "upper": 2.3}),
             dict(deepcopy(base), fact_id="blocked", review_blockers=[{"code": "unresolved_header", "severity": "blocker"}])]
    tables = [{"doc_id": "synthetic-a", "table_id": "table-01", "source": {"file": "invented-a.pdf", "sha": "a" * 64, "page": 2},
               "product_family": "InventedPress", "review_status": "pending", "facts": facts, "issues": [],
               "normalized_table": {"rows": [["height", "37.19"]]}}]
    services = [parse_text("Reference: FSR-9003\nSite: Invented site, Chile\nOn-site hours: 2.7\nTotal Cost: EUR 900000000000.1234\n")]
    return docs, chunks, tables, services


class StoreRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "store with spaces.sqlite"
        self.inputs = fixtures()
        self.counts = build_store(self.path, *self.inputs)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fts_finds_relevant_text_with_bm25_and_citations(self):
        hits = search_text(self.path, "moisture control")
        self.assertEqual({h["id"] for h in hits}, {"context-a", "caption-a"})
        self.assertTrue(all(h["citation"]["source_file"] == "invented-a.pdf" for h in hits))
        self.assertEqual([h["score"] for h in hits], sorted(h["score"] for h in hits))

    def test_user_operators_and_punctuation_cannot_change_query_structure(self):
        for query in ['moisture" OR text:* NEAR(pump) --', "' OR 1=1; DROP TABLE facts; --", "***\"()"]:
            with self.subTest(query=query):
                hits = search_text(self.path, query)
                self.assertTrue(all(h["kind"] != "table_context" for h in hits))
        self.assertTrue(validate_store(self.path)["valid"])

    def test_raw_table_context_is_excluded_by_default_and_explicitly_labelled(self):
        self.assertEqual(search_text(self.path, "SecretDimension"), [])
        hits = search_text(self.path, "SecretDimension", kinds=("table_context",))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["authority"], "raw_context_only")
        self.assertEqual(query_facts(self.path, parameter="width"), [])
        self.assertEqual(len(query_facts(self.path, parameter="width", verified_only=False)), 1)

    def test_verified_gate_excludes_ambiguous_invalid_pending_and_blocked(self):
        hits = query_facts(self.path)
        self.assertEqual({h["fact_id"] for h in hits}, {"height", "missing", "zero", "vector", "range"})
        self.assertEqual(len(query_facts(self.path, verified_only=False)), 9)

    def test_json_preserves_zero_null_vectors_and_ranges(self):
        indexed = {f["fact_id"]: f for f in query_facts(self.path)}
        self.assertEqual(indexed["zero"]["value"], 0)
        self.assertIsNone(indexed["missing"]["value"])
        self.assertEqual(indexed["vector"]["value_components"], [17, 29])
        self.assertEqual(indexed["range"]["value_range"], {"lower": Decimal("0.7"), "upper": Decimal("2.3")})
        self.assertNotEqual(indexed["zero"]["store_ref"]["fact_key"], indexed["missing"]["store_ref"]["fact_key"])

    def test_services_preserve_exact_money_json_and_can_be_filtered(self):
        service = query_services(self.path, report_id="FSR-9003")[0]
        self.assertEqual(service["cost"]["reported_total"]["amount"], Decimal("900000000000.1234"))
        self.assertEqual(query_services(self.path, country="Canada"), [])
        self.assertEqual(len(query_services(self.path, country="Chile")), 1)

    def test_readonly_handle_and_missing_path_never_create_or_write(self):
        with KnowledgeStore(self.path) as store:
            with self.assertRaises(sqlite3.OperationalError):
                store.connection.execute("DELETE FROM facts")
        missing = self.path.parent / "not-created.sqlite"
        with self.assertRaises(FileNotFoundError):
            KnowledgeStore(missing)
        self.assertFalse(missing.exists())

    def test_counts_and_integrity_include_fts_and_fact_review_states(self):
        quality = validate_store(self.path)
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["counts"], self.counts)
        self.assertEqual(quality["counts"]["documents"], 2)
        self.assertEqual(quality["counts"]["verified_facts"], 5)
        self.assertEqual(quality["counts"]["fts_chunks"], 4)

    def test_refuses_overwrite_without_changing_existing_store(self):
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            build_store(self.path, *self.inputs)
        self.assertEqual(self.path.read_bytes(), original)

    def test_detects_missing_fts_content_and_changed_source_identity(self):
        with closing(sqlite3.connect(self.path)) as writable, writable:
            writable.execute("DELETE FROM chunks_fts WHERE chunk_id='context-a'")
            writable.execute("UPDATE chunks SET source_sha256=? WHERE id='context-b'", ("c" * 64,))
        quality = validate_store(self.path)
        self.assertFalse(quality["valid"])
        self.assertEqual(quality["fts_missing_or_changed"], 1)
        self.assertEqual(quality["source_identity_errors"], 1)

    def test_detects_pending_fact_index_promotion_without_json_review(self):
        with closing(sqlite3.connect(self.path)) as writable, writable:
            writable.execute("UPDATE facts SET review_status='verified' WHERE fact_id='pending-width'")
        quality = validate_store(self.path)
        self.assertFalse(quality["valid"])
        self.assertTrue(quality["payload_index_errors"])

    def test_mislabelled_raw_context_is_not_returned_as_narrative(self):
        with closing(sqlite3.connect(self.path)) as writable, writable:
            writable.execute("UPDATE chunks SET kind='narrative' WHERE id='raw-a'")
        self.assertEqual(search_text(self.path, "SecretDimension"), [])
        quality = validate_store(self.path)
        self.assertFalse(quality["valid"])
        self.assertEqual(quality["chunk_authority_errors"], 1)


class StoreBuildValidationTests(unittest.TestCase):
    def build(self, inputs):
        with tempfile.TemporaryDirectory() as tmp:
            return build_store(Path(tmp) / "new.sqlite", *inputs)

    def test_local_table_and_fact_ids_are_scoped_by_document_and_page(self):
        docs, chunks, tables, services = fixtures()
        second = deepcopy(tables[0]); second["source"]["page"] = 3
        for fact in second["facts"]:
            fact["source_refs"][0]["page"] = 3
        third = deepcopy(tables[0]); third["doc_id"] = "synthetic-b"
        third["source"].update(file="invented-b.pdf", sha="b" * 64)
        for fact in third["facts"]:
            fact["source_refs"][0].update(file="invented-b.pdf", sha="b" * 64)
        counts = self.build((docs, chunks, tables + [second, third], services))
        self.assertEqual(counts["tables"], 3)
        self.assertEqual(counts["facts"], 27)

    def test_rejects_duplicate_documents_chunks_tables_and_services(self):
        for slot, message in [(0, "Duplicate document"), (1, "Duplicate chunk"), (2, "Duplicate table"), (3, "Duplicate service")]:
            with self.subTest(slot=slot):
                inputs = fixtures(); inputs[slot].append(deepcopy(inputs[slot][0]))
                if slot == 1:
                    inputs[slot][-1]["doc_id"] = "synthetic-b"
                with self.assertRaisesRegex(ValueError, message):
                    self.build(inputs)

    def test_rejects_cross_document_source_hash_page_and_citation_mismatch(self):
        edits = [lambda x: x[1][0].update(source_sha256="b" * 64),
                 lambda x: x[1][0].update(page=4),
                 lambda x: x[1][0]["citation"].update(page=3),
                 lambda x: x[2][0]["source"].update(sha="b" * 64),
                 lambda x: x[2][0]["facts"][0]["source_refs"][0].update(page=1)]
        for edit in edits:
            inputs = fixtures(); edit(inputs)
            with self.assertRaises(ValueError):
                self.build(inputs)

    def test_rejects_raw_context_promotion_and_bad_numeric_types(self):
        edits = [lambda x: x[1][2].update(evidence_status="source_text"),
                 lambda x: x[2][0]["facts"][0].update(value="37.19"),
                 lambda x: x[2][0]["facts"][0].update(value=True),
                 lambda x: x[2][0]["facts"][0].update(value=float("nan"))]
        for edit in edits:
            inputs = fixtures(); edit(inputs)
            with self.assertRaises(ValueError):
                self.build(inputs)

    def test_can_build_an_explicit_empty_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.sqlite"; path.touch()
            counts = build_store(path, [], [], [], [])
            self.assertEqual(counts["documents"], 0)
            self.assertTrue(validate_store(path)["valid"])


if __name__ == "__main__":
    unittest.main()
