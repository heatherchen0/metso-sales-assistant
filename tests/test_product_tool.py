"""Synthetic retrieval, quantity authority and local visual provenance checks."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from metso_assistant.product_tool import retrieve_product, relative_artifact, visual_intent
from metso_assistant.store import build_store


def synthetic_inputs():
    docs = [{"id": "invented-control", "title": "Invented moisture feedback control", "source_file": "invented-control.pdf", "source_sha256": "c" * 64, "page_count": 3},
            {"id": "invented-press-brochure-a", "title": "Invented press brochure A", "source_file": "invented-press-a.pdf", "source_sha256": "a" * 64, "page_count": 3},
            {"id": "invented-press-brochure-b", "title": "Invented press brochure B", "source_file": "invented-press-b.pdf", "source_sha256": "b" * 64, "page_count": 3}]
    chunks = []
    texts = [(0, 1, "Changing slurry characteristics can reduce filter performance. Moisture feedback control adjusts cycle setpoints."),
             (0, 2, "Control architecture diagram\n\nMoisture feedback returns to the controller."),
             (1, 1, "Invented press uses pressure filtration.\n\nMaximum width is 999.99 mm."),
             (2, 1, "Invented press uses pressure filtration.")]
    for ordinal, (doc_index, page, text) in enumerate(texts):
        doc = docs[doc_index]
        chunks.append({"id": f"synthetic-{ordinal}", "doc_id": doc["id"], "page": page, "section": None, "text": text,
                       "kind": "narrative", "source_sha256": doc["source_sha256"], "evidence_status": "source_text",
                       "citation": {"source_file": doc["source_file"], "page": page},
                       "blocks": [{"text": block, "label": "text"} for block in text.split("\n\n")]})
    chunks.append({"id": "raw-secret", "doc_id": docs[1]["id"], "page": 3, "text": "Secret dimension 12345.67", "kind": "table_context", "evidence_status": "raw_table", "source_sha256": docs[1]["source_sha256"], "citation": "invented press page 3"})
    tables = []
    for ordinal, doc in enumerate(docs[1:]):
        source = {"file": doc["source_file"], "sha": doc["source_sha256"], "page": 3}
        facts = []
        for fid, parameter, review, value, status in [
            ("height", "height", "verified", 37.19 + ordinal, "reported"),
            ("width", "width", "pending", 91.73, "reported"),
            ("length", "length", "ambiguous", 82.61, "ambiguous"),
            ("weight", "weight", "verified", None, "missing"),
            ("depth", "depth", "verified", 42.83, "reported"),
            ("chambers", "Chambers", "verified", 18, "reported"),
        ]:
            fact = {"fact_id": fid, "product_family": "InventedPress", "configuration": "P3", "parameter": parameter,
                    "value": value, "value_kind": "scalar", "unit": "mm", "status": status, "review_status": review,
                    "issues": [], "review_blockers": [{"code": "unresolved_association"}] if fid == "depth" else [],
                    "source_refs": [{**source, "table_id": "table-01", "locator": {"row": len(facts), "column": 1}}]}
            facts.append(fact)
        tables.append({"doc_id": doc["id"], "table_id": "table-01", "source": source, "product_family": "InventedPress", "facts": facts, "issues": [], "normalized_table": {"cells": []}})
    return docs, chunks, tables, []


class ProductToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = self.root / "store with spaces.sqlite"
        self.inputs = synthetic_inputs()
        build_store(self.store, *self.inputs)

    def tearDown(self):
        self.temporary.cleanup()

    def retrieve(self, query, **kwargs):
        return retrieve_product(self.store, query, **kwargs)

    def test_symptom_paraphrase_finds_actual_context_without_product_trigger(self):
        result = self.retrieve("Our filters lose efficiency when feed grade varies")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["passages"][0]["doc_id"], "invented-control")
        self.assertIn("Changing slurry characteristics", result["data"]["passages"][0]["text"])
        self.assertEqual(result["data"]["facts"], [])

    def test_visible_narrative_numbers_never_become_approved_facts_and_raw_tables_stay_excluded(self):
        result = self.retrieve("Invented press P3 guaranteed recovery gain")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["data"]["facts"], [])
        self.assertIn("999.99", json.dumps(result["data"]["passages"]))
        self.assertTrue(all(row["numeric_assertion_authority"] is False for row in result["data"]["passages"]))
        self.assertTrue(all(row["omitted_numeric_blocks"] == 0 for row in result["data"]["passages"]))
        self.assertNotIn("12345.67", json.dumps(self.retrieve("Secret dimension")))
        contextual_count = self.retrieve("Invented press P3 with 18 chambers, what is the quantum flux rating?")
        self.assertEqual(contextual_count["status"], "insufficient_evidence")
        self.assertEqual(contextual_count["data"]["facts"], [])
        self.assertIn("supported_parameter", contextual_count["missing_fields"])

    def test_numeric_application_paragraph_remains_searchable_without_approving_its_threshold(self):
        docs, chunks, tables, services = synthetic_inputs()
        chunks.append({"id": "numeric-application", "doc_id": docs[0]["id"], "page": 3,
                       "section": "Fine material recovery", "text": "The feedback controller can be combined with the existing vessel to recover valuable fines below 17 microns from tailings.",
                       "kind": "narrative", "evidence_status": "source_text", "source_sha256": docs[0]["source_sha256"],
                       "citation": {"source_file": docs[0]["source_file"], "page": 3}})
        expanded = self.root / "expanded-source.sqlite"
        build_store(expanded, docs, chunks, tables, services)
        self.store = expanded
        result = self.retrieve("Our existing vessel loses valuable fines below 17 microns to tailings")
        self.assertEqual(result["data"]["passages"][0]["chunk_id"], "numeric-application")
        self.assertIn("17 microns", result["data"]["passages"][0]["text"])
        self.assertEqual(result["data"]["facts"], [])
        self.assertFalse(result["data"]["passages"][0]["numeric_assertion_authority"])

    def test_ordinary_operation_explanation_does_not_require_visual_evidence(self):
        self.assertFalse(visual_intent("How does the Invented control work?"))
        self.assertFalse(visual_intent("Our control loops oscillate when feed changes"))
        result = self.retrieve("Explain the Invented control flow diagram")
        self.assertTrue(result["data"]["needs_visual_inspection"])
        self.assertEqual(result["data"]["request_kind"], "visual")

    def test_product_identity_is_source_derived_separately_from_retrieved_capability(self):
        result = self.retrieve("Invented press pressure filtration")
        identity = next(row for row in result["data"]["products"] if row["doc_id"] == "invented-press-brochure-a")
        self.assertEqual(identity["name"], "Invented press")
        self.assertEqual(identity["identity_evidence"][0]["quote"], "Invented press")
        self.assertEqual(identity["authority"], "source_identity_only")
        self.assertEqual(result["data"]["passages"][0]["product_id"], result["data"]["passages"][0]["doc_id"])

    def test_mixed_recommendation_keeps_context_and_withholds_guarantee(self):
        result = self.retrieve("Recommend a control upgrade for changing feed properties with a guaranteed improvement percentage")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["request_kind"], "mixed")
        self.assertTrue(result["data"]["passages"])
        self.assertEqual(result["data"]["facts"], [])
        self.assertTrue(any("numerical" in warning for warning in result["warnings"]))

    def test_only_reported_reviewed_unblocked_facts_expose_values(self):
        approved = self.retrieve("Invented press P3 height brochure A")
        self.assertEqual(len(approved["data"]["facts"]), 1)
        fact = approved["data"]["facts"][0]
        self.assertEqual(fact["value"], 37.19)
        self.assertTrue(fact["trusted"])
        source = next(s for s in approved["sources"] if s["id"] == fact["source_id"])
        self.assertEqual(source["file"], "invented-press-a.pdf")
        self.assertEqual(json.loads(source["excerpt"])["value"], fact["value"])
        for parameter, forbidden in [("width", "91.73"), ("length", "82.61"), ("depth", "42.83")]:
            with self.subTest(parameter=parameter):
                result = self.retrieve(f"Invented press P3 {parameter} brochure A")
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertEqual(result["data"]["facts"], [])
                self.assertTrue(result["data"]["unavailable_facts"])
                self.assertNotIn(forbidden, json.dumps(result))
        self.assertEqual(self.retrieve("Invented press P3 weight")["data"]["facts"], [])

    def test_editions_remain_distinct_and_source_ids_stable(self):
        result = self.retrieve("Invented press P3 height")
        facts = result["data"]["facts"]
        self.assertEqual({fact["value"] for fact in facts}, {37.19, 38.19})
        self.assertEqual(len({fact["source_id"] for fact in facts}), 2)
        repeated = self.retrieve("Invented press P3 height")
        self.assertEqual([s["id"] for s in result["sources"]], [s["id"] for s in repeated["sources"]])

    def test_missing_or_prefix_configuration_does_not_guess(self):
        for query in ("Invented press height", "Invented press P30 height"):
            with self.subTest(query=query):
                result = self.retrieve(query)
                self.assertEqual(result["data"]["facts"], [])
                self.assertIn("configuration", result["missing_fields"])
                self.assertNotIn("37.19", json.dumps(result))

    def test_fts_is_candidate_source_and_missing_store_is_not_created(self):
        with closing(sqlite3.connect(self.store)) as connection, connection:
            connection.execute("DELETE FROM chunks_fts WHERE chunk_id='synthetic-0'")
        self.assertEqual(self.retrieve("changing slurry characteristics")["data"]["passages"], [])
        missing = self.root / "missing.sqlite"
        with self.assertRaises(FileNotFoundError):
            retrieve_product(missing, "something")
        self.assertFalse(missing.exists())

    def test_index_review_promotion_cannot_bypass_payload_gate(self):
        with closing(sqlite3.connect(self.store)) as connection, connection:
            connection.execute("UPDATE facts SET review_status='verified' WHERE fact_id='width'")
        result = self.retrieve("Invented press P3 width")
        self.assertEqual(result["data"]["facts"], [])
        self.assertNotIn("91.73", json.dumps(result))

    def write_visual(self, reference="invented-control/page-002/source.png", source_hash="c" * 64):
        raw = self.root / "raw"
        page_dir = raw / "docling" / "invented-control" / "page-002"
        page_dir.mkdir(parents=True, exist_ok=True)
        image = page_dir / "source.png"
        image.write_bytes(b"synthetic image content")
        page = {"status": "success", "page": 2, "source_file": "invented-control.pdf", "source_sha256": source_hash,
                "source_render_file": reference, "figures": [], "artifacts": [{"path": reference, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}
        (page_dir / "page.json").write_text(json.dumps(page), encoding="utf-8")
        return raw, image

    def test_visuals_bound_to_original_page_hash_and_need_inspection(self):
        raw, image = self.write_visual()
        result = self.retrieve("Invented control architecture diagram", raw_dir=raw)
        self.assertTrue(result["data"]["needs_visual_inspection"])
        candidate = result["data"]["visual_candidates"][0]
        self.assertEqual(Path(candidate["image_file"]), image.resolve())
        self.assertEqual(candidate["page"], 2)
        self.assertTrue(candidate["requires_inspection"])
        image.write_bytes(b"changed")
        tampered = self.retrieve("Invented control architecture diagram", raw_dir=raw)
        self.assertEqual(tampered["data"]["visual_candidates"], [])

    def test_flowsheet_layout_and_plural_diagram_questions_trigger_inspection(self):
        for question in ("In the installation flowsheet, where does the drain return?", "Explain the flow sheets", "Show the flowsheets", "Show the layout", "Explain the layouts", "Follow the arrows in the diagrams", "Read these schematics"):
            with self.subTest(question=question):
                self.assertTrue(visual_intent(question))
        self.assertFalse(visual_intent("Which filter helps changing feed characteristics?"))
        raw, image = self.write_visual()
        result = self.retrieve("In the Invented control flowsheet, what returns to the controller?", raw_dir=raw)
        self.assertTrue(result["data"]["needs_visual_inspection"])
        self.assertEqual(Path(result["data"]["visual_candidates"][0]["image_file"]), image.resolve())
        self.assertTrue(result["data"]["visual_candidates"][0]["requires_inspection"])

    def test_unsafe_visual_paths_and_wrong_source_identity_are_rejected(self):
        for reference in ("../outside.png", "https://example.test/image.png", "C:outside.png"):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                relative_artifact(self.root, reference)
        raw, image = self.write_visual(source_hash="e" * 64)
        result = self.retrieve("Invented control architecture diagram", raw_dir=raw)
        self.assertEqual(result["data"]["visual_candidates"], [])

    def test_query_and_limit_validation_and_read_only_result(self):
        before = self.store.read_bytes()
        self.retrieve('control" OR text:* NEAR(secret)')
        self.assertEqual(before, self.store.read_bytes())
        for query in ("", " ", 42):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.retrieve(query)
        for limit in (0, True, 31):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.retrieve("control", top_k=limit)

    def test_validated_explicit_document_scope_applies_to_qualitative_and_numerical_queries(self):
        scope = ["invented-press-brochure-a"]
        qualitative = self.retrieve("Invented control and pressure filtration", scope_doc_ids=scope)
        self.assertEqual({row["doc_id"] for row in qualitative["data"]["passages"]}, set(scope))
        self.assertEqual(qualitative["data"]["scope_doc_ids"], scope)
        numerical = self.retrieve("What is the P3 height?", scope_doc_ids=scope)
        self.assertEqual([row["doc_id"] for row in numerical["data"]["facts"]], scope)
        self.assertEqual(numerical["data"]["facts"][0]["value"], 37.19)

    def test_explicit_document_scope_rejects_unknown_or_invalid_ids_before_dense_inference(self):
        index = self.root / "synthetic-index.json"
        index.write_text("{}", encoding="utf-8")
        with patch("metso_assistant.semantic.semantic_search", side_effect=AssertionError("No dense inference for invalid scope")):
            for scope in ([], "invented-control", [None], [""], ["not-in-this-store"]):
                with self.subTest(scope=scope), self.assertRaises(ValueError):
                    self.retrieve("control", scope_doc_ids=scope, retrieval_mode="semantic", semantic_index=index)

    def test_explicit_document_scope_filters_both_optional_candidate_pools(self):
        index = self.root / "synthetic-index.json"
        index.write_text("{}", encoding="utf-8")
        dense = [dict(row, semantic_score=.9) for row in self.inputs[1][:3]]
        for mode in ("semantic", "hybrid"):
            with self.subTest(mode=mode), patch("metso_assistant.semantic.semantic_search", return_value=dense):
                result = self.retrieve("moisture and pressure filtration", scope_doc_ids=["invented-control"],
                                       retrieval_mode=mode, semantic_index=index)
                self.assertTrue(result["data"]["passages"])
                self.assertEqual({row["doc_id"] for row in result["data"]["passages"]}, {"invented-control"})

    def test_explicit_document_scope_also_bounds_visual_page_candidates(self):
        raw, _ = self.write_visual()
        result = self.retrieve("Explain the Invented press and control architecture diagrams", raw_dir=raw,
                               scope_doc_ids=["invented-control"])
        self.assertTrue(result["data"]["visual_candidates"])
        self.assertEqual({row["doc_id"] for row in result["data"]["visual_candidates"]}, {"invented-control"})

    def test_optional_dense_modes_keep_zero_overlap_passage_and_filter_raw_tables(self):
        index = self.root / "synthetic-index.json"
        index.write_text("{}", encoding="utf-8")
        source = dict(self.inputs[1][0], semantic_score=.9)
        raw = dict(self.inputs[1][-1], semantic_score=1.0)
        for mode in ("semantic", "hybrid"):
            with self.subTest(mode=mode), patch("metso_assistant.semantic.semantic_search", return_value=[raw, source]):
                result = self.retrieve("Sense soggy piles as they are dropped", retrieval_mode=mode, semantic_index=index)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["data"]["passages"][0]["chunk_id"], source["id"])
                self.assertEqual(len(result["data"]["passages"]), 1)
                self.assertEqual(result["data"]["facts"], [])
                self.assertNotIn("12345.67", json.dumps(result))
                self.assertEqual(result["data"]["retrieval_metadata"]["candidate_pool_limit"], 60)

    def test_optional_retrieval_cannot_promote_numeric_candidates(self):
        index = self.root / "synthetic-index.json"
        index.write_text("{}", encoding="utf-8")
        source = dict(self.inputs[1][0], semantic_score=.9)
        for parameter in ("height", "width", "length", "depth"):
            base = self.retrieve(f"Invented press P3 {parameter} brochure A")
            for mode in ("semantic", "hybrid"):
                with self.subTest(parameter=parameter, mode=mode), patch("metso_assistant.semantic.semantic_search", return_value=[source]):
                    result = self.retrieve(f"Invented press P3 {parameter} brochure A", retrieval_mode=mode, semantic_index=index)
                    self.assertEqual(result["data"]["facts"], base["data"]["facts"])
                    self.assertEqual(result["data"]["unavailable_facts"], base["data"]["unavailable_facts"])

    def test_keyword_mode_ignores_optional_assets_and_dense_endpoint_must_be_local(self):
        base = self.retrieve("moisture control")
        with patch("metso_assistant.semantic.semantic_search", side_effect=AssertionError("No dense call in keyword mode")):
            result = self.retrieve("moisture control", retrieval_mode="keyword", semantic_index=self.root / "absent.json", embedding_endpoint="https://example.test")
            self.assertEqual(result, base)
        index = self.root / "synthetic-index.json"
        index.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.retrieve("moisture control", retrieval_mode="semantic", semantic_index=index, embedding_endpoint="https://example.test")
        for mode in ("semantic", "hybrid"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.retrieve("moisture control", retrieval_mode=mode)
        with self.assertRaises(ValueError):
            self.retrieve("moisture control", retrieval_mode="imagined")


if __name__ == "__main__":
    unittest.main()
