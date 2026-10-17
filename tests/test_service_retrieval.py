"""Mocked adapter integration: rankings cannot change numeric authority."""
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_semantic import search
from metso_assistant.services import parse_text
from metso_assistant.store import build_store


class ServiceRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        record = parse_text("""Reference: FSR-9991
Site: Invented test plant, Chile
Equipment: Helix Separator
| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Quartz injector cartridges | 3 | 12.50/ea | 37.50 |
| Zircon coupler bearings | 2 | 7.25/ea | 14.50 |
""", source_path="invented-adapter-source.txt")
        self.store = self.root / "prepared.sqlite"
        build_store(self.store, [], [], [], [record])
        self.workspace = ServiceWorkspace(self.store, self.root / "observations.sqlite")
        self.index = self.root / "service-vectors.json"
        self.index.write_text(json.dumps({"model": {"tag": "qwen3-embedding:0.6b"}}), encoding="utf-8")
        self.query = "Quartz injector cartridges"
        self.keyword = self.workspace.search(self.query)
        self.workspace.read_report("FSR-9991")
        self.original_facts = deepcopy(self.workspace.facts)
        self.bearing = [f for f in self.original_facts.values() if "Zircon" in f["label"]]
        self.events = []
        self.model = Mock()
        self.model.unload.side_effect = lambda: self.events.append("chat_unload") or {"operation": "chat_unload"}

    def dense(self, ids):
        return [{"id": identifier, "semantic_score": .9, "group_id": self.original_facts[identifier]["scope"]["group"],
                 "group_rank": index + 1} for index, identifier in enumerate(ids)]

    def patches(self, rows=None, failure=None):
        def embed(*args, **kwargs):
            self.events.append("embedding_query")
            if failure:
                raise failure
            return rows
        def unload(endpoint, model):
            self.events.append("embedding_unload")
            return {"operation": "embedding_unload", "model": model}
        return (patch("metso_assistant.service_semantic.search_service_embeddings", side_effect=embed),
                patch("metso_assistant.service_semantic.unload_local_model", side_effect=unload),
                patch("metso_assistant.service_semantic.OllamaClient", return_value=object()))

    def test_keyword_bypasses_optional_index_models_and_gpu_lifecycle(self):
        with patch("metso_assistant.service_semantic.search_service_embeddings") as dense, \
             patch("metso_assistant.service_semantic.unload_local_model") as unload, \
             patch("metso_assistant.service_semantic.OllamaClient") as client:
            result, traces = search(self.workspace, self.query, mode="keyword", model=self.model,
                semantic_index=self.root / "does-not-exist.json", country="Chile", equipment="Helix")
        self.assertEqual(result["retrieval_mode"], "keyword")
        self.assertEqual(result["facts"], self.workspace.search(self.query, country="Chile", equipment="Helix")["facts"])
        self.assertEqual(traces, [])
        dense.assert_not_called(); unload.assert_not_called(); client.assert_not_called()
        self.model.unload.assert_not_called()

    def test_hybrid_fuses_rankings_and_preserves_original_fact_source_groups(self):
        common = self.keyword["facts"][0]["id"]
        rows = self.dense([self.bearing[0]["id"], common, *[f["id"] for f in self.bearing[1:]]])
        embed_patch, unload_patch, client_patch = self.patches(rows)
        before = self.store.read_bytes()
        with embed_patch as dense, unload_patch as unload, client_patch:
            result, traces = search(self.workspace, self.query, mode="hybrid", model=self.model,
                                    semantic_index=self.index, country="Chile", equipment="Helix")
        self.assertEqual(result["facts"][0]["id"], common)
        self.assertTrue({f["id"] for f in self.bearing}.issubset({f["id"] for f in result["facts"]}))
        self.assertEqual(self.events, ["chat_unload", "embedding_query", "embedding_unload"])
        self.assertEqual([t["operation"] for t in traces], ["chat_unload", "embedding_unload"])
        self.assertEqual(result["semantic_groups"], 2)
        for fact in result["facts"]:
            self.assertEqual(fact, self.original_facts[fact["id"]])
            self.assertIsInstance(fact["value"], Decimal)
        expected_sources = {sid for fact in result["facts"] for sid in fact["source_ids"]}
        self.assertEqual({source["id"] for source in result["sources"]}, expected_sources)
        self.assertTrue(all(source["report_id"] == "FSR-9991" and source["lines"] for source in result["sources"]))
        self.assertEqual(self.store.read_bytes(), before)
        dense.assert_called_once(); unload.assert_called_once_with("http://127.0.0.1:11435", "qwen3-embedding:0.6b")

    def test_semantic_mode_uses_dense_order_deduplicates_ids_and_does_not_add_keyword_candidates(self):
        ids = [self.bearing[-1]["id"], self.bearing[0]["id"]]
        rows = self.dense([ids[0], ids[0], ids[1]])
        embed_patch, unload_patch, client_patch = self.patches(rows)
        with embed_patch, unload_patch, client_patch:
            result, _ = search(self.workspace, self.query, mode="semantic", semantic_index=self.index)
        self.assertEqual([f["id"] for f in result["facts"]], ids)
        self.assertEqual(result["retrieval_mode"], "semantic")
        self.assertEqual(self.events, ["embedding_query", "embedding_unload"])
        self.assertTrue(all(f["id"] not in {k["id"] for k in self.keyword["facts"]} for f in result["facts"]))

    def test_embedding_failure_still_unloads_embedding_model_after_chat_release(self):
        embed_patch, unload_patch, client_patch = self.patches(failure=RuntimeError("mock inference failed"))
        with embed_patch, unload_patch, client_patch:
            with self.assertRaisesRegex(RuntimeError, "mock inference failed"):
                search(self.workspace, self.query, mode="hybrid", semantic_index=self.index, model=self.model)
        self.assertEqual(self.events, ["chat_unload", "embedding_query", "embedding_unload"])

    def test_missing_index_and_missing_model_binding_fail_before_any_gpu_actions(self):
        for path in (self.root / "absent.json", self.index):
            if path == self.index:
                path.write_text("{}", encoding="utf-8")
            with self.subTest(path=path.name), \
                 patch("metso_assistant.service_semantic.search_service_embeddings") as dense, \
                 patch("metso_assistant.service_semantic.unload_local_model") as unload:
                with self.assertRaises(ValueError):
                    search(self.workspace, self.query, mode="hybrid", semantic_index=path, model=self.model)
            dense.assert_not_called(); unload.assert_not_called()
        self.model.unload.assert_not_called()

    def test_chat_unload_failure_does_not_start_embedding_or_claim_success(self):
        self.model.unload.side_effect = RuntimeError("mock chat unload failed")
        with patch("metso_assistant.service_semantic.search_service_embeddings") as dense, \
             patch("metso_assistant.service_semantic.unload_local_model") as unload:
            with self.assertRaisesRegex(RuntimeError, "mock chat unload failed"):
                search(self.workspace, self.query, mode="semantic", semantic_index=self.index, model=self.model)
        dense.assert_not_called(); unload.assert_not_called()

    def test_unknown_mode_is_rejected_before_workspace_or_models_are_used(self):
        with patch.object(self.workspace, "search") as keyword, \
             patch("metso_assistant.service_semantic.search_service_embeddings") as dense:
            with self.assertRaisesRegex(ValueError, "Unknown service retrieval mode"):
                search(self.workspace, self.query, mode="invented", semantic_index=self.index)
        keyword.assert_not_called(); dense.assert_not_called()


if __name__ == "__main__":
    unittest.main()
