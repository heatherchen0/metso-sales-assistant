"""Synthetic service dense-index bindings; no network or real model calls."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_semantic import (_hash, build_service_embeddings,
    check_service_embeddings, search_service_embeddings)
from metso_assistant.services import parse_text
from metso_assistant.store import build_store


class MockEmbeddingClient:
    endpoint = "http://127.0.0.1:11435"
    timeout = 1
    tag = "qwen3-embedding:0.6b"

    def __init__(self):
        self.calls = []
        self.digest = "e" * 64
        self.remote = False
        self.invalid = None
        self.on_embed = None

    def json(self, route, payload=None):
        self.calls.append((route, deepcopy(payload)))
        if route == "/api/tags":
            return {"models": [{"name": self.tag, "model": self.tag, "digest": self.digest,
                                "size": 123, "details": {"family": "synthetic"}}]}
        if route == "/api/show":
            return {"capabilities": ["embedding"], "model_info": {"dimensions": 2},
                    "remote_host": "https://example.invalid" if self.remote else ""}
        if route == "/api/embed":
            if self.on_embed:
                self.on_embed()
            vectors = [self.invalid if self.invalid is not None else
                       [4.0, 0.0] if "bearing" in text.casefold() else [0.0, 3.0]
                       for text in payload["input"]]
            return {"model": payload["model"], "embeddings": vectors, "prompt_eval_count": 7,
                    "load_duration": 0, "total_duration": 1}
        raise AssertionError(route)


class ServiceSemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        record = parse_text("""Reference: FSR-9901
Site: Invented facility, Chile
Equipment: Quartz Separator
On-site hours: 4
| Task | Time Spent (hrs) |
|------|------------------|
| Replace quartz injectors | 1.5 |
| Inspect rotor bearings | 2.5 |

| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Quartz injectors | 2 | 17.15/ea | 34.30 |
| Rotor bearings | 3 | 11/ea | 33 |
""", source_path="invented-semantic-source.txt")
        # This test supplies prepared metadata directly; it does not test the
        # legacy parser's finite list of recognized equipment families.
        record["equipment"]["description"] = "Quartz Separator"
        self.store = self.root / "prepared.sqlite"
        build_store(self.store, [], [], [], [record])
        self.workspace = ServiceWorkspace(self.store, self.root / "observations.sqlite")
        self.output = self.root / "vectors.json"
        self.client = MockEmbeddingClient()

    def build(self, **kwargs):
        return build_service_embeddings(self.workspace, self.output, self.client, **kwargs)

    def test_groups_avoid_duplicate_vectors_and_embed_labels_scope_literal_units_not_values(self):
        before = self.store.read_bytes()
        result = self.build(batch_size=2)
        self.assertLess(result["groups"], result["facts"])
        requests = [payload for route, payload in self.client.calls if route == "/api/embed"]
        text = "\n".join(item for request in requests for item in request["input"])
        self.assertIn("Quartz Separator", text)
        self.assertIn("Chile", text)
        self.assertIn("money/each", text)
        self.assertNotIn("17.15", text)
        self.assertTrue(all(request["truncate"] is False and request["options"]["num_ctx"] == 2048 for request in requests))
        artifact = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len({entry["group_id"] for entry in artifact["entries"]}), result["groups"])
        self.assertEqual(self.store.read_bytes(), before)

    def test_search_returns_current_fact_ids_without_promoting_or_registering_numeric_evidence(self):
        self.build()
        result = search_service_embeddings(self.workspace, self.output, "rotor bearing", self.client, limit=4)
        self.assertTrue(result)
        self.assertEqual(self.workspace.facts, {})
        with closing(self.workspace._connect()) as connection:
            allowed = {row[0] for row in connection.execute("SELECT id FROM observations")}
        self.assertTrue(all(row["id"] in allowed for row in result))
        self.assertTrue(all(set(row) == {"id", "semantic_score", "group_id", "group_rank"} for row in result))
        self.assertTrue(all(-1 <= row["semantic_score"] <= 1 for row in result))
        query = [payload for route, payload in self.client.calls if route == "/api/embed"][-1]
        self.assertTrue(query["input"][0].startswith("Instruct:"))

    def test_check_is_model_metadata_only_and_existing_artifact_cannot_be_overwritten(self):
        self.build()
        calls = sum(route == "/api/embed" for route, _ in self.client.calls)
        self.assertTrue(check_service_embeddings(self.workspace, self.output, self.client)["valid"])
        self.assertEqual(sum(route == "/api/embed" for route, _ in self.client.calls), calls)
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError): self.build()
        self.assertEqual(self.output.read_bytes(), original)

    def test_artifact_content_and_group_identity_cannot_be_rebound_by_editing_ids(self):
        self.build()
        artifact = json.loads(self.output.read_text(encoding="utf-8"))
        artifact["entries"][0]["fact_ids"][0] = "invented-reference"
        self.output.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            check_service_embeddings(self.workspace, self.output, self.client)
        artifact["content_sha256"] = _hash({k: v for k, v in artifact.items() if k != "content_sha256"})
        self.output.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity"):
            check_service_embeddings(self.workspace, self.output, self.client)

    def test_changed_local_model_or_query_configuration_rejects_before_query_embedding(self):
        self.build()
        calls = sum(route == "/api/embed" for route, _ in self.client.calls)
        self.client.digest = "f" * 64
        with self.assertRaisesRegex(ValueError, "model changed"):
            search_service_embeddings(self.workspace, self.output, "bearing", self.client)
        self.assertEqual(sum(route == "/api/embed" for route, _ in self.client.calls), calls)
        self.client.digest = "e" * 64
        with patch("metso_assistant.service_semantic.QUERY_INSTRUCTION", "changed"):
            with self.assertRaisesRegex(ValueError, "settings"):
                check_service_embeddings(self.workspace, self.output, self.client)

    def test_sidecar_content_change_is_rejected_during_query_even_if_source_database_is_unchanged(self):
        self.build()
        before = self.store.read_bytes()
        def mutate():
            with closing(sqlite3.connect(self.workspace.index_path)) as connection:
                connection.execute("UPDATE observations_fts SET text='changed search text'")
                connection.commit()
        self.client.on_embed = mutate
        with self.assertRaisesRegex(ValueError, "sidecar content"):
            search_service_embeddings(self.workspace, self.output, "bearing", self.client)
        self.assertEqual(self.store.read_bytes(), before)

    def test_model_changes_during_build_do_not_publish_an_artifact(self):
        self.client.on_embed = lambda: setattr(self.client, "digest", "f" * 64)
        with self.assertRaisesRegex(ValueError, "model changed"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_invalid_vectors_or_dimensions_cannot_publish_or_search(self):
        for vector in ([0.0, 0.0], [float("nan"), 0.0], [True, 0.0]):
            with self.subTest(vector=str(vector)):
                self.client.invalid = vector
                with self.assertRaises(ValueError): self.build()
                self.assertFalse(self.output.exists())
        self.client.invalid = None
        self.build()
        self.client.invalid = [1.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            search_service_embeddings(self.workspace, self.output, "bearing", self.client)

    def test_loopback_local_model_and_bounded_arguments_are_enforced(self):
        for kwargs in ({"batch_size": True}, {"batch_size": 0}):
            with self.assertRaises(ValueError): self.build(**kwargs)
        self.client.remote = True
        with self.assertRaises(ValueError): self.build()
        self.client.remote = False
        self.client.endpoint = "https://example.invalid"
        with self.assertRaises(ValueError): self.build()
        self.client.endpoint = "http://127.0.0.1:11435"
        self.build()
        for query, limit in (("", 3), ("bearing", True), ("bearing", 0), ("x" * 6001, 3)):
            with self.assertRaises(ValueError):
                search_service_embeddings(self.workspace, self.output, query, self.client, limit=limit)


if __name__ == "__main__":
    unittest.main()
