"""Synthetic cache and ranking regressions; no HTTP or real model inference."""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from metso_assistant.semantic import (QUERY_INSTRUCTION, build_embedding_index,
                                     check_embedding_index, cosine_similarity,
                                     fuse_rankings, semantic_search)
from metso_assistant.store import build_store


class FakeClient:
    endpoint = "http://127.0.0.1:11434"
    timeout = 1
    tag = "qwen3-embedding:0.6b"

    def __init__(self):
        self.digest = "c" * 64
        self.calls = []
        self.remote = False
        self.invalid_vector = None
        self.dimensions = 2
        self.on_embed = None

    def json(self, route, payload=None):
        self.calls.append((route, deepcopy(payload)))
        if route == "/api/tags":
            return {"models": [{"name": self.tag, "model": self.tag, "digest": self.digest,
                                "size": 123, "details": {"family": "synthetic"}}]}
        if route == "/api/show":
            return {"capabilities": ["embedding"], "model_info": {"synthetic.dimension": self.dimensions},
                    "remote_host": "https://cloud.invalid" if self.remote else ""}
        if route == "/api/embed":
            if self.on_embed:
                self.on_embed()
            vectors = []
            for text in payload["input"]:
                if self.invalid_vector is not None:
                    vectors.append(self.invalid_vector)
                elif self.dimensions != 2:
                    vectors.append([1.0] * self.dimensions)
                elif "bearing" in text.casefold():
                    vectors.append([0.0, 4.0])
                else:
                    vectors.append([3.0, 0.0])
            return {"model": payload["model"], "embeddings": vectors,
                    "prompt_eval_count": 7, "total_duration": 1234, "load_duration": 12}
        raise AssertionError(f"Unexpected API route: {route}")


def synthetic_inputs():
    document = {"id": "synthetic-brochure", "title": "Invented process controller",
                "source_file": "invented.pdf", "source_sha256": "a" * 64, "page_count": 2}
    chunks = []
    for identifier, text, kind, status in (
        ("control", "Feed composition changes require adaptive control.\n\nUnreviewed numerical claim 19.7 percent.", "narrative", "source_text"),
        ("bearing", "Bearing lubrication and shaft inspection.", "narrative", "source_text"),
        ("caption", "Sch\u00e9ma de contr\u00f4le. \u041a\u043e\u043d\u0442\u0440\u043e\u043b\u044c \u043f\u043e\u0434\u0430\u0447\u0438.", "figure_caption", "source_text"),
        ("numeric-only", "Specimen output 19.7 percent.", "narrative", "source_text"),
        ("table-secret", "Private raw quantity 88.2 mm.", "table_context", "raw_table"),
    ):
        chunks.append({"id": identifier, "doc_id": document["id"], "page": 1, "text": text,
                       "kind": kind, "evidence_status": status, "source_sha256": document["source_sha256"],
                       "section": "Invented context", "citation": {"source_file": "invented.pdf", "page": 1}})
    return [document], chunks, [], []


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Path(self.temporary.name) / "synthetic.sqlite"
        self.index = Path(self.temporary.name) / "private-vectors.json"
        self.inputs = synthetic_inputs()
        build_store(self.store, *self.inputs)
        self.client = FakeClient()

    def build(self, **kwargs):
        return build_embedding_index(self.store, self.index, self.client, **kwargs)

    def test_rrf_combines_ranks_and_does_not_double_count_duplicates(self):
        rows = fuse_rankings(["a", "a", "b"], ["b", "c"])
        self.assertEqual([r["id"] for r in rows], ["b", "a", "c"])
        self.assertAlmostEqual(rows[0]["rrf_score"], 1 / 62 + 1 / 61)
        self.assertEqual(rows[0]["keyword_rank"], 2)
        self.assertEqual(rows[0]["dense_rank"], 1)
        self.assertEqual(fuse_rankings([], []), [])
        self.assertEqual([r["id"] for r in fuse_rankings(["z"], ["a"])], ["a", "z"])
        with self.assertRaises(ValueError):
            fuse_rankings(["a"], ["b"], k=True)

    def test_cosine_scale_invariance_and_invalid_vectors(self):
        self.assertAlmostEqual(cosine_similarity([3, 4], [6, 8]), 1)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 5]), 0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-2, 0]), -1)
        for bad in ([0, 0], [], [True, 0], [math.nan, 1], [math.inf, 1], [1], ["1", 0]):
            with self.subTest(vector=bad), self.assertRaises(ValueError):
                cosine_similarity([1, 0], bad)

    def test_build_retains_complete_narrative_and_unicode_but_never_raw_tables_or_assertion_authority(self):
        report = self.build(batch_size=2)
        self.assertEqual(report["chunks"], 4)
        self.assertEqual(report["dimensions"], 2)
        artifact = json.loads(self.index.read_text(encoding="utf-8"))
        self.assertEqual({e["id"] for e in artifact["entries"]}, {"control", "bearing", "caption", "numeric-only"})
        requests = [payload for route, payload in self.client.calls if route == "/api/embed"]
        self.assertEqual([len(p["input"]) for p in requests], [2, 2])
        texts = "\n".join(text for payload in requests for text in payload["input"])
        self.assertIn("19.7", texts)
        self.assertTrue(all(entry["numeric_assertion_authority"] is False for entry in artifact["entries"]))
        self.assertNotIn("88.2", texts)
        self.assertNotIn("Instruct:", texts)
        self.assertIn("\u041a\u043e\u043d\u0442\u0440\u043e\u043b\u044c", texts)
        self.assertIn("Product: Invented process controller", texts)
        self.assertIn("Section: Invented context\nPage: 1", texts)
        self.assertTrue(all(p["truncate"] is False for p in requests))
        self.assertTrue(check_embedding_index(self.store, self.index, self.client)["valid"])
        self.assertEqual(len([r for r, _ in self.client.calls if r == "/api/embed"]), 2)

    def test_search_preserves_current_source_and_applies_query_instruction(self):
        self.build()
        before = len(self.client.calls)
        hits = semantic_search(self.store, self.index, "bearing maintenance", limit=1, client=self.client)
        self.assertEqual(hits[0]["id"], "bearing")
        self.assertEqual(hits[0]["source_sha256"], "a" * 64)
        self.assertEqual(hits[0]["citation"]["page"], 1)
        self.assertAlmostEqual(hits[0]["semantic_score"], 1)
        request = next(p for r, p in self.client.calls[before:] if r == "/api/embed")
        self.assertEqual(request["input"], [f"Instruct: {QUERY_INSTRUCTION}\nQuery: bearing maintenance"])
        self.assertNotIn("trusted", hits[0])

    def test_changed_store_is_rejected_before_query_inference(self):
        self.build()
        with closing(sqlite3.connect(self.store)) as connection:
            connection.execute("INSERT INTO metadata VALUES ('changed','yes')")
            connection.commit()
        count = len([r for r, _ in self.client.calls if r == "/api/embed"])
        with self.assertRaisesRegex(ValueError, "store bytes changed"):
            semantic_search(self.store, self.index, "control", client=self.client)
        self.assertEqual(len([r for r, _ in self.client.calls if r == "/api/embed"]), count)

    def test_changed_model_digest_is_rejected_before_query_inference(self):
        self.build()
        self.client.digest = "d" * 64
        count = len([r for r, _ in self.client.calls if r == "/api/embed"])
        with self.assertRaisesRegex(ValueError, "model digest"):
            semantic_search(self.store, self.index, "control", client=self.client)
        self.assertEqual(len([r for r, _ in self.client.calls if r == "/api/embed"]), count)

    def test_content_corruption_fails_closed(self):
        self.build()
        artifact = json.loads(self.index.read_text(encoding="utf-8"))
        artifact["entries"][0]["embedded_text"] = "changed text"
        self.index.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            check_embedding_index(self.store, self.index, self.client)

    def test_invalid_model_vectors_never_publish_an_index(self):
        for vector in ([0, 0], [math.nan, 0], [True, 2]):
            with self.subTest(vector=vector):
                self.client.invalid_vector = vector
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.index.exists())

    def test_model_or_database_change_during_build_never_publishes(self):
        def mutate_model():
            self.client.digest = "e" * 64
        self.client.on_embed = mutate_model
        with self.assertRaisesRegex(ValueError, "model changed"):
            self.build()
        self.assertFalse(self.index.exists())
        self.client.digest = "c" * 64
        def mutate_database():
            with closing(sqlite3.connect(self.store)) as connection:
                connection.execute("INSERT INTO metadata VALUES ('changed_during_embedding','yes')")
                connection.commit()
        self.client.on_embed = mutate_database
        with self.assertRaisesRegex(ValueError, "store changed"):
            self.build()
        self.assertFalse(self.index.exists())

    def test_local_only_endpoint_and_remote_backed_model_rejection(self):
        self.client.endpoint = "https://api.external.invalid"
        with self.assertRaisesRegex(ValueError, "loopback"):
            self.build()
        self.assertEqual(self.client.calls, [])
        self.client.endpoint = "http://127.0.0.1:11434"
        with self.assertRaisesRegex(ValueError, "cloud"):
            self.build(modeltag="synthetic-cloud:latest")
        self.client.remote = True
        with self.assertRaisesRegex(ValueError, "Remote-backed"):
            self.build()
        self.assertFalse(any(route == "/api/embed" for route, _ in self.client.calls))

    def test_bounded_document_input_keeps_original_snapshot_and_reports_truncation(self):
        self.store.unlink()
        inputs = synthetic_inputs()
        inputs[1][0]["text"] = "Adaptive feed control. " * 200
        build_store(self.store, *inputs)
        report = self.build()
        self.assertEqual(report["truncated_chunks"], 1)
        artifact = json.loads(self.index.read_text(encoding="utf-8"))
        entry = next(e for e in artifact["entries"] if e["id"] == "control")
        self.assertEqual(len(entry["embedded_text"]), 2000)
        self.assertGreater(entry["omitted_input_characters"], 0)
        self.assertEqual(len(entry["source_chunk_sha256"]), 64)
        self.assertTrue(check_embedding_index(self.store, self.index, self.client)["valid"])

    def test_overwrite_and_empty_corpus_are_refused_without_inference(self):
        self.build()
        before = len(self.client.calls)
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(len(self.client.calls), before)
        self.store.unlink()
        build_store(self.store, [], [], [], [])
        with self.assertRaisesRegex(ValueError, "no eligible"):
            build_embedding_index(self.store, self.index.parent / "empty.json", self.client)
        self.assertEqual(len(self.client.calls), before)


if __name__ == "__main__":
    unittest.main()
