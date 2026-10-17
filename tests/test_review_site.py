"""Synthetic review-site path and HTML handling checks; no brochure facts."""

import json
from pathlib import Path
import tempfile
import unittest

from metso_assistant.review_site import build_review_site, render_review_html, safe_artifact_path


class ReviewSiteTests(unittest.TestCase):
    def test_artifact_paths_reject_urls_drives_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            root.mkdir()
            (root / "native table.html").write_text("<table></table>", encoding="utf-8")
            self.assertEqual(safe_artifact_path(root, "native table.html"), (root / "native table.html").resolve())
            for reference in ("../outside.html", "/absolute.html", "C:relative.html", "https://example.test/x", "javascript:alert(1)", "\\server\\share", "a/../native table.html"):
                with self.subTest(reference=reference), self.assertRaises(ValueError):
                    safe_artifact_path(root, reference)

    def test_source_text_escaped_native_html_linked_and_inputs_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / ".pilot-cache"
            prepared, raw = private / "prepared", private / "raw-stable"
            prepared.mkdir(parents=True)
            page_dir = raw / "docling" / "sample-doc" / "page-001"
            page_dir.mkdir(parents=True)
            prefix = "sample-doc/page-001/"
            (page_dir / "source.png").write_bytes(b"synthetic image")
            (page_dir / "native table.html").write_text('<script>UNTRUSTED_NATIVE_PAYLOAD</script><table></table>', encoding="utf-8")
            (page_dir / "cells.json").write_text("[]", encoding="utf-8")
            (page_dir / "native.json").write_text("{}", encoding="utf-8")
            raw_page = {"source_sha256": "synthetic-source", "page": 1, "page_size_points": [100, 100], "source_render_file": prefix + "source.png", "native_json_file": prefix + "native.json", "tables": [{"id": "table-01", "html_file": prefix + "native table.html"}]}
            (page_dir / "page.json").write_text(json.dumps(raw_page), encoding="utf-8")
            (raw / "docling" / "manifest.json").write_text(json.dumps({"configuration_sha256": "synthetic-config"}), encoding="utf-8")
            table = {"table_id": "table-01", "doc_id": "sample-doc", "source": {"file": "sample.pdf", "sha": "synthetic-source", "page": 1}, "table_fingerprint": "synthetic-table", "review_status": "pending", "parser_fingerprint": {"primary": "docling", "raw_configuration_sha256": {"docling": "synthetic-config"}}, "raw_evidence": {"parser": "docling", "source_render_file": prefix + "source.png", "cells_file": prefix + "cells.json"}, "issues": [{"code": "synthetic", "message": '<img src="x" onerror="attack()">'}], "facts": [{"configuration": "Fake model", "parameter": "Fake property", "raw_cell_text": "<script>alert(1)</script>", "review_status": "pending", "status": "reported", "issues": [], "review_blockers": [], "source_refs": [{"cell_bbox": [10, 10, 20, 20]}]}]}
            values = {"tables": [table], "pages": [{"doc_id": "sample-doc", "page": 1, "citation": "sample.pdf, page 1", "raw_artifacts": {"docling": {"source_render_file": prefix + "source.png"}}}], "documents": [{"id": "sample-doc", "title": 'Fake <source> "title"'}]}
            for stem, value in values.items():
                (prepared / f"{stem}.json").write_text(json.dumps(value), encoding="utf-8")
            before = {path: path.read_bytes() for directory in (prepared, raw) for path in directory.rglob("*") if path.is_file()}
            output = private / "review" / "index.html"
            content, report = render_review_html(prepared, raw, output)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", content)
            self.assertNotIn("<script>alert(1)</script>", content)
            self.assertNotIn("UNTRUSTED_NATIVE_PAYLOAD", content)
            self.assertIn("native%20table.html", content)
            self.assertIn("Locate cell", content)
            self.assertEqual(report["approvals_written"], 0)
            built = build_review_site(prepared, raw, output, private_root=private)
            self.assertEqual(built["tables"], 1)
            self.assertTrue(output.is_file())
            self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))
            with self.assertRaises(ValueError):
                build_review_site(prepared, raw, root / "public.html", private_root=private)


if __name__ == "__main__":
    unittest.main()
