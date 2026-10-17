"""Asset setup tests use invented bytes, repositories and commits; no network."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metso_assistant.model_assets import AssetRepository, main, prepare_assets, repositories


COMMIT = "a" * 40
PAYLOAD = b"independently invented synthetic weight bytes"
CHECKSUM = hashlib.sha256(PAYLOAD).hexdigest()


def fixture(local=False):
    return AssetRepository("synthetic/model", COMMIT,
        "mineru-home/models/SyntheticModel" if local else f"huggingface/hub/models--synthetic--model/snapshots/{COMMIT}",
        {"weights.bin": CHECKSUM}, reference=None if local else "main",
        local_directory=local, completion_marker=local)


class AssetSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.cache = self.workspace / ".pilot-cache"
        self.repo = fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def call(self, *, download=False, fn=None, repo=None):
        with patch("metso_assistant.model_assets.repositories", return_value=(repo or self.repo,)):
            return prepare_assets(download=download, workspace_root=self.workspace, snapshot_download_fn=fn)

    def put(self, repo=None, payload=PAYLOAD, reference=True, metadata=False, marker=False):
        repo = repo or self.repo
        directory = self.cache / repo.directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "weights.bin").write_bytes(payload)
        if reference and repo.reference:
            ref = self.cache / "huggingface/hub/models--synthetic--model/refs/main"
            ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(COMMIT, encoding="utf-8")
        if metadata:
            path = directory / ".cache/huggingface/download/weights.bin.metadata"
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(COMMIT + "\nsynthetic-etag\n0\n", encoding="utf-8")
        if marker:
            (directory / ".mineru_complete").touch()
        return directory

    def test_default_missing_asset_check_never_imports_download_or_writes(self):
        with patch("socket.socket.connect", side_effect=AssertionError("Unexpected network")):
            report = self.call(fn=lambda **_: self.fail("Unexpected downloader"))
        self.assertFalse(report["ready"])
        self.assertFalse(self.cache.exists())
        self.assertFalse(report["network_download_requested"])

    def test_check_only_verifies_hashes_and_alias_without_mutation(self):
        self.put(); before = {p.relative_to(self.cache).as_posix(): p.read_bytes() for p in self.cache.rglob("*") if p.is_file()}
        report = self.call()
        after = {p.relative_to(self.cache).as_posix(): p.read_bytes() for p in self.cache.rglob("*") if p.is_file()}
        self.assertTrue(report["ready"])
        self.assertEqual(report["verified_files"], 1)
        self.assertEqual(before, after)

    def test_missing_alias_does_not_trigger_default_download(self):
        self.put(reference=False)
        report = self.call()
        self.assertFalse(report["ready"])
        self.assertTrue(report["repositories"][0]["assets_valid"])
        self.assertFalse(report["repositories"][0]["reference_valid"])

    def test_explicit_prepare_repairs_alias_without_redownloading_existing_weights(self):
        self.put(reference=False)
        report = self.call(download=True, fn=lambda **_: self.fail("Existing assets must not download"))
        self.assertTrue(report["ready"])
        self.assertEqual(report["downloaded_repositories"], [])
        self.assertEqual(len(report["reference_aliases_written"]), 1)

    def test_explicit_download_uses_commit_pin_workspace_cache_and_restores_environment(self):
        before = dict(os.environ); calls = []
        def fake(**kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["revision"], COMMIT)
            self.assertEqual(kwargs["cache_dir"], str(self.cache / "huggingface/hub"))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
            self.assertFalse(kwargs["token"])
            self.assertEqual(os.environ.get("HOME"), before.get("HOME"))
            self.assertEqual(os.environ.get("USERPROFILE"), before.get("USERPROFILE"))
            return self.put(reference=False)
        report = self.call(download=True, fn=fake)
        self.assertTrue(report["ready"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(os.environ, before)

    def test_bad_downloaded_hash_never_gets_reference_alias(self):
        before = dict(os.environ)
        def fake(**kwargs):
            return self.put(reference=False, payload=b"incorrect bytes")
        with self.assertRaisesRegex(ValueError, "failed pinned"):
            self.call(download=True, fn=fake)
        self.assertFalse((self.cache / "huggingface/hub/models--synthetic--model/refs/main").exists())
        self.assertEqual(os.environ, before)

    def test_corruption_requests_forced_download_of_the_invalid_asset(self):
        self.put(payload=b"corruption")
        def fake(**kwargs):
            self.assertTrue(kwargs["force_download"])
            self.assertIn("weights.bin", kwargs["allow_patterns"])
            return self.put()
        self.assertTrue(self.call(download=True, fn=fake)["ready"])

    def test_wrong_returned_directory_never_gets_alias(self):
        def fake(**kwargs):
            self.put(reference=False)
            return self.cache / "different-snapshot"
        with self.assertRaisesRegex(ValueError, "different snapshot"):
            self.call(download=True, fn=fake)
        self.assertFalse((self.cache / "huggingface/hub/models--synthetic--model/refs/main").exists())

    def test_cache_path_outside_workspace_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            prepare_assets(cache_root=self.workspace / "outside", workspace_root=self.workspace)

    def test_optional_local_profile_requires_provider_commit_and_completion_marker(self):
        repo = fixture(local=True)
        self.put(repo, reference=False, metadata=True)
        self.assertFalse(self.call(repo=repo)["ready"])
        report = self.call(repo=repo, download=True, fn=lambda **_: self.fail("Verified files already exist"))
        self.assertTrue(report["ready"])
        self.assertEqual(len(report["completion_markers_written"]), 1)
        self.assertEqual(report["repositories"][0]["provider_commits"], [COMMIT])

    def test_wrong_provider_metadata_prevents_completion_marker(self):
        repo = fixture(local=True); directory = self.put(repo, reference=False, metadata=True)
        metadata = directory / ".cache/huggingface/download/weights.bin.metadata"
        metadata.write_text("b" * 40 + "\nwrong-etag\n0", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provider-commit"):
            self.call(repo=repo, download=True, fn=lambda **_: directory)
        self.assertFalse((directory / ".mineru_complete").exists())

    def test_failed_download_restores_temporary_environment(self):
        before = dict(os.environ)
        def fake(**kwargs):
            raise OSError("Synthetic download failure")
        with self.assertRaises(OSError):
            self.call(download=True, fn=fake)
        self.assertEqual(os.environ, before)

    def test_profiles_pin_complete_docling_snapshots_for_extraction_fingerprints(self):
        specs = repositories("docling")
        self.assertEqual(len(specs), 2)
        self.assertTrue(all(len(s.revision) == 40 for s in specs))
        self.assertTrue(any("/fast/" in path for s in specs for path in s.files))
        self.assertTrue(any(path.endswith("preprocessor_config.json") for s in specs for path in s.files))
        self.assertTrue(all(s.exact_snapshot_inventory for s in specs))
        self.assertEqual(sum(len(s.files) for s in specs), 14)

    def test_extra_snapshot_file_cannot_inherit_cached_configuration_identity(self):
        repo = AssetRepository(**{**self.repo.__dict__, "exact_snapshot_inventory": True})
        directory = self.put(repo); (directory / "untracked.txt").write_text("unexpected", encoding="utf-8")
        self.assertFalse(self.call(repo=repo)["ready"])
        with self.assertRaisesRegex(ValueError, "Unexpected snapshot"):
            self.call(repo=repo, download=True, fn=lambda **_: self.fail("No automatic deletion/repair"))

    def test_entire_snapshot_is_requested_when_preparing_fresh_exact_inventory(self):
        repo = AssetRepository(**{**self.repo.__dict__, "exact_snapshot_inventory": True})
        def fake(**kwargs):
            self.assertNotIn("allow_patterns", kwargs)
            return self.put(repo, reference=False)
        self.assertTrue(self.call(repo=repo, download=True, fn=fake)["ready"])

    def test_cli_check_only_exit_status_and_no_writes(self):
        with patch("metso_assistant.model_assets.prepare_assets", return_value={"ready": False}) as mocked:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--check-only"]), 1)
            self.assertFalse(mocked.call_args.kwargs["download"])


if __name__ == "__main__":
    unittest.main()
