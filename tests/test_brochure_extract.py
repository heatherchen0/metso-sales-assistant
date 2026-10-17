"""Synthetic checks for raw extraction integrity and geometry, without source facts."""
import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metso_assistant.brochure_extract import (
    ExistingOutputError,
    HtmlCells,
    artifact_inventory,
    canonical_bbox,
    configure_local,
    fingerprint,
    parse_pages,
    portable_settings,
    select_files,
    stable_docling_settings,
    validate_existing_page,
    write_json,
)
from metso_assistant.model_assets import MINERU_REVISIONS, docling_models, model_inventory, package_versions, repositories, verify_weights


def portable_fixture(parser, root):
    args = argparse.Namespace(parser=parser, models_cache=root, mineru_home=root / "mineru-home")
    specs = repositories("docling" if parser == "docling" else "mineru-standard")
    files = lambda spec, prefix: [{"path": spec.directory.removeprefix(prefix) + "/" + name,
                                 "bytes": 1, "sha256": sha} for name, sha in spec.files.items()]
    if parser == "docling":
        return args, {"pipeline": {"artifacts_path": None, "do_table_structure": True},
                      "models": {"cache_root": str(root / "huggingface/hub"),
                                 "snapshots": [{"repository": spec.repo_id, "reference": spec.reference,
                                                "revision": spec.revision, "files": files(spec, "huggingface/hub/")}
                                               for spec in specs]},
                      "device": "cpu", "configuration_canonicalization": []}
    config_args = argparse.Namespace(home=args.mineru_home, models_dir=args.mineru_home / "models",
                                    small_backend="onnx", vlm_engine="llama-cpp", table_device="cpu", threads=4)
    _, controls = configure_local(config_args)
    inventory = [item for spec in specs for item in files(spec, "mineru-home/models/")]
    inventory.append({"path": "MinerU-4_models_onnx/.cache/huggingface/download/synthetic.metadata",
                      "bytes": 1, "sha256": "a" * 64})
    return args, {"tier": "standard", "parse_mode": "auto", "image_analysis": True,
                  "controls": controls, "models": {"directory": str(args.mineru_home / "models"), "files": inventory,
                      "provider_commits_by_repo": {name: [revision] for name, revision in MINERU_REVISIONS.items()}},
                  "hardware_note": "Synthetic local backend"}


class BrochureExtractionTests(unittest.TestCase):
    def test_docling_snapshot_labels_survive_huggingface_blob_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            hub = cache / "huggingface/hub"
            repository = hub / "models--synthetic--model"
            revision = "a" * 40
            logical = repository / "snapshots" / revision / "weights.bin"
            blob = repository / "blobs" / "synthetic-blob"
            reference = repository / "refs/main"
            for path in (logical, blob, reference):
                path.parent.mkdir(parents=True, exist_ok=True)
            logical.write_bytes(b"synthetic model bytes")
            blob.write_bytes(logical.read_bytes())
            reference.write_text(revision, encoding="utf-8")
            relative = logical.relative_to(hub).as_posix()
            original_resolve = Path.resolve
            # Simulate HF symlink resolution without Windows symlink privileges.
            def resolve(path, *args, **kwargs):
                return blob if path == logical else original_resolve(path, *args, **kwargs)
            with patch("metso_assistant.model_assets.DOCLING_REVISIONS", {"synthetic/model": ("main", revision)}), \
                 patch("metso_assistant.model_assets.DOCLING_WEIGHTS", {relative: hashlib.sha256(blob.read_bytes()).hexdigest()}), \
                 patch.object(Path, "resolve", resolve):
                models = docling_models(cache)
            self.assertEqual(models["snapshots"][0]["files"][0]["path"], relative)

    def test_configuration_identity_ignores_only_placement_and_download_bookkeeping(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            root = Path(folder)
            for parser in ("docling", "mineru"):
                with self.subTest(parser=parser):
                    first_args, first = portable_fixture(parser, root / "checkout-one")
                    second_args, second = portable_fixture(parser, root / "checkout-two")
                    if parser == "mineru":
                        second["models"]["files"][-1].update(bytes=9, sha256="b" * 64)
                        for file in second["models"]["files"]:
                            file["path"] = file["path"].replace("/", "\\")
                    before = json.dumps(first, sort_keys=True)
                    self.assertEqual(fingerprint(portable_settings(first_args, first)),
                                     fingerprint(portable_settings(second_args, second)))
                    self.assertEqual(json.dumps(first, sort_keys=True), before)

    def test_configuration_identity_detects_changed_active_assets_and_inference_options(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            root = Path(folder)
            for parser in ("docling", "mineru"):
                with self.subTest(parser=parser):
                    args, settings = portable_fixture(parser, root)
                    original = fingerprint(portable_settings(args, settings))
                    if parser == "docling":
                        settings["pipeline"]["do_table_structure"] = False
                        active = settings["models"]["snapshots"][0]["files"][0]
                    else:
                        settings["image_analysis"] = False
                        active = settings["models"]["files"][0]
                    self.assertNotEqual(fingerprint(portable_settings(args, settings)), original)
                    active["sha256"] = "0" * 64
                    with self.assertRaisesRegex(ValueError, "missing or changed"):
                        portable_settings(args, settings)

    def test_configuration_identity_rejects_unknown_path_overrides_and_model_assets(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            root = Path(folder)
            args, settings = portable_fixture("docling", root)
            settings["pipeline"]["artifacts_path"] = str(root / "other-models")
            with self.assertRaisesRegex(ValueError, "path override"):
                portable_settings(args, settings)
            args, settings = portable_fixture("mineru", root)
            settings["controls"]["environment"]["MINERU_CONFIG"] = str(root / "unexpected.yaml")
            with self.assertRaisesRegex(ValueError, "path override"):
                portable_settings(args, settings)
            args, settings = portable_fixture("mineru", root)
            settings["controls"]["environment"]["NEW_MODEL_PATH"] = str(root)
            with self.assertRaisesRegex(ValueError, "environment fields"):
                portable_settings(args, settings)
            args, settings = portable_fixture("mineru", root)
            settings["models"]["files"].append({"path": "MinerU-4_models_onnx/rogue.onnx", "bytes": 1, "sha256": "a" * 64})
            with self.assertRaisesRegex(ValueError, "Unrecognized model asset"):
                portable_settings(args, settings)

    def test_source_selection_is_explicit_and_confined_to_brochures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            brochures = root / "brochures"
            brochures.mkdir()
            first, second = brochures / "first.pdf", brochures / "second.pdf"
            first.touch(); second.touch()
            self.assertEqual(select_files(root, None), [first, second])
            self.assertEqual(select_files(root, ["SECOND.PDF", "second", "first"]), [second, first])
            for unsafe in ("../outside.pdf", "sub/file.pdf", "C:\\outside.pdf"):
                with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                    select_files(root, [unsafe])

    def test_page_selection_rejects_benchmark_alias_and_unbounded_ranges(self):
        self.assertEqual(parse_pages("all"), "all")
        self.assertEqual(parse_pages("5-7,2,5"), (2, 5, 6, 7))
        for invalid in ("benchmark", "0", "-1", "4-2", "1-10002", "1,,2"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_pages(invalid)

    def test_wrong_installed_parser_version_fails_before_extraction(self):
        with patch("metso_assistant.model_assets.importlib.metadata.version", return_value="wrong-version"):
            with self.assertRaisesRegex(ValueError, "requires pinned"):
                package_versions("docling")

    def test_weight_verification_refuses_missing_or_changed_local_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = b"invented parser weight"
            weights = {"weights.bin": hashlib.sha256(payload).hexdigest()}
            with self.assertRaises(FileNotFoundError):
                verify_weights(root, weights)
            (root / "weights.bin").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_weights(root, weights)
            (root / "weights.bin").write_bytes(payload)
            self.assertEqual(verify_weights(root, weights)[0]["sha256"], weights["weights.bin"])

    def test_mineru_setup_discards_inherited_service_and_api_configuration(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"MINERU_REMOTE_API_KEY": "synthetic", "MINERU_CONFIG": "remote.yaml"}):
            home = Path(folder)
            args = argparse.Namespace(home=home, models_dir=home / "models", small_backend="onnx",
                                      vlm_engine="llama-cpp", table_device="cpu", threads=4)
            path, settings = configure_local(args)
            self.assertNotIn("MINERU_REMOTE_API_KEY", os.environ)
            self.assertEqual(os.environ["MINERU_CONFIG"], str(path))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(settings["config"]["model"]["source"], "local")
            self.assertEqual(settings["config"]["model"]["vlm"]["server_url"], "")
            self.assertEqual(settings["config"]["model"]["vlm"]["api_key"], "")
            self.assertFalse(any(settings["config"]["llm_aided"]["features"].values()))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), settings["config"])

    def test_asset_inventory_retains_conflicting_provider_commits_and_ignores_locks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / "synthetic-model"
            repo.mkdir()
            for name, commit in (("first.metadata", "a" * 40), ("second.metadata", "b" * 40)):
                (repo / name).write_text(commit + "\nsynthetic-etag", encoding="utf-8")
            (repo / "busy.lock").write_bytes(b"lock")
            inventory = model_inventory(root)
            self.assertEqual(inventory["provider_commits_by_repo"], {"synthetic-model": ["a" * 40, "b" * 40]})
            self.assertEqual(len(inventory["files"]), 2)

    def test_configuration_canonicalizes_only_observed_unordered_set(self):
        first = {'do_chart_extraction': False, 'chart_extraction_options': {'model_spec': {'supported_engines': ['engine_b', 'engine_a']}}, 'ordered_values': ['z', 'a']}
        second = {'do_chart_extraction': False, 'chart_extraction_options': {'model_spec': {'supported_engines': ['engine_a', 'engine_b']}}, 'ordered_values': ['z', 'a']}
        self.assertEqual(fingerprint(stable_docling_settings(first)), fingerprint(stable_docling_settings(second)))
        self.assertEqual(stable_docling_settings(first)['ordered_values'], ['z', 'a'])
        self.assertEqual(first['chart_extraction_options']['model_spec']['supported_engines'], ['engine_b', 'engine_a'])

    def test_html_preserves_explicit_spans_and_missing_cell_bbox(self):
        parser = HtmlCells()
        parser.feed('<table><tr><th rowspan="2">Model</th><th colspan="2">Group</th></tr><tr><td>A</td><td>B</td></tr></table>')
        value = parser.document()
        self.assertEqual((value['row_count'], value['column_count']), (2, 3))
        self.assertEqual([(c['row_index'], c['col_index'], c['row_span'], c['col_span']) for c in value['cells']], [(0, 0, 2, 1), (0, 1, 1, 2), (1, 1, 1, 1), (1, 2, 1, 1)])
        self.assertTrue(all(c['bbox'] is None for c in value['cells']))
        self.assertFalse(value['header_inference'])

    def test_geometry_retains_original_coordinate_meaning(self):
        self.assertEqual(canonical_bbox({'l': 10, 't': 90, 'r': 30, 'b': 70, 'coord_origin': 'BOTTOMLEFT'}, [100, 100]), [10, 10, 30, 30])
        self.assertEqual(canonical_bbox([.1, .2, .3, .4], [200, 100], normalized=True), [20, 20, 60, 40])

    def test_reuse_rejects_changed_artifacts_metadata_and_config(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            target = output / 'sample' / 'page-001'
            target.mkdir(parents=True)
            artifact = target / 'text.md'
            artifact.write_text('Synthetic unchanged text', encoding='utf-8')
            page = {'source_sha256': 'source-digest', 'page': 1, 'configuration_sha256': 'config-digest', 'status': 'success', 'artifacts': artifact_inventory(target, output)}
            page['record_sha256'] = fingerprint(page)
            write_json(target / 'page.json', page)
            self.assertEqual(validate_existing_page(target, output, 'source-digest', 1, 'config-digest')['status'], 'success')
            with self.assertRaises(ExistingOutputError):
                validate_existing_page(target, output, 'source-digest', 1, 'another-config')
            artifact.write_text('Changed synthetic text', encoding='utf-8')
            with self.assertRaises(ExistingOutputError):
                validate_existing_page(target, output, 'source-digest', 1, 'config-digest')
            artifact.write_text('Synthetic unchanged text', encoding='utf-8')
            page['page'] = 2
            write_json(target / 'page.json', page)
            with self.assertRaises(ExistingOutputError):
                validate_existing_page(target, output, 'source-digest', 2, 'config-digest')


if __name__ == '__main__':
    unittest.main()
