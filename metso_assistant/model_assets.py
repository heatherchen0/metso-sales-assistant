"""Check pinned extraction assets offline; download only with --download.

Run this setup command separately from the deliberately offline extractor.
All cache/ref/marker writes are constrained to this checkout's .pilot-cache.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import importlib.metadata
import re
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator, Mapping

DOCLING_PINS = {"docling-slim": "2.138.0", "docling-core": "2.101.1", "docling-parse": "7.22.3", "docling-ibm-models": "4.0.3", "torch": "2.14.1", "pypdfium2": "5.14.0", "Pillow": "12.3.0"}

MINERU_PINS = {"mineru": "4.0.12", "docvortex": "0.5.16", "mineru-vl-utils": "2.0.7", "mineru-llama-cpp": "0.1.3", "onnxruntime": "1.31.0", "pypdfium2": "5.14.0"}

DOCLING_REVISIONS = {"docling-project/docling-models": ("v2.3.0", "fc0f2d45e2218ea24bce5045f58a389aed16dc23"), "docling-project/docling-layout-heron": ("main", "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8")}

DOCLING_WEIGHTS = {
    "models--docling-project--docling-layout-heron/snapshots/8f39ad3c0b4c58e9c2d2c84a38465abf757272d8/model.safetensors": "00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
    "models--docling-project--docling-models/snapshots/fc0f2d45e2218ea24bce5045f58a389aed16dc23/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors": "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9",
}

MINERU_REVISIONS = {"MinerU-4_models_onnx": "358310b4f64b95f9fefc372ad899356e4111f376", "MinerU2.5-Pro-2605-1.2B-GGUF": "9185688a0495e1577d521a757c7c0b62dd38ca48"}

MINERU_WEIGHTS = {
    "MinerU-4_models_onnx/Layout/PP-DocLayoutV2/inference.onnx": "cd540dc296ff3115fe78efa65b68501e6a8dc74b198acc9834a725ffaa095aac",
    "MinerU-4_models_onnx/MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.onnx": "20a32595c2b30282dcd069e6295e9fad96115697beb66acb9b681b814006d193",
    "MinerU-4_models_onnx/OCR/paddleocr/ch_PP-OCRv6_small_rec_infer.onnx": "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    "MinerU-4_models_onnx/OCR/paddleocr/ch_PP-OCRv6_tiny_det_infer.onnx": "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
    "MinerU-4_models_onnx/OCR/paddleocr/seal_PP-OCRv4_det_infer.onnx": "3e190df84542c2756975febe8f73603b587a23c7f7badfdd3fc8a986540d5bc0",
    "MinerU-4_models_onnx/Table/PP-LCNet_x1_0_table_cls.onnx": "c84bf1d79c1c74d534b5b12adb14dd12151c42f7ae3e4be4f1042b830f80b949",
    "MinerU-4_models_onnx/Table/slanet-plus.onnx": "d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b",
    "MinerU-4_models_onnx/Table/unet.onnx": "0ea48d3a17e35ef5c2e498a5e799566073234d39b1079ca21d9f4fafe73c6d20",
    "MinerU2.5-Pro-2605-1.2B-GGUF/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf": "3f3523429c880d675c0330cb3de78240452bc40a961d037e24f91ab8488f03e7",
    "MinerU2.5-Pro-2605-1.2B-GGUF/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf": "b6c08ab50352677f7396af45ee60682f353f55488e26911d8154bfc37514d3f8",
}

def model_inventory(models_dir: Path) -> dict:
    """Record real assets and provider commit markers without probing the internet."""
    records = []
    commits = set()
    commits_by_repo: dict[str, set[str]] = {}
    manifests = []
    if not models_dir.is_dir():
        return {"directory": str(models_dir), "exists": False, "files": [], "provider_commits": []}
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        relative = str(path.relative_to(models_dir))
        record = {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        records.append(record)
        if path.name.endswith(".metadata") and path.stat().st_size < 10000:
            first = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if first and re.fullmatch(r"[a-fA-F0-9]{40,64}", first[0]):
                commits.add(first[0])
                commits_by_repo.setdefault(path.relative_to(models_dir).parts[0], set()).add(first[0])
        if path.name in {"manifest.json", "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt"} and path.stat().st_size < 2_000_000:
            manifests.append({"path": relative, "sha256": record["sha256"]})
    return {
        "directory": str(models_dir), "exists": True, "files": records,
        "provider_commits": sorted(commits),
        "provider_commits_by_repo": {name: sorted(values) for name, values in commits_by_repo.items()},
        "cards_and_manifests": manifests,
        "revision_policy": "Installed MinerU models-download CLI has no revision flag. Reproduce resolved provider commits and verify asset hashes; package pin alone does not pin weights.",
    }

def package_versions(parser: str) -> dict[str, str | None]:
    names = set(DOCLING_PINS if parser == "docling" else MINERU_PINS)
    names.update(("pydantic", "huggingface-hub", "numpy", "Pillow", "pandas", "opencv-python-headless", "rtree", "scipy"))
    result = {}
    for name in sorted(names):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    for name, expected in (DOCLING_PINS if parser == "docling" else MINERU_PINS).items():
        if result[name] != expected:
            raise ValueError(f"{parser} requires pinned {name}=={expected}; installed {result[name]!r}. Use its prepared environment.")
    return result

def verify_weights(root: Path, weights: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for name, expected in sorted(weights.items()):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned local model missing: {path}. Prefetch separately; extraction never downloads weights.")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Pinned weight SHA256 mismatch: {path}")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": actual})
    return records

def docling_models(cache: Path) -> dict[str, Any]:
    hub = cache / "huggingface" / "hub"
    snapshots = []
    for repo, (ref, revision) in DOCLING_REVISIONS.items():
        repository = hub / ("models--" + repo.replace("/", "--"))
        ref_file = repository / "refs" / ref
        if not ref_file.is_file() or ref_file.read_text(encoding="utf-8").strip() != revision:
            raise ValueError(f"Offline {repo} reference {ref} must resolve to pinned revision {revision}.")
        directory = repository / "snapshots" / revision
        # Preserve the logical snapshot filename when HF uses symlinks to blobs.
        snapshots.append({"repository": repo, "reference": ref, "revision": revision, "files": [{"path": path.relative_to(hub).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(_inside(path, hub))} for path in sorted(directory.rglob("*")) if path.is_file()]})
    return {"cache_root": str(hub), "snapshots": snapshots, "verified_weights": verify_weights(hub, DOCLING_WEIGHTS)}

def mineru_models(home: Path) -> dict[str, Any]:
    root = home / "models"
    inventory = model_inventory(root)
    for repo, revision in MINERU_REVISIONS.items():
        if inventory.get("provider_commits_by_repo", {}).get(repo) != [revision]:
            raise ValueError(f"Local {repo} metadata must resolve exclusively to pinned revision {revision}.")
    inventory["verified_weights"] = verify_weights(root, MINERU_WEIGHTS)
    return inventory


@dataclass(frozen=True)
class AssetRepository:
    repo_id: str
    revision: str
    directory: str
    files: Mapping[str, str]
    reference: str | None = None
    local_directory: bool = False
    completion_marker: bool = False
    exact_snapshot_inventory: bool = False


DOCLING_CONFIGS = {
    "docling-project/docling-layout-heron": {
        "config.json": "fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",
        "preprocessor_config.json": "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1"},
    "docling-project/docling-models": {
        "config.json": "9c34024dc28ff47b75818f415e769809798c29bf9bde6f2ccc63a4acb62396d9",
        "model_artifacts/tableformer/accurate/tm_config.json": "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd"},
}
DOCLING_SNAPSHOT_EXTRAS = {
    "docling-project/docling-models": {
        ".gitattributes": "360aba3ff7ce75396f13f0a77925fe9e79997e8d4bac064595ebb00c82700386",
        ".gitignore": "b44b404189eaff3631c470946e7350c2e88cead304d332a8ad50325e6f56e95d",
        "README.md": "d17f233378eff1240b623b36da76ee8b40afcca05d505949713bf03f7e00822a",
        "model_artifacts/tableformer/fast/tableformer_fast.safetensors": "3119563aab5a7c96fda4d621119b63fd8806272b86c30936d15507616422f718",
        "model_artifacts/tableformer/fast/tm_config.json": "dca6762508dddfae6d57d6cb4ef822c6000119dff0f3b6489db7413118c2622a"},
    "docling-project/docling-layout-heron": {
        ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        "README.md": "175700839bc7808eac6af1d0c23e4f483606ab2276fe01122f4093e61a1a65b6",
        "docling_heron_400.png": "e7f78610372b32a7938e480d2c7fa1c3037ee170bd82282a5bd026232f6e6f9e"},
}
MINERU_CONFIGS = {
    "Layout/PP-DocLayoutV2/inference.yml": "93100e0fb2cd76b9d246a853deea85e191d8333b2cfca7be76e82e4c4ebfa819",
    "MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M_inference.yml": "87b5f3d7f2b2fe553627d77b37f496608ca150ebd0ef62d362591edca47b5538",
    "OCR/paddleocr/ch_PP-OCRv6_small_rec_inference.yml": "c0541b06832948d7d3052e9ac6b572c6063d79ecd800493f4ec14d1b901c16d7",
    "OCR/paddleocr/ch_PP-OCRv6_tiny_det_inference.yml": "a1fa68ab31e29252936328e728633180908f997023d9eacb81f175e326676554",
    "OCR/paddleocr/seal_PP-OCRv4_det_inference.yml": "e5b8dd5a727602bdd526d9db152cc767bc761198c453460df7c8f7ae59d2ea3c",
}


def repositories(profile: str = "docling") -> tuple[AssetRepository, ...]:
    selected = []
    if profile not in {"docling", "mineru-standard", "both"}:
        raise ValueError("Unknown extraction model profile")
    if profile in {"docling", "both"}:
        for repo_id, (reference, revision) in DOCLING_REVISIONS.items():
            prefix = "models--" + repo_id.replace("/", "--") + "/snapshots/" + revision + "/"
            files = {name.removeprefix(prefix): digest for name, digest in DOCLING_WEIGHTS.items() if name.startswith(prefix)}
            files.update(DOCLING_CONFIGS[repo_id])
            files.update(DOCLING_SNAPSHOT_EXTRAS[repo_id])
            selected.append(AssetRepository(repo_id, revision, "huggingface/hub/" + prefix.rstrip("/"), files, reference,
                                            exact_snapshot_inventory=True))
    if profile in {"mineru-standard", "both"}:
        for name, repo_id in (("MinerU-4_models_onnx", "opendatalab/MinerU-4_models_onnx"),
                              ("MinerU2.5-Pro-2605-1.2B-GGUF", "jinzhenj/MinerU2.5-Pro-2605-1.2B-GGUF")):
            files = {path.removeprefix(name + "/"): digest for path, digest in MINERU_WEIGHTS.items() if path.startswith(name + "/")}
            if name == "MinerU-4_models_onnx":
                files.update(MINERU_CONFIGS)
            selected.append(AssetRepository(repo_id, MINERU_REVISIONS[name], "mineru-home/models/" + name,
                                            files, local_directory=True, completion_marker=name.endswith("GGUF")))
    return tuple(selected)


def model_identity(parser: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind actual model/config bytes, excluding download transport bookkeeping."""
    if parser not in {"docling", "mineru"}:
        raise ValueError("Unknown extraction parser")
    profile = "docling" if parser == "docling" else "mineru-standard"
    specs = repositories(profile)
    if parser == "docling":
        snapshots = evidence["snapshots"]
        expected = {(spec.repo_id, spec.reference, spec.revision) for spec in specs}
        actual = {(item["repository"], item["reference"], item["revision"]) for item in snapshots}
        if actual != expected or len(snapshots) != len(specs):
            raise ValueError("Docling snapshot identity differs from pinned repositories")
        files = [file for item in snapshots for file in item["files"]]
    else:
        if evidence.get("provider_commits_by_repo") != {name: [revision] for name, revision in MINERU_REVISIONS.items()}:
            raise ValueError("MinerU provider revisions differ from pinned repositories")
        files = evidence["files"]
    inventory = {}
    for file in files:
        name = file["path"].replace("\\", "/")
        if name.startswith("/") or re.match(r"^[a-zA-Z]:", name) or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError("Model inventory must contain safe relative paths")
        if name in inventory:
            raise ValueError("Duplicate model inventory path")
        inventory[name] = file
    identity, required = [], set()
    for spec in specs:
        prefix = spec.directory.removeprefix("huggingface/hub/" if parser == "docling" else "mineru-home/models/")
        selected = []
        for name, expected_sha in sorted(spec.files.items()):
            path = prefix + "/" + name
            required.add(path)
            file = inventory.get(path)
            if (file is None or file.get("sha256") != expected_sha
                    or type(file.get("bytes")) is not int or file["bytes"] < 0):
                raise ValueError(f"Pinned model/config asset is missing or changed: {path}")
            selected.append({"path": name, "bytes": file["bytes"], "sha256": file["sha256"]})
        identity.append({"repository": spec.repo_id, "reference": spec.reference,
                         "revision": spec.revision, "files": selected})
    for name in set(inventory) - required:
        parts = name.split("/")
        transport = (len(parts) >= 4 and parts[1:3] == [".cache", "huggingface"]
                     and (len(parts) == 4 and parts[3] in {".gitignore", "CACHEDIR.TAG"}
                          or parts[3] == "download" and len(parts) > 4 and parts[-1].endswith(".metadata")
                          or parts[3] == "trees" and len(parts) == 5 and parts[4] == MINERU_REVISIONS.get(parts[0], "") + ".json"))
        incidental = (parser == "mineru" and len(parts) > 1 and parts[0] in MINERU_REVISIONS
                      and (transport or len(parts) == 2 and parts[1] in {
                          ".mineru_complete", ".gitattributes", "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt"}))
        if not incidental:
            raise ValueError(f"Unrecognized model asset outside pinned inventory: {name}")
    return identity


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes the workspace cache: {path}")
    return resolved


def _cache(cache_root: str | Path, workspace_root: str | Path | None) -> Path:
    workspace = Path(workspace_root).resolve() if workspace_root is not None else Path(__file__).resolve().parents[1]
    cache = Path(cache_root)
    if not cache.is_absolute():
        cache = workspace / cache
    return _inside(cache, workspace / ".pilot-cache")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_path(cache: Path, repo: AssetRepository) -> Path | None:
    if repo.reference is None:
        return None
    return _inside(cache / "huggingface/hub" / ("models--" + repo.repo_id.replace("/", "--")) / "refs" / repo.reference, cache)


def _check_repo(cache: Path, repo: AssetRepository) -> dict[str, Any]:
    directory = _inside(cache / repo.directory, cache)
    files = []
    for name, expected in sorted(repo.files.items()):
        path = directory / name
        _inside(path, cache)  # HF snapshot symlinks may point to workspace blobs.
        actual = file_sha256(path) if path.is_file() else None
        files.append({"path": path.relative_to(cache).as_posix(), "expected_sha256": expected,
                      "sha256": actual, "status": "verified" if actual == expected else "missing" if actual is None else "hash_mismatch"})
    ref = _reference_path(cache, repo)
    actual_ref = ref.read_text(encoding="utf-8").strip() if ref and ref.is_file() else None
    provider_commits = set()
    metadata_files = 0
    if repo.local_directory:
        metadata_root = _inside(directory / ".cache/huggingface/download", cache)
        for path in metadata_root.rglob("*.metadata"):
            _inside(path, cache)
            parts = path.read_text(encoding="utf-8").splitlines()
            if parts:
                provider_commits.add(parts[0]); metadata_files += 1
    metadata_valid = not repo.local_directory or provider_commits == {repo.revision}
    marker = _inside(directory / ".mineru_complete", cache)
    marker_valid = not repo.completion_marker or marker.is_file()
    assets_valid = all(file["status"] == "verified" for file in files)
    actual_files = set()
    if repo.exact_snapshot_inventory:
        for path in directory.rglob("*"):
            if path.is_file():
                _inside(path, cache)
                actual_files.add(path.relative_to(directory).as_posix())
    unexpected = sorted(actual_files - set(repo.files)) if repo.exact_snapshot_inventory else []
    inventory_valid = not unexpected
    refs_valid = ref is None or actual_ref == repo.revision
    return {"repo_id": repo.repo_id, "revision": repo.revision, "reference": repo.reference,
            "directory": directory.relative_to(cache).as_posix(), "files": files,
            "assets_valid": assets_valid, "reference_valid": refs_valid, "actual_reference": actual_ref,
            "provider_commits": sorted(provider_commits), "provider_metadata_files": metadata_files,
            "provider_metadata_valid": metadata_valid, "completion_marker_valid": marker_valid,
            "unexpected_snapshot_files": unexpected, "snapshot_inventory_valid": inventory_valid,
            "ready": assets_valid and refs_valid and metadata_valid and marker_valid and inventory_valid}


@contextmanager
def _download_environment(cache: Path) -> Iterator[None]:
    """Temporary process-local cache settings, restored even after a failure."""
    temporary = _inside(cache / "tmp/model-download", cache)
    temporary.mkdir(parents=True, exist_ok=True)
    variables = {"HF_HOME": str(cache / "huggingface"), "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "HF_XET_CACHE": str(cache / "huggingface/xet"), "HF_HUB_OFFLINE": "0", "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "TMP": str(temporary), "TEMP": str(temporary), "TMPDIR": str(temporary)}
    before = {key: os.environ.get(key) for key in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_reference(path: Path, revision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(revision, encoding="utf-8")
    temporary.replace(path)


def prepare_assets(profile: str = "docling", cache_root: str | Path = ".pilot-cache", *,
                   download: bool = False, workspace_root: str | Path | None = None,
                   snapshot_download_fn: Callable | None = None) -> dict[str, Any]:
    """Check locally by default. Explicit download fetches immutable revisions."""
    if type(download) is not bool:
        raise TypeError("download must be boolean")
    cache = _cache(cache_root, workspace_root)
    repos = repositories(profile)
    initial = [_check_repo(cache, repo) for repo in repos]
    downloaded, alias_writes, marker_writes = [], [], []
    if download:
        if any(not item["snapshot_inventory_valid"] for item in initial):
            raise ValueError("Unexpected snapshot files would change extraction fingerprints; use a fresh workspace cache")
        needs_network = [(repo, current) for repo, current in zip(repos, initial)
                         if not current["assets_valid"] or not current["provider_metadata_valid"]]
        if needs_network:
            with _download_environment(cache):
                if snapshot_download_fn is None:
                    try:
                        from huggingface_hub import snapshot_download
                    except ImportError as exc:
                        raise RuntimeError("Run uv sync --extra docling --locked before explicit model download") from exc
                    snapshot_download_fn = snapshot_download
                for repo, current in needs_network:
                    args: dict[str, Any] = {"repo_id": repo.repo_id, "revision": repo.revision,
                        "cache_dir": str(_inside(cache / "huggingface/hub", cache)), "token": False,
                        "local_files_only": False, "endpoint": "https://huggingface.co", "max_workers": 2}
                    if repo.local_directory:
                        args["local_dir"] = str(_inside(cache / repo.directory, cache))
                    if current["provider_metadata_valid"]:
                        needed = [Path(file["path"]).relative_to(repo.directory).as_posix()
                                  for file in current["files"] if file["status"] != "verified"]
                    else:
                        needed = list(repo.files)
                    args["allow_patterns"] = [*needed, "README.md", ".gitattributes"]
                    args["force_download"] = any(file["status"] == "hash_mismatch" for file in current["files"])
                    if repo.exact_snapshot_inventory and not args["force_download"]:
                        args.pop("allow_patterns")  # Entire immutable Docling repo.
                    returned = Path(snapshot_download_fn(**args))
                    expected_directory = _inside(cache / repo.directory, cache)
                    if _inside(returned, cache) != expected_directory:
                        raise ValueError("Downloader returned a different snapshot/directory than the pinned target")
                    downloaded.append(repo.repo_id)
        # Never advertise readiness before every requested profile asset and
        # provider commit has been verified; wrong payloads do not get aliases.
        checked = [_check_repo(cache, repo) for repo in repos]
        if any(not item["assets_valid"] or not item["provider_metadata_valid"] or not item["snapshot_inventory_valid"] for item in checked):
            raise ValueError("Downloaded assets failed pinned hash/provider-commit checks; refs/markers were not written")
        for repo, current in zip(repos, checked):
            ref = _reference_path(cache, repo)
            if ref is not None and not current["reference_valid"]:
                _write_reference(ref, repo.revision)
                alias_writes.append(ref.relative_to(cache).as_posix())
            if repo.completion_marker and not current["completion_marker_valid"]:
                marker = _inside(cache / repo.directory / ".mineru_complete", cache)
                marker.touch()
                marker_writes.append(marker.relative_to(cache).as_posix())
    final = [_check_repo(cache, repo) for repo in repos] if download else initial
    return {"schema_version": 1, "profile": profile, "cache_root": str(cache),
            "ready": all(item["ready"] for item in final), "network_download_requested": download,
            "downloaded_repositories": downloaded, "reference_aliases_written": alias_writes,
            "completion_markers_written": marker_writes, "repositories": final,
            "required_files": sum(len(item["files"]) for item in final),
            "verified_files": sum(file["status"] == "verified" for item in final for file in item["files"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("docling", "mineru-standard", "both"), default="docling")
    parser.add_argument("--cache-root", type=Path, default=Path(".pilot-cache"), help="Workspace cache parent, containing huggingface/hub and optional mineru-home")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true", help="Offline, read-only asset/alias verification (the default)")
    group.add_argument("--download", action="store_true", help="Explicitly fetch missing/incorrect pinned assets and prepare verified aliases")
    args = parser.parse_args(argv)
    try:
        report = prepare_assets(args.profile, args.cache_root, download=args.download)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0 if report["ready"] else 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Model setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
