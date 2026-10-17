"""Prepare the private corpus end to end; no assistant/model inference API is used.

Default preparation uses Docling plus the reviewed MinerU table fallback.
Install the locked environments first. --download-models explicitly fetches
pinned extraction assets; the extraction processes themselves run offline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from . import WORK_DIR
from .model_assets import prepare_assets


ROOT = Path(__file__).resolve().parents[1]


def prepare_workspace(data_dir=Path("data"), output=WORK_DIR, *, models_cache=Path(".pilot-cache"),
                      mineru_python=None, docling_only=False, download_models=False,
                      reviews=None, resolutions=None, run=subprocess.run):
    """Build a new workspace. Existing source data and prepared outputs stay intact."""
    data_dir, output, models_cache = (Path(path).resolve() for path in (data_dir, output, models_cache))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preparation output must be new or empty. Use the existing assistant, or choose a new --output directory.")
    if not list((data_dir / "brochures").glob("*.pdf")) or not list((data_dir / "HistoricalServices").glob("*.txt")):
        raise FileNotFoundError("Supply the original data/brochures PDFs and data/HistoricalServices reports before preparation.")
    interpreter = Path(mineru_python or ROOT / "environments/mineru/.venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")).resolve()
    if not docling_only:
        if not interpreter.is_file():
            raise FileNotFoundError("MinerU environment missing. Run uv sync --project environments/mineru --locked, or select --docling-only for reduced table coverage.")
    reviews = Path(reviews).resolve() if reviews else ROOT / "configs/table-reviews.json"
    resolutions = Path(resolutions).resolve() if resolutions else ROOT / "configs/normalization-resolutions.json"
    if not reviews.is_file() or not resolutions.is_file():
        raise FileNotFoundError("The source-bound table review and normalization configs are required.")
    assets = prepare_assets("docling" if docling_only else "both", models_cache, download=download_models)
    if not assets["ready"]:
        raise ValueError("Pinned extraction assets are missing or inconsistent. Run preparation with --download-models, or inspect model_assets --check-only.")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"; logs.mkdir()
    manifest = output / "preparation.json"
    report = {"schema_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "data_dir": str(data_dir), "output": str(output), "profile": "docling" if docling_only else "docling+mineru",
              "models_download_requested": download_models, "stages": []}

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def stage(name, interpreter, module, arguments, expected):
        log = logs / (name + ".log")
        command = [str(interpreter), "-B", "-m", module, *map(str, arguments)]
        print("Preparing " + name + "...", flush=True)
        entry = {"name": name, "command": command, "log": str(log), "status": "running"}
        report["stages"].append(entry); save()
        try:
            with log.open("xb") as handle:
                result = run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        except OSError as error:
            entry.update(status="failed", error=str(error)); save()
            raise
        entry["exit_code"] = result.returncode
        if result.returncode or not expected.is_file():
            entry["status"] = "failed"
            raise RuntimeError("Preparation stage " + name + " failed; inspect " + str(log))
        entry["status"] = "complete"; save()

    try:
        raw, services, prepared = output / "raw", output / "services", output / "prepared"
        stage("docling", sys.executable, "metso_assistant.brochure_extract",
              ["--parser", "docling", "--data-dir", data_dir, "--models-cache", models_cache,
               "--output", raw / "docling"], raw / "docling/manifest.json")
        if not docling_only:
            stage("mineru", interpreter, "metso_assistant.brochure_extract",
                  ["--parser", "mineru", "--data-dir", data_dir, "--mineru-home", models_cache / "mineru-home",
                   "--output", raw / "mineru"], raw / "mineru/manifest.json")
        stage("services", sys.executable, "metso_assistant.services",
              ["--data-dir", data_dir, "--output", services], services / "manifest.json")
        stage("knowledge", sys.executable, "metso_assistant.ingest",
              ["--data-dir", data_dir, "--raw-dir", raw, "--services-dir", services, "--output", prepared,
               "--reviews", reviews, "--resolutions", resolutions], prepared / "manifest.json")
        knowledge = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        if knowledge.get("status") != "prepared" or not (prepared / "knowledge.sqlite").is_file():
            raise RuntimeError("Preparation did not produce a complete knowledge store.")
        report.update(status="prepared", store=str(prepared / "knowledge.sqlite"), raw_dir=str(raw),
                      coverage={key: knowledge[key] for key in ("source_documents", "physical_pages", "tables",
                                                               "candidate_facts", "fact_review_status", "service_records")})
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=WORK_DIR)
    parser.add_argument("--models-cache", type=Path, default=Path(".pilot-cache"))
    parser.add_argument("--mineru-python", type=Path, help="Override the isolated MinerU Python interpreter")
    parser.add_argument("--docling-only", action="store_true", help="Reduced table coverage; source review may remain pending")
    parser.add_argument("--download-models", action="store_true", help="Explicitly download pinned missing/corrupt extraction assets")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--resolutions", type=Path)
    args = parser.parse_args(argv)
    try:
        report = prepare_workspace(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "store", "raw_dir", "coverage")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
