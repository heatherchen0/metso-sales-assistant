"""Immutable page-level extraction of the local brochure PDFs.

Run Docling with the main environment and MinerU with environments/mineru/.venv.
Both adapters require the already pinned local weights and block remote Python
connections. No source numbers, units, headers or associations are repaired.
All facade artifact paths are relative to the extraction output root.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

from .model_assets import docling_models, file_sha256, mineru_models, model_identity, package_versions
from .review import source_code_hash


SCHEMA_VERSION = 2
EXTRACTOR_VERSION = "2.0.0"
COORDINATES = "PDF points, top-left origin: x0,y0,x1,y1"
DOCLING_PINS = {"docling-slim": "2.138.0", "docling-core": "2.101.1", "docling-parse": "7.22.3", "docling-ibm-models": "4.0.3", "torch": "2.14.1", "pypdfium2": "5.14.0", "Pillow": "12.3.0"}
MINERU_PINS = {"mineru": "4.0.12", "docvortex": "0.5.16", "mineru-vl-utils": "2.0.7", "mineru-llama-cpp": "0.1.3", "onnxruntime": "1.31.0", "pypdfium2": "5.14.0"}
DOCLING_REVISIONS = {"docling-project/docling-models": ("v2.3.0", "fc0f2d45e2218ea24bce5045f58a389aed16dc23"), "docling-project/docling-layout-heron": ("main", "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8")}
DOCLING_WEIGHTS = {
    "models--docling-project--docling-layout-heron/snapshots/8f39ad3c0b4c58e9c2d2c84a38465abf757272d8/model.safetensors": "00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
    "models--docling-project--docling-models/snapshots/fc0f2d45e2218ea24bce5045f58a389aed16dc23/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors": "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9",
}
MINERU_REVISIONS = {"MinerU-4_models_onnx": "358310b4f64b95f9fefc372ad899356e4111f376", "MinerU2.5-Pro-2605-1.2B-GGUF": "9185688a0495e1577d521a757c7c0b62dd38ca48"}

def parse_pages(value: str) -> tuple[int, ...] | str:
    """Parse an explicit one-based selection or all physical pages."""
    value = value.strip().lower()
    if value == "all":
        return value
    selected: set[int] = set()
    try:
        for item in value.split(","):
            if "-" in item:
                first, last = (int(part.strip()) for part in item.split("-"))
                if first < 1 or last < first:
                    raise ValueError
                # A typo should not allocate a huge range before inspecting the PDF.
                if last - first > 10000:
                    raise ValueError
                selected.update(range(first, last + 1))
            else:
                number = int(item.strip())
                if number < 1:
                    raise ValueError
                selected.add(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use all or one-based pages such as 2,5-8.") from exc
    if not selected:
        raise argparse.ArgumentTypeError("At least one page is required.")
    return tuple(sorted(selected))

def select_files(data_dir: Path, names: list[str] | None) -> list[Path]:
    brochure_dir = data_dir / "brochures" if (data_dir / "brochures").is_dir() else data_dir
    if not brochure_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {brochure_dir}")
    available = {path.stem.lower(): path for path in sorted(brochure_dir.glob("*.pdf"))}
    if names:
        selected = []
        for name in names:
            if any(character in name for character in "/\\:"):
                raise ValueError("--files accepts brochure filenames or stems, not paths.")
            stem = Path(name).stem.lower() if name.lower().endswith(".pdf") else name.lower()
            selected.append(available.get(stem, brochure_dir / f"{stem}.pdf"))
        return list(dict.fromkeys(selected))
    if not available:
        raise ValueError(f"No PDF files found in {brochure_dir}")
    return list(available.values())

def make_docling_converter(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    # Set caches before importing Hugging Face/Docling modules, which may read
    # environment defaults at import time. Never repurpose HOME or USERPROFILE.
    cache = args.output_dir.resolve() / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    if args.models_dir:
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(args.models_dir.resolve())
    else:
        # Native first-use download goes into the explicitly selected HF cache.
        # A pre-existing artifact setting must not redirect extraction elsewhere.
        os.environ.pop("DOCLING_ARTIFACTS_PATH", None)
    # Check local table/layout dependencies before converting the first page.
    for module in ("cv2", "rtree"):
        importlib.import_module(module)
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TableStructureOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = args.ocr
    options.do_table_structure = True
    options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE, do_cell_matching=not args.no_cell_matching,
    )
    options.generate_page_images = True
    options.generate_picture_images = True
    options.images_scale = args.dpi / 72.0
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=args.threads)
    if args.models_dir:
        options.artifacts_path = str(args.models_dir.resolve())
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    settings = options.model_dump(mode="json")
    # The base-typed nested field can serialize as {} in Pydantic; keep the
    # actual TableFormer mode and cell-matching setting explicit in manifests.
    settings["table_structure_options"] = options.table_structure_options.model_dump(mode="json")
    return converter, settings

def configure_local(args: argparse.Namespace) -> tuple[Path, dict]:
    """Isolate config, reject inherited service settings, and disable LLM aids."""
    args.home.mkdir(parents=True, exist_ok=True)
    models = args.models_dir or args.home / "models"
    models = models.resolve()
    config = {
        "model": {
            "source": "local", "base_dir": str(models),
            "small_backend": args.small_backend,
            "vlm": {"engine": args.vlm_engine, "server_url": "", "api_key": "", "model": "", "max_concurrency": 1},
        },
        "llm_aided": {"features": {"title_leveling": False, "cross_page_table_cell_merge": False}},
    }
    config_path = args.home / f"benchmark-{args.small_backend}-{args.vlm_engine}.yaml"
    # JSON is valid YAML and avoids a separate configuration dependency.
    write_json(config_path, config)
    for name in list(os.environ):
        if name.startswith("MINERU_"):
            del os.environ[name]
    controls = {
        "MINERU_HOME": str(args.home), "MINERU_CONFIG": str(config_path),
        "MINERU_MODEL_SOURCE": "local", "MINERU_MODEL_SMALL_BACKEND": args.small_backend,
        "MINERU_MODEL_VLM_ENGINE": args.vlm_engine,
        "MINERU_TABLE_DEVICE": args.table_device,
        "MINERU_INTRA_OP_NUM_THREADS": str(args.threads),
        "MINERU_INTER_OP_NUM_THREADS": "1", "MINERU_PDF_RENDER_THREADS": "1",
        "MINERU_MODEL_VLM_MAX_CONCURRENCY": "1",
        "HF_HOME": str(args.home / "huggingface"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    }
    os.environ.update(controls)
    return config_path, {"config": config, "environment": controls}


class ExistingOutputError(ValueError):
    """Existing evidence is inconsistent or incomplete and must stay untouched."""


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def relative(path: Path, output: Path) -> str:
    return path.resolve().relative_to(output.resolve()).as_posix()



def local_only() -> None:
    """Reject Python network connections outside loopback, including DNS lookup."""
    def audit(event: str, values: tuple[Any, ...]) -> None:
        host = None
        if event == "socket.connect" and len(values) > 1 and isinstance(values[1], tuple):
            host = values[1][0]
        elif event == "socket.getaddrinfo" and values:
            host = values[0]
        if isinstance(host, bytes):
            host = host.decode("ascii", errors="replace")
        if host not in (None, "", "localhost", "127.0.0.1", "::1"):
            raise PermissionError("Brochure extraction permits local models and loopback only.")
    sys.addaudithook(audit)





def canonical_bbox(value: Any, size: list[float], *, normalized: bool = False) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        box = [value["l"], value["t"], value["r"], value["b"]]
        if str(value.get("coord_origin", "TOPLEFT")).upper().endswith("BOTTOMLEFT"):
            box[1], box[3] = size[1] - box[1], size[1] - box[3]
    else:
        box = list(value)
    if len(box) != 4:
        raise ValueError("A native bbox must have four coordinates.")
    if normalized:
        box = [box[0] * size[0], box[1] * size[1], box[2] * size[0], box[3] * size[1]]
    return box


def stable_docling_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize the observed unordered set, preserving every ordered list.

    Docling serializes this disabled model-spec set to a randomly ordered JSON
    list in each interpreter. Sorting its members changes no inference setting.
    This deliberately does not recurse over or sort other configuration lists.
    """
    result = json.loads(json.dumps(settings))
    if result.get("do_chart_extraction") is False:
        model_spec = result.get("chart_extraction_options", {}).get("model_spec", {})
        engines = model_spec.get("supported_engines")
        if engines is not None:
            if not isinstance(engines, list) or not all(isinstance(item, str) for item in engines):
                raise TypeError("Unexpected Docling supported_engines serialization; refusing a guessed canonicalization.")
            model_spec["supported_engines"] = sorted(engines)
    return result


def portable_settings(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, Any]:
    """Normalize only known placement fields; preserve every inference option."""
    result = json.loads(json.dumps(settings))
    models = result["models"]
    if args.parser == "docling":
        if set(result) != {"pipeline", "models", "device", "configuration_canonicalization"}:
            raise ValueError("Unexpected Docling settings fields require identity review")
        hub = args.models_cache.resolve() / "huggingface/hub"
        if Path(models["cache_root"]).resolve() != hub or result["pipeline"].get("artifacts_path") is not None:
            raise ValueError("Unexpected Docling model/cache path override")
    else:
        if args.parser != "mineru" or set(result) != {"tier", "parse_mode", "image_analysis", "controls", "models", "hardware_note"}:
            raise ValueError("Unexpected MinerU settings fields require identity review")
        controls = result["controls"]
        if set(controls) != {"config", "environment"} or set(controls["config"]) != {"model", "llm_aided"}:
            raise ValueError("Unexpected MinerU control fields require identity review")
        model = controls["config"]["model"]
        if set(model) != {"source", "base_dir", "small_backend", "vlm"}:
            raise ValueError("Unexpected MinerU model controls require identity review")
        home = args.mineru_home.resolve()
        if Path(models["directory"]).resolve() != home / "models" or Path(model["base_dir"]).resolve() != home / "models":
            raise ValueError("Unexpected MinerU model path override")
        environment = controls["environment"]
        path_fields = {"MINERU_HOME": (home, "."),
                       "MINERU_CONFIG": (home / f"benchmark-{model['small_backend']}-{model['vlm']['engine']}.yaml",
                                         f"benchmark-{model['small_backend']}-{model['vlm']['engine']}.yaml"),
                       "HF_HOME": (home / "huggingface", "huggingface")}
        scalar_fields = {"MINERU_MODEL_SOURCE", "MINERU_MODEL_SMALL_BACKEND", "MINERU_MODEL_VLM_ENGINE",
                         "MINERU_TABLE_DEVICE", "MINERU_INTRA_OP_NUM_THREADS", "MINERU_INTER_OP_NUM_THREADS",
                         "MINERU_PDF_RENDER_THREADS", "MINERU_MODEL_VLM_MAX_CONCURRENCY", "HF_HUB_OFFLINE",
                         "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"}
        if set(environment) != set(path_fields) | scalar_fields:
            raise ValueError("Unexpected MinerU environment fields require identity review")
        for name, (expected, relative_path) in path_fields.items():
            if Path(environment[name]).resolve() != expected:
                raise ValueError(f"Unexpected MinerU path override: {name}")
            environment[name] = relative_path
        model["base_dir"] = "models"
    result["models"] = model_identity(args.parser, models)
    return result


class HtmlCells(HTMLParser):
    """Read explicit HTML cell spans; retain text without numeric normalization."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.row = -1
        self.column = 0
        self.current: dict[str, Any] | None = None
        self.cells: list[dict[str, Any]] = []
        self.occupied: set[tuple[int, int]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row += 1
            self.column = 0
        elif tag in ("td", "th"):
            if self.current is not None:
                raise ValueError("Nested table cells are not supported; native HTML remains preserved.")
            if self.row < 0:
                raise ValueError("Native HTML cell has no enclosing row.")
            attributes = dict(attrs)
            row_span = int(attributes.get("rowspan") or 1)
            col_span = int(attributes.get("colspan") or 1)
            if min(row_span, col_span) < 1:
                raise ValueError("Zero/negative native spans require review; no inferred replacement.")
            while (self.row, self.column) in self.occupied:
                self.column += 1
            self.current = {"text": "", "row_index": self.row, "col_index": self.column, "row_span": row_span, "col_span": col_span, "bbox": None, "native_cell": {"tag": tag, "attributes": attributes}, "_parts": []}
        elif tag == "br" and self.current is not None:
            self.current["_parts"].append("\n")

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current["_parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.current is not None:
            cell = self.current
            cell["text"] = "".join(cell.pop("_parts"))
            for row in range(cell["row_index"], cell["row_index"] + cell["row_span"]):
                for col in range(cell["col_index"], cell["col_index"] + cell["col_span"]):
                    if (row, col) in self.occupied:
                        raise ValueError("Native HTML spans overlap; no repaired grid emitted.")
                    self.occupied.add((row, col))
            self.cells.append(cell)
            self.column += cell["col_span"]
            self.current = None

    def document(self) -> dict[str, Any]:
        if self.current is not None:
            raise ValueError("Unclosed native HTML cell.")
        return {"cells": self.cells, "row_count": max((row for row, _ in self.occupied), default=-1) + 1, "column_count": max((col for _, col in self.occupied), default=-1) + 1, "coordinate_system": None, "header_inference": False, "value_normalization": False}


def render_source(source: Path, page_number: int, target: Path, dpi: int, pdfium: Any) -> dict[str, Any]:
    with pdfium.PdfDocument(str(source)) as document:
        page = document[page_number - 1]
        try:
            size = list(page.get_size())
            bitmap = page.render(scale=dpi / 72)
            try:
                image = bitmap.to_pil()
                try:
                    image.save(target)
                    pixels = list(image.size)
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            page.close()
    return {"page_size_points": size, "source_render_pixel_size": pixels, "source_render_dpi": dpi}


def load_docling(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    model_evidence = docling_models(args.models_cache)
    options = argparse.Namespace(output_dir=args.models_cache, models_dir=None, ocr=False, no_cell_matching=False, dpi=args.dpi, threads=args.threads)
    converter, settings = make_docling_converter(options)
    if settings.get("enable_remote_services") or settings.get("allow_external_plugins"):
        raise ValueError("Docling must use local stages without remote services/plugins.")
    return converter, {"pipeline": stable_docling_settings(settings), "models": model_evidence, "device": "cpu", "configuration_canonicalization": ["disabled chart_extraction_options.model_spec.supported_engines set sorted for stable JSON fingerprint"]}


def extract_docling(converter: Any, source: Path, number: int, target: Path, output: Path, size: list[float]) -> dict[str, Any]:
    from docling_core.types.doc.base import ImageRefMode
    started = time.perf_counter()
    conversion = converter.convert(source, page_range=(number, number))
    seconds = time.perf_counter() - started
    document = conversion.document
    if list(document.pages) != [number]:
        raise ValueError(f"Docling physical-page mapping mismatch: expected {number}, got {list(document.pages)}.")
    native = target / "native.json"
    # Embedded native images retain SDK output without absolute artifact URIs.
    document.save_as_json(native, image_mode=ImageRefMode.EMBEDDED)
    text = target / "text.md"
    text.write_text(document.export_to_markdown(), encoding="utf-8")
    tables = []
    for index, table in enumerate(document.tables, 1):
        base = target / f"table-{index:02d}"
        native_item = table.model_dump(mode="json")
        write_json(base.with_suffix(".native.json"), native_item)
        base.with_suffix(".html").write_text(table.export_to_html(doc=document), encoding="utf-8")
        cells = [{"text": cell.text, "row_index": cell.start_row_offset_idx, "col_index": cell.start_col_offset_idx, "row_span": cell.row_span, "col_span": cell.col_span, "bbox": canonical_bbox(cell.bbox.model_dump(mode="json") if cell.bbox else None, size), "native_cell": cell.model_dump(mode="json")} for cell in table.data.table_cells]
        write_json(base.with_suffix(".cells.json"), {"cells": cells, "row_count": table.data.num_rows, "column_count": table.data.num_cols, "coordinate_system": COORDINATES, "header_inference": False, "value_normalization": False})
        provenance = [item.model_dump(mode="json") for item in table.prov]
        bbox = provenance[0].get("bbox") if provenance else None
        record = {"id": base.name, "html_file": relative(base.with_suffix(".html"), output), "native_json_file": relative(base.with_suffix(".native.json"), output), "cells_file": relative(base.with_suffix(".cells.json"), output), "bbox": canonical_bbox(bbox, size), "coordinate_system": COORDINATES, "native_bbox": bbox, "native_provenance": provenance, "native_json_path": f"#/tables/{index - 1}"}
        image = table.get_image(document)
        if image is not None:
            image.save(base.with_suffix(".png"))
            record["image_file"] = relative(base.with_suffix(".png"), output)
        tables.append(record)
    figures = []
    for index, picture in enumerate(document.pictures, 1):
        base = target / f"figure-{index:02d}"
        provenance = [item.model_dump(mode="json") for item in picture.prov]
        bbox = provenance[0].get("bbox") if provenance else None
        write_json(base.with_suffix(".native.json"), picture.model_dump(mode="json"))
        record = {"id": base.name, "order": index - 1, "bbox": canonical_bbox(bbox, size), "coordinate_system": COORDINATES, "native_bbox": bbox, "native_provenance": provenance, "native_json_path": f"#/pictures/{index - 1}", "native_json_file": relative(base.with_suffix(".native.json"), output)}
        image = picture.get_image(document)
        if image is not None:
            image.save(base.with_suffix(".png"))
            record["image_file"] = relative(base.with_suffix(".png"), output)
        figures.append(record)
    errors = [error.model_dump(mode="json") for error in conversion.errors]
    status = conversion.status.value
    return {"text_file": relative(text, output), "native_json_file": relative(native, output), "tables": tables, "figures": figures, "conversion_seconds": round(seconds, 4), "native_status": status, "native_page_numbers": list(document.pages), "status": "success" if status == "success" and not errors else "partial", "errors": errors}


def load_mineru(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    settings_args = argparse.Namespace(home=args.mineru_home, models_dir=args.mineru_home / "models", small_backend="onnx", vlm_engine="llama-cpp", table_device="cpu", threads=args.threads)
    _, controls = configure_local(settings_args)
    model_evidence = mineru_models(args.mineru_home)
    from mineru.config import VlmConfig, config
    from mineru.model.download import verify_model_repo
    from mineru.model.registry import model_repos_for_tier
    from mineru.parser import MinerUParser
    from mineru.parser.writer import FileBasedDataWriter
    if config.model.source != "local" or config.model.vlm.server_url or config.llm_aided.features.title_leveling or config.llm_aided.features.cross_page_table_cell_merge:
        raise ValueError("MinerU configuration violates local-only extraction.")
    for repo in model_repos_for_tier("standard", small_backend="onnx", vlm_engine="llama-cpp"):
        if not verify_model_repo(repo).ready:
            raise FileNotFoundError(f"Required local MinerU models incomplete: {repo.name}")
    vlm = VlmConfig(engine="llama-cpp", server_url="", api_key="", model="", max_concurrency=1)
    parser = MinerUParser(tier="standard", parse_mode="auto", image_analysis=True, vlm_config=vlm)
    return parser, FileBasedDataWriter, {"tier": "standard", "parse_mode": "auto", "image_analysis": True, "controls": controls, "models": model_evidence, "hardware_note": "CPU ONNX tables; packaged llama.cpp VLM may use local GPU offload, matching the pilot Standard configuration."}


def walk_nodes(value: Any, location: str = "$") -> Iterator[tuple[dict[str, Any], str]]:
    if isinstance(value, dict):
        yield value, location
        for key, child in value.items():
            yield from walk_nodes(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_nodes(child, f"{location}[{index}]")


def extract_mineru(parser: Any, writer: Any, source: Path, number: int, target: Path, output: Path, size: list[float]) -> dict[str, Any]:
    started = time.perf_counter()
    result = parser.parse(source, page_range=str(number))
    seconds = time.perf_counter() - started
    raw = result.to_dict()
    write_json(target / "unmaterialized.native.json", raw)
    indices = [page.get("page_idx") for page in raw.get("pages", [])]
    if indices != [number - 1]:
        raise ValueError(f"MinerU original page mapping mismatch: expected {number - 1}, got {indices}.")
    native_dir = target / "native"
    result.save(writer(str(native_dir)))
    native = native_dir / "middle_json.json"
    materialized = json.loads(native.read_text(encoding="utf-8"))
    markdown = sorted(native_dir.glob("*.md"))
    if len(markdown) != 1:
        raise ValueError(f"Expected one native MinerU markdown file, found {len(markdown)}.")
    tables, figures = [], []
    for block_index, block in enumerate(materialized["pages"][0]["blocks"]):
        block_path = f"$.pages[0].blocks[{block_index}]"
        if block.get("type") == "table":
            for node, location in walk_nodes(block, block_path):
                content = node.get("content")
                if not isinstance(content, str) or not re.search(r"<table\b", content, re.I):
                    continue
                index = len(tables) + 1
                base = target / f"table-{index:02d}"
                base.with_suffix(".html").write_text(content, encoding="utf-8")
                write_json(base.with_suffix(".native.json"), {"native_block": block, "native_json_path": location})
                cells = HtmlCells()
                try:
                    cells.feed(content)
                    cell_document = cells.document()
                    cell_status = "parsed"
                except ValueError as exc:
                    cell_document = {"cells": [], "row_count": None, "column_count": None, "coordinate_system": None, "header_inference": False, "value_normalization": False, "errors": [{"type": type(exc).__name__, "message": str(exc)}], "note": "Native HTML is preserved; no repaired grid emitted."}
                    cell_status = "unparsed"
                write_json(base.with_suffix(".cells.json"), cell_document)
                bbox = node.get("bbox", block.get("bbox"))
                record = {"id": base.name, "html_file": relative(base.with_suffix(".html"), output), "native_json_file": relative(base.with_suffix(".native.json"), output), "cells_file": relative(base.with_suffix(".cells.json"), output), "cells_status": cell_status, "bbox": canonical_bbox(bbox, size, normalized=True), "coordinate_system": COORDINATES, "native_bbox": bbox, "native_coordinate_system": "normalized0..1, top-left origin", "native_json_path": location}
                if node.get("image_path"):
                    image = (native_dir / node["image_path"]).resolve()
                    if not image.is_relative_to(native_dir.resolve()) or not image.is_file():
                        raise ValueError("MinerU table image path is missing or escapes its native output.")
                    record["image_file"] = relative(image, output)
                tables.append(record)
        else:
            for node, location in walk_nodes(block, block_path):
                if not node.get("image_path"):
                    continue
                image = (native_dir / node["image_path"]).resolve()
                if not image.is_relative_to(native_dir.resolve()) or not image.is_file():
                    raise ValueError("MinerU figure component path is missing or escapes its native output.")
                index = len(figures) + 1
                base = target / f"figure-{index:02d}"
                write_json(base.with_suffix(".native.json"), {"native_component": node, "native_parent_block": block, "native_json_path": location})
                bbox = node.get("bbox", block.get("bbox"))
                figures.append({"id": base.name, "order": block_index, "component_order": index - 1, "kind": node.get("type"), "image_file": relative(image, output), "native_json_file": relative(base.with_suffix(".native.json"), output), "bbox": canonical_bbox(bbox, size, normalized=True), "coordinate_system": COORDINATES, "native_bbox": bbox, "native_coordinate_system": "normalized0..1, top-left origin", "native_json_path": location})
    return {"text_file": relative(markdown[0], output), "native_json_file": relative(native, output), "model_output_file": relative(native_dir / "model_output.json", output) if (native_dir / "model_output.json").is_file() else None, "tables": tables, "figures": figures, "conversion_seconds": round(seconds, 4), "native_page_indices": indices, "native_metadata": raw.get("metadata"), "native_extensions": raw.get("extensions"), "status": "success", "errors": []}


def artifact_inventory(target: Path, output: Path) -> list[dict[str, Any]]:
    return [{"path": relative(path, output), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in sorted(target.rglob("*")) if path.is_file() and path.name not in {"page.json", "page.json.tmp"}]


def validate_existing_page(target: Path, output: Path, source_sha256: str, number: int, config_sha256: str) -> dict[str, Any] | None:
    if not target.exists():
        return None
    if not (target / "page.json").is_file():
        raise ExistingOutputError(f"Incomplete existing page unit: {target}. Use a new output root; no evidence overwritten.")
    page = json.loads((target / "page.json").read_text(encoding="utf-8"))
    recorded_integrity = page.get("record_sha256")
    if not recorded_integrity or recorded_integrity != fingerprint({key: value for key, value in page.items() if key != "record_sha256"}):
        raise ExistingOutputError(f"Existing page metadata failed its integrity SHA: {target}.")
    expected = (source_sha256, number, config_sha256, "success")
    actual = (page.get("source_sha256"), page.get("page"), page.get("configuration_sha256"), page.get("status"))
    if expected != actual:
        raise ExistingOutputError(f"Existing page provenance/configuration/status mismatch: {target}.")
    if not page.get("artifacts"):
        raise ExistingOutputError(f"Existing page has no verified artifact inventory: {target}.")
    expected_paths = set()
    for item in page["artifacts"]:
        path = (output / item["path"]).resolve()
        if not path.is_relative_to(target.resolve()) or not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ExistingOutputError(f"Existing artifact failed path/size/SHA validation: {path}")
        expected_paths.add(path)
    actual_paths = {path.resolve() for path in target.rglob("*") if path.is_file() and path.name != "page.json"}
    if actual_paths != expected_paths:
        raise ExistingOutputError(f"Existing page has extra/missing files: {target}.")
    return page


def extract_page(args: argparse.Namespace, sdk: Any, writer: Any, source: Path, source_digest: str, number: int, config_digest: str, pdfium: Any) -> dict[str, Any]:
    target = args.output / source.stem / f"page-{number:03d}"
    reused = validate_existing_page(target, args.output, source_digest, number, config_digest)
    if reused is not None:
        return {"page": number, "page_file": relative(target / "page.json", args.output), "page_file_sha256": file_sha256(target / "page.json"), "status": "success", "reused": True, "conversion_seconds": 0, "tables": len(reused.get("tables", [])), "figures": len(reused.get("figures", []))}
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    page: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "path_base": "output_root", "source_file": source.name, "source_sha256": source_digest, "source_path": str(source), "page": number, "page_numbering": "physical, one-based", "citation": f"{source.name}, page {number}", "parser": args.parser, "configuration_sha256": config_digest, "status": "error", "tables": [], "figures": [], "errors": [], "started_at": timestamp()}
    try:
        if file_sha256(source) != source_digest:
            raise ValueError("Source PDF changed after the corpus inventory was recorded.")
        render = target / "source.png"
        page.update(render_source(source, number, render, args.dpi, pdfium))
        page["source_render_file"] = relative(render, args.output)
        with (target / "parser.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                if args.parser == "docling":
                    page.update(extract_docling(sdk, source, number, target, args.output, page["page_size_points"]))
                else:
                    page.update(extract_mineru(sdk, writer, source, number, target, args.output, page["page_size_points"]))
            except Exception:
                traceback.print_exc()
                raise
    except Exception as exc:
        page["status"] = "error"
        page["errors"].append({"phase": "page_extraction", "type": type(exc).__name__, "message": str(exc)})
    page["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    page["artifacts"] = artifact_inventory(target, args.output)
    page["record_sha256"] = fingerprint(page)
    write_json(target / "page.json", page)
    return {"page": number, "page_file": relative(target / "page.json", args.output), "page_file_sha256": file_sha256(target / "page.json"), "status": page["status"], "reused": False, "conversion_seconds": page.get("conversion_seconds"), "elapsed_seconds": page["elapsed_seconds"], "tables": len(page["tables"]), "figures": len(page["figures"]), "errors": page["errors"]}


def extract_brochures(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    versions = package_versions(args.parser)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})
    temp = args.output.parent / "cache" / args.parser / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.update({"TMP": str(temp), "TEMP": str(temp), "TORCH_HOME": str(temp.parent / "torch")})
    local_only()
    import pypdfium2 as pdfium
    sources = select_files(args.data_dir, args.files)
    corpus = []
    for source in sources:
        with pdfium.PdfDocument(str(source)) as document:
            count = len(document)
        pages = list(range(1, count + 1)) if args.pages == "all" else list(args.pages)
        if any(page > count for page in pages):
            raise ValueError(f"Selected page outside {source.name}'s {count} physical pages.")
        corpus.append({"source_file": source.name, "source_sha256": file_sha256(source), "physical_page_count": count, "selected_pages": pages})
    setup_started = time.perf_counter()
    writer = None
    if args.parser == "docling":
        sdk, settings = load_docling(args)
    else:
        sdk, writer, settings = load_mineru(args)
    configuration = {"schema_version": SCHEMA_VERSION, "identity_protocol": 2,
                     "extractor_version": EXTRACTOR_VERSION, "extractor_sha256": source_code_hash(Path(__file__)),
                     "asset_engine_sha256": source_code_hash(Path(__file__).with_name("model_assets.py")),
                     "identity_engine_sha256": source_code_hash(Path(__file__).with_name("review.py")),
                     "parser": args.parser, "dpi": args.dpi, "threads": args.threads, "packages": versions,
                     "settings": portable_settings(args, settings), "coordinate_system": COORDINATES,
                     "policy": "raw values preserved; no numeric/unit/header/association repair; offline local models"}
    config_digest = fingerprint(configuration)
    manifest_path = args.output / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if existing_manifest is not None and (existing_manifest.get("configuration_sha256") != config_digest or existing_manifest.get("corpus") != corpus):
        raise ExistingOutputError("Existing extraction root has different configuration/source selection. Use a new output root.")
    for source, item in zip(sources, corpus):
        for number in item["selected_pages"]:
            validate_existing_page(args.output / source.stem / f"page-{number:03d}", args.output, item["source_sha256"], number, config_digest)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {"schema_version": SCHEMA_VERSION, "operation": "brochure_extraction", "path_base": "output_root", "run_id": run_id, "started_at": timestamp(), "parser": args.parser, "configuration_sha256": config_digest, "configuration": configuration, "corpus": corpus, "physical_page_count": sum(item["physical_page_count"] for item in corpus), "selected_page_count": sum(len(item["selected_pages"]) for item in corpus), "python": platform.python_version(), "platform": platform.platform(), "setup_seconds": round(time.perf_counter() - setup_started, 4), "status": "running", "pages": [], "errors": []}
    report["configuration_provenance"] = {"settings": settings,
        "source_file_sha256": {name: file_sha256(Path(__file__).with_name(name))
                               for name in ("brochure_extract.py", "model_assets.py", "review.py")},
        "identity_policy": "Known model placement paths and download bookkeeping are recorded here; immutable model/config bytes and inference options bind configuration identity."}
    run_path = args.output / "runs" / f"{run_id}.json"
    progress_target = run_path if existing_manifest is not None else manifest_path
    source_snapshot = args.output / f"extractor-{report['configuration_provenance']['source_file_sha256']['brochure_extract.py']}.py"
    if not source_snapshot.exists():
        source_snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(progress_target, report)
    started = time.perf_counter()
    try:
        manager = sdk if args.parser == "mineru" else contextlib.nullcontext()
        with manager:
            for source, item in zip(sources, corpus):
                for number in item["selected_pages"]:
                    page = extract_page(args, sdk, writer, source, item["source_sha256"], number, config_digest, pdfium)
                    report["pages"].append(page)
                    write_json(progress_target, report)
                    print(f"Brochure {args.parser}: {source.name} page {number}: {page['status']} {'(reused)' if page['reused'] else '(' + str(page.get('elapsed_seconds', 0)) + 's)'}", flush=True)
    except Exception as exc:
        report["errors"].append({"phase": "run", "type": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
    report["finished_at"] = timestamp()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    report["summary"] = {"pages_requested": report["selected_page_count"], "pages_completed": len(report["pages"]), "successful_pages": sum(page["status"] == "success" for page in report["pages"]), "reused_pages": sum(page["reused"] for page in report["pages"]), "table_count": sum(page.get("tables", 0) for page in report["pages"]), "figure_count": sum(page.get("figures", 0) for page in report["pages"]), "conversion_seconds": round(sum(page.get("conversion_seconds", 0) or 0 for page in report["pages"]), 4)}
    report["status"] = "success" if not report["errors"] and report["summary"]["successful_pages"] == report["selected_page_count"] else "partial_or_error"
    write_json(progress_target, report)
    if progress_target != run_path:
        write_json(run_path, report)
    print(json.dumps({"manifest": str(progress_target), "status": report["status"], "summary": report["summary"]}))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser", choices=("docling", "mineru"), default="docling")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Extraction root itself; default .pilot-cache/assistant/raw/<parser>.")
    parser.add_argument("--models-cache", type=Path, default=Path(".pilot-cache"), help="Existing Docling cache parent containing huggingface/hub (never downloaded by this command).")
    parser.add_argument("--mineru-home", type=Path, default=Path(".pilot-cache/mineru-home"))
    parser.add_argument("--files", nargs="+", help="Optional diagnostic subset; default all twelve PDFs.")
    parser.add_argument("--pages", type=parse_pages, default="all", help="all or one-based pages; default all 73 physical pages.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1 or not 72 <= args.dpi <= 600:
        parser.error("Threads must be positive and DPI between 72 and 600.")
    args.output = (args.output or Path(f".pilot-cache/assistant/raw/{args.parser}")).resolve()
    args.data_dir, args.models_cache, args.mineru_home = args.data_dir.resolve(), args.models_cache.resolve(), args.mineru_home.resolve()
    try:
        report = extract_brochures(args)
    except Exception as exc:
        print(f"Brochure extraction setup refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
