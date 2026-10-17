"""Prepare a private, provenance-preserving knowledge base from original inputs.

Extraction is a separate, pinned offline step. This command validates every
cached artifact, normalizes source geometry, applies matched review decisions,
and builds an offline SQLite index. No agent or estimation model runs here.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any, TypedDict

from .brochure_extract import validate_existing_page
from .geometry import extract_geometry_pages, match_geometry_table
from .review import apply_reviews, content_hash, fingerprint_candidate, source_code_hash, validate_manifest
from .resolutions import apply_resolutions, fingerprint_conflict, validate_resolution_config
from .tables import NORMALIZER_VERSION, issue, normalize_table, text


class NormalizeArguments(TypedDict):
    table_id: str
    source: dict
    parser_fingerprint: dict
    raw_table_sha256: str
    product_family: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in values), encoding="utf-8")


def artifact(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f"Invalid or unavailable source artifact: {name}")
    return target


def page_bbox(native: dict | None, height: float) -> list[float] | None:
    if not native or not all(key in native for key in ("l", "t", "r", "b")):
        return None
    if native.get("coord_origin") == "BOTTOMLEFT":
        return [native["l"], height - native["t"], native["r"], height - native["b"]]
    return [native["l"], native["t"], native["r"], native["b"]]


def inside(box: list[float] | None, region: list[float]) -> bool:
    if not box:
        return False
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def narrative_chunks(root: Path, page: dict, doc_id: str) -> list[dict]:
    native = read_json(artifact(root, page["native_json_file"]))
    blocks = native.get("texts", [])
    output, group = [], []
    section = None
    regions = [table["bbox"] for table in page["tables"] if table.get("bbox")]
    height = page["page_size_points"][1]
    def flush() -> None:
        if not group:
            return
        ordinal = len(output)
        output.append({"id": f"{doc_id}:p{page['page']}:n{ordinal}", "doc_id": doc_id,
                       "page": page["page"], "section": section, "text": "\n\n".join(b["text"] for b in group),
                       "kind": "figure_caption" if all(b["label"] == "caption" for b in group) else "narrative",
                       "evidence_status": "source_text", "source_sha256": page["source_sha256"],
                       "citation": page["citation"], "blocks": list(group)})
        group.clear()
    for block in blocks:
        value = text(block.get("text"))
        label = block.get("label", "text")
        prov = block.get("prov", [])
        box = page_bbox(prov[0].get("bbox"), height) if prov else None
        if not value or label in {"page_footer", "page_header"} or any(inside(box, region) for region in regions):
            continue
        if label in {"title", "section_header"}:
            flush(); section = value
        if group and sum(len(b["text"]) for b in group) + len(value) > 1400:
            flush()
        group.append({"text": value, "label": label, "bbox": box, "native_ref": block.get("self_ref")})
    flush()
    if not blocks:
        raw = artifact(root, page["text_file"]).read_text(encoding="utf-8")
        raw = re.sub(r"<table\b.*?</table>", "", raw, flags=re.S | re.I)
        raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("|") and not re.search(r"!\[.*?\]\(", line))
        if raw.strip():
            output.append({"id": f"{doc_id}:p{page['page']}:n0", "doc_id": doc_id, "page": page["page"],
                           "section": None, "text": raw.strip(), "kind": "narrative", "evidence_status": "source_text",
                           "source_sha256": page["source_sha256"], "citation": page["citation"], "blocks": []})
    return output


def parameter_key(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", value.lower())
    token = token.replace("heightoffilter", "height").replace("filterheight", "height").replace("clothswidth", "clothwidth")
    token = token.replace("clothlengthlengthstobedefinedcasebycase", "clothlength")
    return token


def fact_key(fact: dict) -> tuple:
    return (fact.get("configuration"), parameter_key(fact["parameter"]), fact.get("unit"),
            fact.get("quantity_kind"), fact.get("value_kind"), fact.get("operator"))


def reported_value(fact: dict) -> tuple:
    return (fact["status"], fact.get("value"), fact.get("value_range"), fact.get("value_components"), fact.get("model_options"))


def comparable_fields(candidate: dict, alternate: dict, *, independent_geometry: bool) -> dict:
    other = defaultdict(list)
    for fact in alternate["facts"]:
        if fact["status"] not in {"ambiguous", "invalid"}:
            other[fact_key(fact)].append(fact)
    checked, same, conflicts, matched_ids = 0, 0, [], []
    for fact in candidate["facts"]:
        matches = other.get(fact_key(fact), [])
        if len(matches) != 1 or fact["status"] in {"ambiguous", "invalid"}:
            continue
        checked += 1
        if reported_value(fact) == reported_value(matches[0]):
            same += 1
            matched_ids.append(fact["fact_id"])
        else:
            first, second = copy.deepcopy(fact), copy.deepcopy(matches[0])
            # Earlier comparison warnings are not part of a numeric snapshot.
            first["issues"] = [i for i in first["issues"] if not i.get("conflict_fingerprint")]
            conflict_sha = fingerprint_conflict(candidate["source"], first, second)
            conflicts.append({"fact_id": fact["fact_id"], "configuration": fact["configuration"],
                              "parameter": fact["parameter"], "candidate": first, "alternate": second,
                              "identity": list(fact_key(first)), "alternate_identity": list(fact_key(second)),
                              "conflict_fingerprint": conflict_sha})
            finding = issue("source_geometry_disagreement" if independent_geometry else "parser_disagreement",
                            "Independent extraction disagrees; consult the original source page before accepting this field.")
            finding["conflict_fingerprint"] = conflict_sha
            fact["issues"].append(finding)
    return {"comparable_fields": checked, "matching_fields": same, "matching_fact_ids": matched_ids, "conflicts": conflicts}


def source_anomalies(table: dict) -> None:
    grouped = defaultdict(dict)
    for fact in table["facts"]:
        if fact["value_kind"] == "scalar" and isinstance(fact.get("value"), (int, float)) and fact["status"] == "reported":
            grouped[(fact["configuration"], fact["parameter"])][fact["unit"]] = fact
    for units in grouped.values():
        for metric, imperial, factor in (("kg", "lb", 2.2046226218), ("mm", "in", 1 / 25.4)):
            if metric not in units or imperial not in units:
                continue
            a, b = units[metric], units[imperial]
            expected = a["value"] * factor
            if expected and abs(b["value"] - expected) / abs(expected) > .05:
                warning = issue("inconsistent_printed_unit_pair", "Printed metric/imperial values differ by more than 5%; both source values are retained and require caution.", "warning")
                a["issues"].append(warning); b["issues"].append(warning)


def normalize_candidate(page: dict, raw_table: dict, root: Path, geometry: list[dict], fingerprint: dict, doc_id: str) -> dict:
    native = read_json(artifact(root, raw_table["cells_file"]))
    source = {"file": page["source_file"], "sha": page["source_sha256"], "page": page["page"]}
    kwargs = NormalizeArguments(table_id=raw_table["id"], source=source, parser_fingerprint=fingerprint,
                                raw_table_sha256=sha(artifact(root, raw_table["html_file"])), product_family=doc_id)
    primary_error = None
    try:
        primary = normalize_table(cells=native["cells"], rows=native["row_count"], columns=native["column_count"], **kwargs)
    except ValueError as exc:
        primary_error = str(exc); primary = None
    match = match_geometry_table(raw_table["bbox"], geometry) if raw_table.get("bbox") else None
    geometry_candidate = None
    if match:
        try:
            geometry_candidate = normalize_table(cells=match["cells"], rows=match["row_count"], columns=match["column_count"], **kwargs)
        except ValueError:
            geometry_candidate = None
    choose_geometry = (geometry_candidate is not None and geometry_candidate["facts"]
                       and geometry_candidate["normalized_table"]["orientation"] in {"models_in_columns", "model_options_by_frame"}
                       and not any(i["severity"] == "blocker" for i in geometry_candidate["issues"]))
    candidate = geometry_candidate if choose_geometry else primary
    if candidate is None:
        raise ValueError(f"Unusable table topology: {primary_error}")
    candidate["doc_id"] = doc_id
    candidate["normalizer_version"] = NORMALIZER_VERSION + ":" + source_code_hash(Path(__file__).with_name("tables.py"))
    candidate["normalized_table"]["source_geometry"] = {"available": bool(match),
        "selected": choose_geometry, "geometry_table_id": match.get("id") if match else None,
        "match": match.get("match") if match else None,
        "geometry_content_sha256": content_hash(match) if match else None}
    candidate["raw_evidence"] = {"parser": page["parser"], "html_file": raw_table["html_file"],
                                 "cells_file": raw_table["cells_file"], "source_render_file": page["source_render_file"],
                                 "table_bbox": raw_table.get("bbox")}
    if primary_error:
        candidate["issues"].append(issue("raw_parser_topology_error", primary_error + "; selected evidence is a separate source-derived grid.", "warning"))
    if choose_geometry and primary:
        candidate["normalized_table"]["primary_source_comparison"] = comparable_fields(candidate, primary, independent_geometry=True)
    elif geometry_candidate:
        candidate["normalized_table"]["geometry_crosscheck"] = comparable_fields(candidate, geometry_candidate, independent_geometry=True)
    return candidate


def prepare(data_dir: Path, raw_dir: Path, output: Path, services_dir: Path, preferred: str, ledger_path: Path | None, resolution_path: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepared output must be empty or new; earlier evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    source_files = sorted((data_dir / "brochures").glob("*.pdf"))
    if not source_files:
        raise FileNotFoundError("Original brochure PDFs are required")
    inputs, pages_by_parser = {}, {}
    for parser in ("docling", "mineru"):
        root = raw_dir / parser
        if not (root / "manifest.json").exists():
            continue
        manifest = read_json(root / "manifest.json")
        if manifest["status"] != "success":
            raise ValueError(f"{parser} extraction is not complete")
        page_map = {}
        for document in manifest["corpus"]:
            source = data_dir / "brochures" / document["source_file"]
            if not source.is_file() or sha(source) != document["source_sha256"]:
                raise ValueError("Raw extraction and original PDF hashes differ")
            for number in document["selected_pages"]:
                folder = root / source.stem / f"page-{number:03d}"
                validate_existing_page(folder, root, document["source_sha256"], number, manifest["configuration_sha256"])
                page_map[(source.stem, number)] = read_json(folder / "page.json")
        inputs[parser] = manifest; pages_by_parser[parser] = page_map
    if preferred not in inputs or "docling" not in inputs:
        raise ValueError("The preferred parser and Docling narrative extraction must be available")
    ledger = read_json(ledger_path) if ledger_path else {"schema_version": 1, "decisions": []}
    validate_manifest(ledger)
    resolutions = read_json(resolution_path) if resolution_path else {"schema_version": 1, "rules": []}
    validate_resolution_config(resolutions)
    documents, pages, chunks, tables, figures, geometry_outputs = [], [], [], [], [], {}
    parser_fingerprint = {"primary": preferred, "raw_configuration_sha256": {p: m["configuration_sha256"] for p, m in inputs.items()},
                          "geometry_engine_sha256": source_code_hash(Path(__file__).with_name("geometry.py")),
                          "resolution_engine_sha256": source_code_hash(Path(__file__).with_name("resolutions.py")),
                          "preparation_engine_sha256": source_code_hash(Path(__file__)), "normalizer_sha256": source_code_hash(Path(__file__).with_name("tables.py"))}
    for source_file in source_files:
        doc_id = source_file.stem
        geometry_pages = extract_geometry_pages(source_file)
        geometry_outputs[doc_id] = geometry_pages
        doc_pages = sorted(p for d, p in pages_by_parser[preferred] if d == doc_id)
        if doc_pages != list(range(1, len(geometry_pages) + 1)):
            raise ValueError(f"Not every physical page is extracted for {source_file.name}")
        document = {"id": doc_id, "title": doc_id.replace("-", " "), "source_file": source_file.name,
                    "source_sha256": sha(source_file), "page_count": len(doc_pages), "edition_identity": "separate source file and hash; no automatic edition merge"}
        documents.append(document)
        for number in doc_pages:
            primary_page = pages_by_parser[preferred][doc_id, number]
            narrative_page = pages_by_parser["docling"][doc_id, number]
            page_id = f"{doc_id}:p{number}"
            pages.append({"id": page_id, "doc_id": doc_id, "page": number, "source_sha256": document["source_sha256"],
                          "citation": primary_page["citation"], "raw_artifacts": {p: {"source_render_file": meta["source_render_file"],
                          "text_file": meta["text_file"], "native_json_file": meta["native_json_file"]}
                          for p, mapping in pages_by_parser.items() if (meta := mapping.get((doc_id, number)))}})
            chunks.extend(narrative_chunks(raw_dir / "docling", narrative_page, doc_id))
            for p, mapping in pages_by_parser.items():
                meta = mapping.get((doc_id, number))
                if not meta:
                    continue
                for figure in meta["figures"]:
                    figures.append({"id": f"{page_id}:{p}:{figure['id']}", "doc_id": doc_id, "page": number,
                                    "source_sha256": document["source_sha256"], "citation": meta["citation"], "parser": p, **figure})
            for raw_table in primary_page["tables"]:
                alternate_parser = "mineru" if preferred == "docling" else "docling"
                alternate_page = pages_by_parser.get(alternate_parser, {}).get((doc_id, number))
                alternate_table = None
                if alternate_page and raw_table.get("bbox"):
                    matches = []
                    for other in alternate_page["tables"]:
                        if other.get("bbox"):
                            a, b = raw_table["bbox"], other["bbox"]
                            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                            if intersection:
                                matches.append((intersection / ((a[2] - a[0]) * (a[3] - a[1])), other))
                    if matches:
                        alternate_table = max(matches, key=lambda item: item[0])[1]
                try:
                    candidate = normalize_candidate(primary_page, raw_table, raw_dir / preferred,
                                                    geometry_pages[number], parser_fingerprint, doc_id)
                except ValueError as exc:
                    if alternate_table:
                        candidate = normalize_candidate(alternate_page, alternate_table, raw_dir / alternate_parser,
                                                        geometry_pages[number], parser_fingerprint, doc_id)
                        candidate["table_id"] = raw_table["id"]
                        candidate["issues"].append(issue("alternate_parser_fallback", f"Preferred raw grid was unusable ({exc}); alternate raw evidence selected.", "warning"))
                    else:
                        native = read_json(artifact(raw_dir / preferred, raw_table["cells_file"]))
                        candidate = {"table_id": raw_table["id"], "source": {"file": primary_page["source_file"], "sha": primary_page["source_sha256"], "page": number},
                            "doc_id": doc_id, "product_family": doc_id, "parser_fingerprint": parser_fingerprint,
                            "normalizer_version": NORMALIZER_VERSION + ":" + source_code_hash(Path(__file__).with_name("tables.py")),
                            "raw_table_sha256": sha(artifact(raw_dir / preferred, raw_table["html_file"])), "facts": [],
                            "issues": [issue("unusable_table_topology", str(exc) + "; raw evidence retained without assigned facts.")],
                            "normalized_table": {"row_count": native["row_count"], "column_count": native["column_count"], "cells": native["cells"], "orientation": "retained_for_review",
                                "source_geometry": {"available": False, "selected": False}},
                            "raw_evidence": {"parser": preferred, "html_file": raw_table["html_file"], "cells_file": raw_table["cells_file"], "source_render_file": primary_page["source_render_file"], "table_bbox": raw_table.get("bbox")}}
                if alternate_table and candidate["raw_evidence"]["parser"] == preferred:
                    try:
                        alternate_cells = read_json(artifact(raw_dir / alternate_parser, alternate_table["cells_file"]))
                        alternate = normalize_table(raw_table["id"], alternate_cells["cells"], alternate_cells["row_count"], alternate_cells["column_count"], candidate["source"], parser_fingerprint,
                                                    sha(artifact(raw_dir / alternate_parser, alternate_table["html_file"])), product_family=doc_id)
                        candidate["normalized_table"]["alternate_parser_comparison"] = comparable_fields(candidate, alternate, independent_geometry=bool(candidate["normalized_table"]["source_geometry"]["selected"]))
                    except ValueError as exc:
                        candidate["issues"].append(issue("alternate_parser_topology_error", str(exc), "warning"))
                candidate = apply_resolutions(candidate, resolutions)
                source_anomalies(candidate)
                candidate["table_fingerprint"] = fingerprint_candidate(candidate)
                reviewed = apply_reviews(candidate, ledger)
                tables.append(reviewed)
                native = read_json(artifact(raw_dir / candidate["raw_evidence"]["parser"], candidate["raw_evidence"]["cells_file"]))
                context = "\n".join(text(cell["text"]) for cell in native["cells"] if text(cell["text"]))
                chunks.append({"id": f"{page_id}:{raw_table['id']}:raw", "doc_id": doc_id, "page": number,
                               "text": context, "kind": "table_context", "evidence_status": "raw_table",
                               "source_sha256": document["source_sha256"], "citation": primary_page["citation"]})
    source_reports = sorted((data_dir / "HistoricalServices").glob("*.txt"))
    service_manifest = read_json(services_dir / "manifest.json")
    if sha(services_dir / "records.jsonl") != service_manifest["records_sha256"]:
        raise ValueError("Parsed service artifact hash differs from its manifest")
    services = [json.loads(line, parse_float=Decimal) for line in (services_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(services) != len(source_reports):
        raise ValueError("Service corpus count differs from original inputs")
    service_sources = {p.name: sha(p) for p in source_reports}
    if {Path(s["path"]).name: s["sha256"] for s in service_manifest["inputs"]} != service_sources:
        raise ValueError("Service manifest and original source corpus differ")
    if len({Path(r["source"]["path"]).name for r in services}) != len(service_sources):
        raise ValueError("Duplicate or missing parsed service source")
    for record in services:
        source = record["source"]
        if service_sources.get(Path(source["path"]).name) != source["sha256"]:
            raise ValueError("Parsed service and original report hashes differ")
    from .store import build_store, validate_store
    store_counts = build_store(output / "knowledge.sqlite", documents, chunks, tables, services)
    validation = validate_store(output / "knowledge.sqlite")
    if not validation["valid"]:
        raise ValueError(f"Knowledge store failed integrity checks: {validation}")
    write_json(output / "documents.json", documents)
    write_jsonl(output / "pages.jsonl", pages)
    write_jsonl(output / "chunks.jsonl", chunks)
    write_jsonl(output / "tables.jsonl", tables)
    write_jsonl(output / "figures.jsonl", figures)
    write_json(output / "source-geometry.json", geometry_outputs)
    counts = Counter(f["review_status"] for table in tables for f in table["facts"])
    fact_status = Counter(f["status"] for table in tables for f in table["facts"])
    findings = Counter(i["code"] for table in tables for i in table["issues"])
    findings.update(i["code"] for table in tables for f in table["facts"] for i in f["issues"])
    report = {"schema_version": 1, "status": "prepared", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_documents": len(documents), "physical_pages": len(pages), "tables": len(tables), "figure_crops": len(figures),
              "raw_tables_by_parser": {p: sum(len(page["tables"]) for page in mapping.values()) for p, mapping in pages_by_parser.items()},
              "full_page_visual_fallbacks": len(pages), "narrative_chunks": sum(c["kind"] != "table_context" for c in chunks),
              "table_context_chunks": sum(c["kind"] == "table_context" for c in chunks), "candidate_facts": sum(len(t["facts"]) for t in tables),
              "fact_review_status": dict(counts), "fact_source_status": dict(fact_status), "issue_counts": dict(findings),
              "service_records": len(services), "store_counts": store_counts, "store_validation": validation,
              "parser_fingerprint": parser_fingerprint, "raw_input_root": str(raw_dir.resolve()),
              "source_data_root": str(data_dir.resolve()), "services_input_root": str(services_dir.resolve()),
              "review_ledger": str(ledger_path.resolve()) if ledger_path else None,
              "normalization_resolutions": str(resolution_path.resolve()) if resolution_path else None,
              "no_agent_or_estimation_executed": True, "paid_api_used": False,
              "source_files": [{"file": d["source_file"], "sha256": d["source_sha256"]} for d in documents]}
    write_json(output / "manifest.json", report)
    write_json(output / "quality-report.json", {k: v for k, v in report.items() if k not in {"source_files", "parser_fingerprint"}})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=Path(".pilot-cache/assistant/raw"))
    parser.add_argument("--services-dir", type=Path, default=Path(".pilot-cache/assistant/services"))
    parser.add_argument("--output", type=Path, default=Path(".pilot-cache/assistant/prepared"))
    parser.add_argument("--preferred-parser", choices=("docling", "mineru"), default="docling")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--resolutions", type=Path)
    args = parser.parse_args(argv)
    result = prepare(args.data_dir, args.raw_dir, args.output, args.services_dir, args.preferred_parser, args.reviews, args.resolutions)
    print(json.dumps({k: result[k] for k in ("status", "source_documents", "physical_pages", "tables", "candidate_facts", "fact_review_status", "service_records")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
