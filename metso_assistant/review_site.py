"""Generate a private, read-only HTML hub for prepared brochure evidence.

Example:
    python -m metso_assistant.review_site --prepared-dir .pilot-cache/assistant/prepared --raw-dir .pilot-cache/assistant/raw

Native table HTML is linked, never inserted into this page. Candidate text is
escaped and artifact paths must resolve inside the selected local parser root.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / ".pilot-cache"
DEFAULT_RAW = PRIVATE_ROOT / "assistant" / "raw"
DEFAULT_OUTPUT = PRIVATE_ROOT / "assistant" / "review" / "index.html"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def escaped(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_records(directory: Path, stem: str, *, optional: bool = False) -> list[dict[str, Any]]:
    json_file, lines_file = directory / f"{stem}.json", directory / f"{stem}.jsonl"
    if json_file.is_file():
        records = json.loads(json_file.read_text(encoding="utf-8"))
    elif lines_file.is_file():
        records = [json.loads(line) for line in lines_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif optional:
        return []
    else:
        raise FileNotFoundError(f"Missing {stem}.json or {stem}.jsonl in {directory}")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"{stem} must contain a list of JSON objects.")
    return records


def safe_artifact_path(root: Path, reference: Any, *, extensions: set[str] | None = None) -> Path:
    """Accept filesystem-relative references, never URLs, drives or escapes."""
    if not isinstance(reference, str) or not reference or "\x00" in reference:
        raise ValueError("Artifact reference must be a nonempty relative path.")
    if ":" in reference or PureWindowsPath(reference).drive or reference.startswith(("/", "\\")):
        raise ValueError("Artifact URLs and absolute/drive-relative paths are not allowed.")
    parts = reference.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Artifact path contains an empty or traversal component.")
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("Artifact is unavailable or escapes the selected evidence root.")
    if extensions is not None and target.suffix.lower() not in extensions:
        raise ValueError("Artifact has an unexpected file type.")
    return target


def local_url(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output.parent.resolve()).replace(os.sep, "/"), safe="/")


def valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) for item in value):
        return None
    if value[2] < value[0] or value[3] < value[1]:
        return None
    return list(value)


def identity(record: dict[str, Any]) -> str:
    source = record.get("source", {})
    value = [source.get("file"), source.get("page"), record.get("table_id"), record.get("table_fingerprint")]
    return "table-" + hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def status_badge(value: Any) -> str:
    value = str(value or "pending")
    style = value if value in {"verified", "pending", "ambiguous", "rejected", "invalid", "missing", "reported"} else "pending"
    label = "QA approved" if value == "verified" else value
    return f'<span class="badge {style}">{escaped(label)}</span>'


def issue_markup(items: Any) -> str:
    if not items:
        return '<span class="muted">—</span>'
    if not isinstance(items, list):
        items = [items]
    output = []
    for item in items:
        if isinstance(item, dict):
            severity = item.get("severity", "note")
            code = item.get("code", item.get("type", "note"))
            message = item.get("message", "")
            output.append(f'<li><strong>{escaped(severity)} · {escaped(code)}</strong><br>{escaped(message)}</li>')
        else:
            output.append(f"<li>{escaped(item)}</li>")
    return '<ul class="issues">' + "".join(output) + "</ul>"


def displayed_value(fact: dict[str, Any]) -> str:
    selected = {key: fact[key] for key in ("value", "value_range", "value_components", "model_options", "operator", "quantity_kind") if fact.get(key) is not None}
    return escaped(json.dumps(selected, ensure_ascii=False, sort_keys=True)) if selected else "—"


class Evidence:
    def __init__(self, raw: Path, output: Path) -> None:
        self.raw, self.output = raw.resolve(), output.resolve()
        self.diagnostics: list[str] = []
        self._pages: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.configurations: dict[str, str] = {}
        for parser in ("docling", "mineru"):
            manifest = self.raw / parser / "manifest.json"
            if manifest.is_file():
                self.configurations[parser] = json.loads(manifest.read_text(encoding="utf-8")).get("configuration_sha256", "")

    def resolve(self, parser: Any, reference: Any, extensions: set[str] | None = None) -> Path | None:
        if parser not in {"docling", "mineru"}:
            self.diagnostics.append("Unknown evidence parser; link omitted.")
            return None
        try:
            if not (self.raw / parser).resolve().is_relative_to(self.raw):
                raise ValueError("Parser root escapes the selected evidence root.")
            return safe_artifact_path(self.raw / parser, reference, extensions=extensions)
        except ValueError as exc:
            self.diagnostics.append(f"{parser}: {exc}")
            return None

    def link(self, parser: Any, reference: Any, label: str, extensions: set[str] | None = None) -> str:
        path = self.resolve(parser, reference, extensions)
        if path is None:
            return f'<span class="unavailable">{escaped(label)} unavailable</span>'
        return f'<a href="{escaped(local_url(path, self.output))}" target="_blank" rel="noopener noreferrer">{escaped(label)}</a>'

    def page(self, parser: str, doc_id: str, number: int) -> dict[str, Any]:
        key = (parser, doc_id, number)
        if key not in self._pages:
            if not (self.raw / parser).is_dir():
                self._pages[key] = {}
                return self._pages[key]
            try:
                if not (self.raw / parser).resolve().is_relative_to(self.raw):
                    raise ValueError("Parser root escapes the selected evidence root.")
                path = safe_artifact_path(self.raw / parser, f"{doc_id}/page-{number:03d}/page.json", extensions={".json"})
                self._pages[key] = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.diagnostics.append(f"Raw page metadata unavailable: {parser}/{doc_id} page {number}: {exc}")
                self._pages[key] = {}
        return self._pages[key]


def source_identity(table: dict[str, Any]) -> tuple[str, int, str]:
    source = table.get("source", {})
    filename = source.get("file", table.get("source_file", "Unknown source"))
    doc_id = table.get("doc_id") or Path(str(filename)).stem
    return str(doc_id), int(source.get("page", table.get("page", 0))), str(filename)


def facts_markup(table: dict[str, Any], section_id: str) -> str:
    rows = []
    for fact in table.get("facts", []):
        refs = fact.get("source_refs", [])
        box = next((valid_bbox(ref.get("cell_bbox")) for ref in refs if isinstance(ref, dict) and valid_bbox(ref.get("cell_bbox")) is not None), None)
        locate = ""
        if box is not None:
            locate = f'<button type="button" class="locate" data-section="{section_id}" data-bbox="{escaped(json.dumps(box, allow_nan=False))}">Locate cell</button>'
        detail = escaped(json.dumps(fact, ensure_ascii=False, sort_keys=True, indent=2))
        flags = [*fact.get("issues", []), *fact.get("review_blockers", [])]
        rows.append(
            '<tr><td>' + escaped(fact.get("configuration") or "Shared / unspecified") + '</td>'
            '<td>' + escaped(fact.get("parameter")) + '</td>'
            '<td class="raw-value">' + escaped(fact.get("raw_cell_text")) + '</td>'
            '<td>' + escaped(fact.get("raw_unit") or "—") + '<br><span class="muted">Normalized: ' + escaped(fact.get("unit") or "—") + '</span></td>'
            '<td class="value">' + displayed_value(fact) + '<br><span class="muted">' + escaped(fact.get("value_kind")) + '</span></td>'
            '<td>' + status_badge(fact.get("review_status")) + '<br>' + status_badge(fact.get("status")) + '</td>'
            '<td>' + issue_markup(flags) + '</td>'
            '<td>' + locate + '<details><summary>Provenance</summary><pre>' + detail + '</pre></details></td></tr>'
        )
    return '<div class="facts-scroll"><table class="facts"><thead><tr><th>Configuration</th><th>Parameter</th><th>Printed cell</th><th>Unit</th><th>Candidate value</th><th>Review / source</th><th>Flags</th><th>Source locator</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'


def figure_links(items: list[dict[str, Any]], evidence: Evidence) -> str:
    rows = []
    for figure in items:
        parser = figure.get("parser", "docling")
        label = f"{parser} · {figure.get('id', 'figure')}"
        image = evidence.link(parser, figure.get("image_file"), label, IMAGE_EXTENSIONS)
        native = evidence.link(parser, figure.get("native_json_file"), "native figure JSON", {".json"}) if figure.get("native_json_file") else ""
        rows.append(f'<li>{image} {native}<br><span class="muted">bbox {escaped(figure.get("bbox"))} · {escaped(figure.get("coordinate_system"))}</span></li>')
    return '<ul class="figure-links">' + "".join(rows) + '</ul>' if rows else '<p class="muted">No native figure crops on this page; use the full-page source.</p>'


def table_markup(table: dict[str, Any], evidence: Evidence, pages: dict[tuple[str, int], dict[str, Any]], figures: dict[tuple[str, int], list[dict[str, Any]]], documents: dict[str, dict[str, Any]]) -> str:
    section_id = identity(table)
    doc_id, number, filename = source_identity(table)
    source = table.get("source", {})
    page = pages.get((doc_id, number), {})
    primary = table.get("raw_evidence", {}).get("parser") or table.get("parser_fingerprint", {}).get("primary", "docling")
    raw_page = evidence.page(primary, doc_id, number)
    raw_ref = table.get("raw_evidence", {})
    image_ref = raw_ref.get("source_render_file") or page.get("raw_artifacts", {}).get(primary, {}).get("source_render_file") or raw_page.get("source_render_file")
    source_image = evidence.resolve(primary, image_ref, IMAGE_EXTENSIONS)
    title = documents.get(doc_id, {}).get("title", filename)
    issues = list(table.get("issues", []))
    notices = []
    if raw_page and (raw_page.get("source_sha256") != source.get("sha", source.get("source_sha256")) or raw_page.get("page") != number):
        notices.append("Source identity differs between prepared candidate and selected raw page. Resolve provenance before recording a decision.")
        source_image = None
    expected = table.get("parser_fingerprint", {}).get("raw_configuration_sha256", {})
    changed = [parser for parser, digest in expected.items() if evidence.configurations.get(parser) and evidence.configurations[parser] != digest]
    if changed:
        notices.append("Prepared/raw configuration fingerprints differ for " + ", ".join(changed) + ". Regenerate prepared evidence before recording decisions; this hub grants no approvals.")
    if raw_ref.get("html_file") and table.get("raw_table_sha256"):
        native_html = evidence.resolve(primary, raw_ref["html_file"], {".html", ".htm"})
        if native_html is not None and hashlib.sha256(native_html.read_bytes()).hexdigest() != table["raw_table_sha256"]:
            notices.append("Native table HTML hash differs from the prepared candidate. Resolve the evidence mismatch before recording a decision.")
    links = []
    for parser in ("docling", "mineru"):
        native_page = evidence.page(parser, doc_id, number)
        for item in native_page.get("tables", []):
            links.append(evidence.link(parser, item.get("html_file"), f"{parser} {item.get('id')} · native HTML", {".html", ".htm"}))
        if native_page.get("native_json_file"):
            links.append(evidence.link(parser, native_page["native_json_file"], f"{parser} · native page JSON", {".json"}))
    if raw_ref.get("cells_file"):
        links.append(evidence.link(primary, raw_ref["cells_file"], "Primary raw cell JSON", {".json"}))
    if source_image is not None:
        image_url = escaped(local_url(source_image, evidence.output))
        source_links = f'<a href="{image_url}" target="_blank" rel="noopener noreferrer">Open original full-page render</a>'
        size = raw_page.get("page_size_points", [])
        overlay = ""
        if isinstance(size, list) and len(size) == 2 and all(isinstance(x, (int, float)) and x > 0 and math.isfinite(x) for x in size):
            overlay = f'<svg class="overlay" viewBox="0 0 {size[0]} {size[1]}" preserveAspectRatio="none" aria-hidden="true"><rect class="cell-highlight" x="0" y="0" width="0" height="0"></rect></svg>'
        preview = f'<details class="source-preview"><summary>Original page preview and cell locator</summary><div class="source-frame"><img src="{image_url}" alt="{escaped(filename)} physical page {number}" loading="lazy">{overlay}</div><p class="locator-note muted">Select Locate cell in the candidate rows to highlight its source bbox. Full resolution is available through the source link.</p></details>'
    else:
        source_links, preview = '<span class="unavailable">Full-page source unavailable or identity mismatch</span>', ""
    metadata = {"source": source, "table_fingerprint": table.get("table_fingerprint"), "raw_table_sha256": table.get("raw_table_sha256"), "normalizer_version": table.get("normalizer_version"), "parser_fingerprint": table.get("parser_fingerprint"), "raw_evidence": raw_ref, "review": table.get("review"), "review_diagnostics": table.get("review_diagnostics"), "geometry": table.get("normalized_table", {}).get("source_geometry")}
    diagram_items = figures.get((doc_id, number), [])
    fact_counts = Counter(fact.get("review_status", "pending") for fact in table.get("facts", []))
    notices_html = ''.join(f'<p class="notice">{escaped(message)}</p>' for message in notices)
    return (
        f'<section class="review-table" id="{section_id}"><header><h2>{escaped(title)} · page {number} · {escaped(table.get("table_id"))}</h2>'
        f'<p>{status_badge(table.get("review_status"))} · {len(table.get("facts", []))} candidate facts · {escaped(dict(fact_counts))}</p></header>'
        + notices_html + '<p class="evidence-links">' + source_links + ' · ' + ' · '.join(links) + '</p>'
        + '<div class="table-flags"><strong>Table flags</strong>' + issue_markup(issues) + '</div>'
        + preview + '<details><summary>Fingerprints and provenance</summary><pre>' + escaped(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)) + '</pre></details>'
        + f'<details><summary>{len(diagram_items)} figure crops / visual fallbacks on this page</summary>' + figure_links(diagram_items, evidence) + '</details>'
        + facts_markup(table, section_id) + '</section>'
    )


CSS = """
:root{font-family:Segoe UI,Arial,sans-serif;color:#1a2632;background:#edf1f5;line-height:1.4}*{box-sizing:border-box}body{margin:0}a{color:#075b91}button{font:inherit;cursor:pointer}.top{padding:24px 32px;background:#162f44;color:white}.top h1{margin:0 0 8px}.top p{margin:5px 0}.layout{display:grid;grid-template-columns:270px minmax(0,1fr);gap:22px;padding:22px}nav{position:sticky;top:12px;align-self:start;max-height:95vh;overflow:auto;background:white;border-radius:8px;padding:18px}nav a{display:block;margin:10px 0;font-size:14px}.review-table,.catalogue,.diagnostics{background:white;padding:24px;margin:0 0 24px;border-radius:8px;border:1px solid #d9e0e7}.review-table h2{margin:0 0 8px;font-size:21px}.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:12px;font-weight:650;background:#e8edf1}.verified{background:#daf0df;color:#1c6530}.pending{background:#fff0cf;color:#7d5710}.ambiguous,.invalid,.rejected{background:#fde0de;color:#8b2525}.missing{background:#e9e9ee;color:#51525b}.muted{color:#667686;font-size:12px}.notice,.unavailable{color:#8b2525;background:#fff2e4;padding:8px}.evidence-links{font-size:14px}.evidence-links a{display:inline-block;margin:4px 3px}.issues{margin:6px 0;padding-left:18px;font-size:12px}.issues li{margin:0 0 7px}.table-flags{padding:12px;background:#f6f8fa;margin:16px 0}details{margin:13px 0}summary{cursor:pointer;font-weight:600}pre{overflow:auto;background:#f5f7fa;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.source-frame{position:relative;max-width:1100px;margin-top:12px;border:1px solid #c5cfd8}.source-frame img{display:block;width:100%;height:auto}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.cell-highlight{fill:#ffb90038;stroke:#e75f00;stroke-width:1.5;vector-effect:non-scaling-stroke}.facts-scroll{overflow:auto;margin-top:18px;max-height:760px;border:1px solid #dae1e8}.facts{border-collapse:collapse;width:100%;font-size:13px;min-width:1150px}.facts th,.facts td{padding:10px;border-bottom:1px solid #dae1e8;text-align:left;vertical-align:top}.facts th{position:sticky;top:0;background:#eaf0f5;z-index:1}.facts tbody tr:nth-child(even){background:#fafbfd}.facts tr.selected{background:#fff1cf}.facts .raw-value{white-space:pre-wrap;min-width:90px}.facts .value{font-family:Consolas,monospace;font-size:12px;max-width:230px;overflow-wrap:anywhere}.locate{border:1px solid #9ab3c8;background:#f2f7fb;border-radius:4px;padding:5px 8px;white-space:nowrap}.figure-links{columns:2;padding-left:20px}.figure-links li{break-inside:avoid;margin:0 0 10px}.page-list{width:100%;border-collapse:collapse;font-size:14px}.page-list td,.page-list th{border-bottom:1px solid #dfe5eb;padding:9px;text-align:left;vertical-align:top}@media(max-width:950px){.layout{display:block;padding:10px}nav{position:static;max-height:260px;margin-bottom:16px}.review-table{padding:14px}.figure-links{columns:1}}
"""
SCRIPT = """
document.querySelectorAll('.locate').forEach(button=>button.addEventListener('click',()=>{
 const section=document.getElementById(button.dataset.section);if(!section)return;
 const preview=section.querySelector('.source-preview'),rect=section.querySelector('.cell-highlight');
 if(!preview||!rect)return;let box;try{box=JSON.parse(button.dataset.bbox)}catch{return}
 if(!Array.isArray(box)||box.length!==4||!box.every(Number.isFinite))return;
 preview.open=true;['x','y','width','height'].forEach((key,index)=>rect.setAttribute(key,[box[0],box[1],box[2]-box[0],box[3]-box[1]][index]));
 section.querySelectorAll('tr.selected').forEach(row=>row.classList.remove('selected'));button.closest('tr').classList.add('selected');
 section.querySelector('.locator-note').textContent='Highlighted source bbox in PDF points: '+box.join(', ');
 preview.scrollIntoView({behavior:'smooth',block:'start'});
}));
"""


def render_review_html(prepared_dir: Path, raw_dir: Path, output: Path) -> tuple[str, dict[str, Any]]:
    tables = read_records(prepared_dir, "tables")
    page_records = read_records(prepared_dir, "pages")
    document_records = read_records(prepared_dir, "documents")
    figure_records = read_records(prepared_dir, "figures", optional=True)
    pages = {(str(page["doc_id"]), int(page["page"])): page for page in page_records}
    documents = {str(document["id"]): document for document in document_records}
    figures: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for figure in figure_records:
        figures[(str(figure["doc_id"]), int(figure["page"]))].append(figure)
    evidence = Evidence(raw_dir, output)
    sections = [table_markup(table, evidence, pages, figures, documents) for table in tables]
    navigation = ''.join(f'<a href="#{identity(table)}">{escaped(source_identity(table)[2])}<br>Page {source_identity(table)[1]} · {escaped(table.get("table_id"))} · {status_badge(table.get("review_status"))}</a>' for table in tables)
    catalogue = []
    for page in sorted(page_records, key=lambda p: (str(p["doc_id"]), int(p["page"]))):
        key = (str(page["doc_id"]), int(page["page"]))
        links = []
        for parser, artifacts in page.get("raw_artifacts", {}).items():
            if artifacts.get("source_render_file"):
                links.append(evidence.link(parser, artifacts["source_render_file"], f"{parser} full page", IMAGE_EXTENSIONS))
        crops = figures.get(key, [])
        details = f'<details><summary>{len(crops)} crops</summary>{figure_links(crops,evidence)}</details>' if crops else "0 crops"
        catalogue.append('<tr><td>' + escaped(page.get("citation", f"{key[0]}, page {key[1]}")) + '</td><td>' + ' · '.join(links) + '</td><td>' + details + '</td></tr>')
    facts = [fact for table in tables for fact in table.get("facts", [])]
    states = dict(Counter(fact.get("review_status", "pending") for fact in facts))
    diagnostic_list = sorted(set(evidence.diagnostics))
    diagnostics = '<section class="diagnostics"><h2>Evidence link diagnostics</h2>' + issue_markup(diagnostic_list) + '</section>' if diagnostic_list else ""
    generated = datetime.now(timezone.utc).isoformat()
    report = {"schema_version": 1, "operation": "private_review_hub", "generated_at": generated, "prepared_dir": str(prepared_dir.resolve()), "raw_dir": str(raw_dir.resolve()), "output": str(output.resolve()), "tables": len(tables), "candidate_facts": len(facts), "review_status_counts": states, "pages": len(page_records), "figure_crops": len(figure_records), "link_diagnostics": diagnostic_list, "approvals_written": 0}
    display_states = {"QA approved" if key == "verified" else key: count for key, count in states.items()}
    header = f'<header class="top"><h1>Metso brochure source review</h1><p>{len(tables)} tables · {len(facts)} candidate facts · {len(page_records)} source pages · {len(figure_records)} figure crops</p><p>Review states: {escaped(display_states)}</p><p>Read-only snapshot. QA approved marks the recorded review decision. Decision provenance describes the comparisons and source inspections performed; source warnings remain attached.</p><p class="muted">Generated {escaped(generated)} · Active evidence: {escaped(raw_dir.resolve())}</p></header>'
    page_html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Private brochure source review</title><style>' + CSS + '</style></head><body>' + header + '<div class="layout"><nav aria-label="Tables"><strong>Tables</strong>' + navigation + '<a href="#page-catalogue">All pages and figure fallbacks</a></nav><main>' + ''.join(sections) + '<section class="catalogue" id="page-catalogue"><h2>Source pages and figure fallbacks</h2><p>Full-page renders preserve content that native crops may omit. Figure counts do not establish diagram relationships.</p><table class="page-list"><thead><tr><th>Physical source page</th><th>Full-page evidence</th><th>Figure crops</th></tr></thead><tbody>' + ''.join(catalogue) + '</tbody></table></section>' + diagnostics + '</main></div><script>' + SCRIPT + '</script></body></html>\n'
    return page_html, report


def build_review_site(prepared_dir: Path, raw_dir: Path, output: Path, *, private_root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_relative_to(private_root.resolve()):
        raise ValueError("Source-derived review output must remain inside the private .pilot-cache root.")
    if output.suffix.lower() not in {".html", ".htm"}:
        output = output / "index.html"
    content, report = render_review_html(prepared_dir.resolve(), raw_dir.resolve(), output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, output)
    sidecar = output.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW, help="Parent containing active docling/ and mineru/ extraction roots.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Private HTML file or directory (index.html is added for a directory).")
    args = parser.parse_args(argv)
    try:
        report = build_review_site(args.prepared_dir, args.raw_dir, args.output)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Review hub generation failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
