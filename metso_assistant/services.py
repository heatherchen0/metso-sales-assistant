"""Deterministic, evidence-preserving ingestion of field-service text reports.

No prices, currencies, missing charges, or complete visit hours are inferred from
the customer's country. Decimal arithmetic stays exact through JSON emission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PARSER_VERSION = "1.0.0"
NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
CURRENCIES = ("USD", "EUR", "CAD", "AUD", "SEK", "ZAR")
UNIT_ALIASES = {"hr": "hour", "hrs": "hour", "hours": "hour", "h": "hour",
                "litre": "litre", "liter": "litre", "litres": "litre", "l": "litre",
                "pcs": "piece", "pc": "piece", "pieces": "piece", "piece": "piece",
                "ea": "each", "each": "each", "units": "unit", "unit": "unit",
                "pairs": "pair", "pair": "pair", "tubes": "tube", "tube": "tube",
                "bottles": "bottle", "bottle": "bottle", "kits": "kit", "kit": "kit",
                "sets": "set", "set": "set", "lot": "lot", "ml": "ml", "rag": "rag"}
UNIT_PATTERN = r"hours?|hrs?|h|litres?|liters?|L|ml|pcs|pieces?|units?|pairs?|tubes?|bottles?|kits?|sets?|lot|each|ea|pc|rag"
EQUIPMENT_RULES = [
    ("filter_optimizer", r"\bFilter Optimizer\b", "Filter Optimizer"),
    ("cellstation", r"\bCellStation\b", "CellStation"),
    ("larox_pf_ds", r"\bPF[ -]DS\b", "Larox PF-DS"),
    ("larox_pf", r"\bLarox PF\b", "Larox PF"),
    ("sala_vsp", r"\b(?:SALA|VSP[ -]?750)\b", "SALA VSP"),
    ("md_pump", r"\bMD(?:\s+(?:series|mill)|\d{3})\b", "MD series"),
    ("orion", r"\bOrion\b", "Orion"),
    ("concorde_cell", r"\bConcorde Cell", "Concorde Cell"),
    ("columncell", r"\b(?:ColumnCell|SonicSparger)\b", "ColumnCell"),
]
SERVICE_RULES = [
    ("cloth_replacement", r"(?:cloth\w*[^.;\n]{0,35}(?:change|replacement)|(?:replac\w*|changed?)[^.;\n]{0,45}cloth\w*)"),
    ("impeller_replacement", r"(?:impeller\w*[^.;\n]{0,30}(?:swap\w*|replac\w*)|(?:replac\w*|swapp\w*)[^.;\n]{0,35}impeller\w*)"),
    ("software_update", r"(?:updat\w*[^.;\n]{0,40}(?:software|algorithm|CellStation|Filter Optimizer)|(?:software|algorithm)[^.;\n]{0,35}updat\w*)"),
    ("commissioning", r"\b(?:commissioning|commish)\b"),
    ("calibration", r"\b(?:recalibrat\w*|calibrat\w*|cal sensors|cal sensors)\b"),
    ("inspection", r"\b(?:inspect\w*|inspection)\b"),
    ("cleaning", r"\b(?:clean\w*|flushed?|backflush)\b"),
    ("hydraulic_service", r"\b(?:hydraulic|HYD)\b"),
    ("emergency_visit", r"\b(?:emergency|shutdown)\b"),
]


def clean(value: str) -> str:
    return value.replace("**", "").strip().strip("*").strip()


def evidence(line: int, end: int | None = None) -> list[dict[str, int]]:
    return [{"line_start": line, "line_end": end or line}]


def number(value: str) -> Decimal | None:
    """Only unambiguous English decimal/thousands syntax; never locale guesses."""
    value = clean(value).strip()
    if not re.fullmatch(NUMBER, value):
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def json_numbers(value: Any) -> str:
    """Emit Decimal as exact JSON number tokens, including trailing decimals."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite Decimal is not JSON")
        return format(value, "f")
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + json_numbers(v)
                              for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(json_numbers(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def currency(text: str, context: str | None = None) -> tuple[str | None, str | None]:
    codes = set(re.findall(r"\b(?:USD|EUR|CAD|AUD|SEK|ZAR)\b", text, re.I))
    codes = {code.upper() for code in codes}
    if "€" in text:
        codes.add("EUR")
    code = next(iter(codes)) if len(codes) == 1 else (context if not codes else None)
    return code, ("€" if "€" in text else "$" if "$" in text else None)


def money(text: str, context: str | None = None) -> dict[str, Any]:
    text = clean(text)
    code, symbol = currency(text, context)
    # Monetary cells begin with code/symbol or a number. Do not mine PO numbers.
    m = re.match(rf"^(?:(?:{'|'.join(CURRENCIES)})\s*)?[$€]?\s*({NUMBER})", text, re.I)
    return {"amount": number(m.group(1)) if m else None, "currency": code,
            "currency_symbol": symbol}


def quantity(text: str, unit_text: str = "") -> tuple[Decimal | None, str | None]:
    text = clean(text)
    m = re.fullmatch(rf"({NUMBER})\s*({UNIT_PATTERN})?", text, re.I)
    if not m:
        return None, None
    raw_unit = m.group(2) or clean(unit_text)
    return number(m.group(1)), UNIT_ALIASES.get(raw_unit.lower(), raw_unit or None)


def tables(lines: list[str]) -> list[dict[str, Any]]:
    found = []
    i = 0
    while i + 1 < len(lines):
        if lines[i].lstrip().startswith("|") and re.match(r"^\s*\|[\s:|\-]+\|?\s*$", lines[i + 1]):
            header = [clean(x) for x in lines[i].strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = [clean(x) for x in lines[j].strip().strip("|").split("|")]
                rows.append((j + 1, cells))
                j += 1
            found.append({"header": header, "line": i + 1, "rows": rows})
            i = j
        else:
            i += 1
    return found


def column(header: list[str], pattern: str) -> int | None:
    return next((i for i, h in enumerate(header) if re.search(pattern, h, re.I)), None)


def cell(cells: list[str], index: int | None) -> str:
    return cells[index] if index is not None and index < len(cells) else ""


def labelled(lines: list[str], label: str) -> tuple[str | None, list[dict[str, int]]]:
    for i, raw in enumerate(lines):
        m = re.match(rf"^\s*(?:-\s*)?(?:{label})\s*:\s*(.*)$", clean(raw), re.I)
        if not m:
            continue
        if m.group(1).strip():
            return m.group(1).strip(), evidence(i + 1)
        j = i + 1
        while j < len(lines) and not clean(lines[j]):
            j += 1
        if j < len(lines) and not re.match(r"^(?:---|#|\||[\w /]+:)", clean(lines[j])):
            return clean(lines[j]), evidence(i + 1, j + 1)
    return None, []


def normalize_date(raw: str | None) -> dict[str, Any]:
    if raw:
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
        if m:
            try:
                return {"value": date.fromisoformat(m.group(1)).isoformat(), "raw": raw, "status": "reported"}
            except ValueError:
                pass
    return {"value": None, "raw": raw, "status": "ambiguous" if raw else "not_stated"}


def categorize(description: str) -> str:
    if re.search(r"\btravel", description, re.I):
        return "travel"
    if re.search(r"\blabou?r", description, re.I):
        return "labour"
    if re.search(r"consum|cleaning|gloves|grease|lubricant|oil", description, re.I):
        return "consumables"
    return "parts"


def warning(record: dict[str, Any], code: str, message: str, refs: list[dict[str, int]] | None = None) -> None:
    record["warnings"].append({"code": code, "message": message, "evidence": refs or []})


def measure(candidates: list[dict[str, Any]], name: str, record: dict[str, Any]) -> dict[str, Any]:
    candidates = list({(str(c["value"]), str(c["range"]), c["scope"], c["evidence"][0]["line_start"]): c
                       for c in candidates}.values())
    result = {"value": None, "range": None, "status": "not_stated", "approximate": False,
              "scope": None, "evidence": [], "candidates": candidates}
    if not candidates:
        return result
    usable = [c for c in candidates if c["scope"] in {"visit_total", "billed_hours"}]
    if not usable:
        result["status"] = "ambiguous"
        warning(record, f"{name}_scope_incomplete", "Hours are stated for a partial/one-way scope, not a complete visit total.",
                [e for c in candidates for e in c["evidence"]])
        return result
    values = {(c["value"], str(c["range"])) for c in usable}
    if len(values) > 1:
        result["status"] = "ambiguous"
        warning(record, f"{name}_conflicting_hours", "Multiple reported hour values disagree; none is selected.",
                [e for c in usable for e in c["evidence"]])
        return result
    preferred = next((c for c in usable if c["scope"] == "visit_total"), usable[0])
    result.update({k: preferred[k] for k in ("value", "range", "approximate", "scope")})
    result["evidence"] = [e for c in usable for e in c["evidence"]]
    result["status"] = "range" if preferred["range"] else "reported"
    return result


def hour_candidates(lines: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"on_site": [], "travel": []}
    n = rf"(?P<value>{NUMBER})(?:\s*[-–]\s*(?P<upper>{NUMBER}))?"
    patterns = {
        "on_site": [rf"(?:total\s+)?(?:on[ -]?site(?:\s+(?:hours|time|work))?|hours\s+on[ -]?site|time\s+on[ -]?site|site\s+hours|total\s+hours\s+on[ -]?site)(?:\s+(?:was|were|came\s+to|worked|for))?\s*:?\s*(?:about|approx\.?|approximately|~)?\s*{n}",
                    rf"{n}\s*(?:hours?|hrs?|h)?\s*(?:on[ -]?site|at\s+site)"],
        "travel": [rf"(?:total\s+)?travel(?:led|ed|ing)?(?:\s+(?:hours|time|was|total|here))?(?:\s*\([^)]*\))?\s*:?\s*(?:about|roughly|around|approx\.?|~)?\s*{n}",
                   rf"{n}\s*(?:hours?|hrs?|h)?\s*(?:of\s+|for\s+|spent\s+)?travel(?:ing)?"]}
    for i, raw in enumerate(lines):
        text = clean(raw)
        if text.startswith("|"):
            continue
        for name, rules in patterns.items():
            for rule in rules:
                for m in re.finditer(rule, text, re.I):
                    lower, upper = number(m.group("value")), number(m.group("upper") or "")
                    scope = "visit_total"
                    tail = text[m.end():m.end() + 40]
                    if name == "travel" and re.search(r"each way", tail, re.I):
                        scope = "each_way"
                    elif name == "travel" and re.search(r"travel(?:ed|led)? here", m.group(0), re.I):
                        scope = "outbound_only"
                    left_context = re.split(r"[,;]", text[max(0, m.start()-20):m.start()])[-1]
                    result[name].append({"value": None if upper else lower,
                                         "range": {"lower": lower, "upper": upper} if upper else None,
                                         "approximate": bool(re.search(r"about|approx|roughly|around|nearly|~", left_context + m.group(0), re.I)),
                                         "scope": scope, "evidence": evidence(i + 1)})
        if re.search(r"travel", text, re.I):
            close = re.search(rf"close to\s+({NUMBER})\s*h", text, re.I)
            if close:
                result["travel"].append({"value": number(close.group(1)), "range": None,
                    "approximate": True, "scope": "visit_total", "evidence": evidence(i + 1)})
            outbound = re.search(rf"travel(?:ed|led)\s+in\s+morning\s*\([^)]*?({NUMBER})\s*[-–]\s*({NUMBER})\s*h", text, re.I)
            if outbound:
                result["travel"].append({"value": None,
                    "range": {"lower": number(outbound.group(1)), "upper": number(outbound.group(2))},
                    "approximate": True, "scope": "outbound_only", "evidence": evidence(i + 1)})
    return result


def parse_text(text: str, *, source_path: str = "synthetic.txt", source_sha256: str | None = None) -> dict[str, Any]:
    lines = text.splitlines()
    source_hash = source_sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
    record: dict[str, Any] = {"schema_version": SCHEMA_VERSION,
        "report_id": None, "source": {"path": source_path, "sha256": source_hash, "line_count": len(lines), "encoding": "utf-8"},
        "customer": None, "site": None, "country": None, "visit_date": {},
        "equipment": {"family": None, "model": None, "description": None, "evidence": []},
        "service_tags": [], "service_tag_evidence": {}, "hours": {}, "tasks": [], "materials": [],
        "replacement_quantities": [], "service_quantities": [], "cost": {"line_items": [], "reported_subtotal": None, "reported_vat": None,
            "vat_percent": None, "reported_total": None, "tax_basis": "not_stated", "checks": [], "line_item_sum": None},
        "field_provenance": {}, "warnings": []}
    m = re.search(r"\bFSR-\d+\b", text)
    if m:
        record["report_id"] = m.group(0)
        record["field_provenance"]["report_id"] = evidence(text[:m.start()].count("\n") + 1)
    for field, label in [("customer", r"Customer|Cust"), ("site", r"Site Location|Site|Location|Loc")]:
        value, refs = labelled(lines, label)
        record[field] = value
        if refs:
            record["field_provenance"][field] = refs
    # Unlabelled opening address/customer lines are retained only by their format.
    if record["site"] is None:
        for i, line in enumerate(lines[:12]):
            if re.fullmatch(r"[\w .'-]+,\s*(?:Australia|Canada|Chile|Finland|Peru|Russia|South Africa|Sweden|Zambia)", clean(line), re.I):
                record["site"] = clean(line)
                record["field_provenance"]["site"] = evidence(i + 1)
                if record["customer"] is None and i > 0 and clean(lines[i - 1]):
                    record["customer"] = clean(lines[i - 1])
                    record["field_provenance"]["customer"] = evidence(i)
                break
    if record["site"]:
        site = record["site"]
        country_rules = [("Australia", r"\b(?:Australia|AU)\b"), ("Finland", r"\b(?:Finland|FIN)\b"),
                         ("Sweden", r"\b(?:Sweden|Swed)\b"), *[(x, rf"\b{re.escape(x)}\b") for x in ["Canada", "Chile", "Peru", "Russia", "South Africa", "Zambia"]]]
        record["country"] = next((country for country, pat in country_rules if re.search(pat, site, re.I)), None)
        if record["country"]:
            record["field_provenance"]["country"] = record["field_provenance"]["site"]
    raw_date, refs = labelled(lines, r"Visit Date|Date of Visit|Visit date|Date")
    record["visit_date"] = normalize_date(raw_date)
    record["visit_date"]["evidence"] = refs
    description, erefs = labelled(lines, r"Equipment(?: Serviced| Worked On| Type)?|Equip|Type")
    for family, pattern, model in EQUIPMENT_RULES:
        candidates = [(i + 1, clean(line)) for i, line in enumerate(lines) if re.search(pattern, line, re.I)]
        if candidates:
            line_no, line = candidates[0]
            record["equipment"] = {"family": family, "model": model,
                                   "description": description or line, "evidence": erefs or evidence(line_no)}
            exact = re.search(r"\b(?:MD\d{3}|VSP[ -]?750|PF-\d+/\d+|Orion HZ)\b", text, re.I)
            if exact:
                record["equipment"]["model"] = exact.group(0)
                record["field_provenance"]["equipment.model"] = evidence(text[:exact.start()].count("\n") + 1)
            if family == "columncell" and not re.search(r"ColumnCell", text, re.I):
                record["equipment"]["family"] = None
                record["equipment"]["model"] = None
                warning(record, "equipment_family_not_explicit", "SonicSparger does not establish the serviced equipment's family.", evidence(line_no))
            break
    # Model/family normalization is a classification rule, not a recovered asset ID.
    record["equipment"]["classification_method"] = "ordered_source_phrase_rules"
    for tag, pattern in SERVICE_RULES:
        refs = [evidence(i + 1)[0] for i, line in enumerate(lines) if re.search(pattern, line, re.I)]
        if refs:
            record["service_tags"].append(tag)
            record["service_tag_evidence"][tag] = refs
    candidates = hour_candidates(lines)
    all_tables = tables(lines)
    cost_context_codes = {code for table in all_tables for code in currency(" ".join(table["header"]))[:1] if code}
    context = next(iter(cost_context_codes)) if len(cost_context_codes) == 1 else None
    customer_spares = bool(re.search(r"parts supplied from customer|using customer[’']s on-site spares", text, re.I))
    table_lines: set[int] = set()
    for table in all_tables:
        header = table["header"]
        htext = " ".join(header)
        ctx, _ = currency(htext, context)
        duration = column(header, r"time spent|duration")
        qcol = column(header, r"qty|quantity|units/hours")
        ratecol = column(header, r"rate|unit cost")
        acol = column(header, r"total|amount|subtotal")
        partcol = column(header, r"part number|reference no")
        unitcol = column(header, r"^unit$|^units$")
        material_table = partcol is not None or column(header, r"^source$") is not None
        cost_table = ratecol is not None and not material_table
        for line_no, cells in table["rows"]:
            table_lines.add(line_no)
            desc = cell(cells, 0)
            if not desc:
                continue
            refs = evidence(line_no)
            if duration is not None:
                hours = number(cell(cells, duration))
                if re.search(r"total.*on[ -]?site", desc, re.I):
                    if hours is not None:
                        candidates["on_site"].append({"value": hours, "range": None, "approximate": False, "scope": "visit_total", "evidence": refs})
                else:
                    record["tasks"].append({"description": desc, "hours": hours,
                        "equipment": cell(cells, column(header, r"^equipment$")) or None, "evidence": refs})
                continue
            if not (cost_table or material_table):
                continue
            amt = money(cell(cells, acol), ctx) if acol is not None else {"amount": None, "currency": ctx, "currency_symbol": None}
            amt["evidence"] = refs
            if cost_table and re.match(r"(?:grand total|total(?: amount due)?|subtotal|VAT)", desc, re.I):
                key = "reported_vat" if re.match(r"VAT", desc, re.I) else "reported_subtotal" if re.match(r"subtotal", desc, re.I) else "reported_total"
                record["cost"][key] = amt
                if key == "reported_vat":
                    pct = re.search(rf"({NUMBER})\s*%", desc)
                    record["cost"]["vat_percent"] = number(pct.group(1)) if pct else None
                continue
            qty, unit = quantity(cell(cells, qcol), cell(cells, unitcol))
            rate = money(cell(cells, ratecol), ctx) if ratecol is not None else {"amount": None, "currency": ctx, "currency_symbol": None}
            rateunit = re.search(rf"/\s*({UNIT_PATTERN})\b", cell(cells, ratecol), re.I)
            rate_unit = UNIT_ALIASES.get(rateunit.group(1).lower(), rateunit.group(1)) if rateunit else None
            item = {"description": desc, "quantity": qty, "quantity_unit": unit,
                    "unit_rate": rate["amount"], "rate_unit": rate_unit,
                    "unit_rate_currency": rate["currency"],
                    "amount": amt["amount"], "currency": amt["currency"] or rate["currency"],
                    "currency_symbol": amt["currency_symbol"] or rate["currency_symbol"],
                    "customer_supplied": True if customer_spares and categorize(desc) not in {"travel", "labour"} else None,
                    "evidence": refs, "header_evidence": evidence(table["line"])}
            item["rate_basis"] = "explicit_denominator" if rateunit else "unit_rate_header" if re.search(r"unit (?:rate|cost)", htext, re.I) else "not_stated"
            if material_table:
                raw_part = cell(cells, partcol)
                item["part_number"] = raw_part if raw_part and raw_part.upper() not in {"N/A", "NA", "-", "–"} else None
                item["source_note"] = cell(cells, column(header, r"^source$")) or None
                if item["amount"] is None and item["unit_rate"] is None:
                    item["currency"] = None
                    item["unit_rate_currency"] = None
                record["materials"].append(item)
            else:
                item["category"] = categorize(desc)
                record["cost"]["line_items"].append(item)
                stated_hours = unit == "hour" or rate_unit == "hour" or bool(re.search(r"hours|hrs|\bhr\b", htext + " " + desc, re.I))
                if item["category"] in {"travel", "labour"} and qty is not None and stated_hours:
                    candidates["travel" if item["category"] == "travel" else "on_site"].append(
                        {"value": qty, "range": None, "approximate": False, "scope": "billed_hours", "evidence": refs})
            if "cloth" in desc.lower() and qty is not None and not re.search(r"inspection", desc, re.I):
                record["replacement_quantities"].append({"item": "filter_cloth", "quantity": qty,
                    "quantity_unit": unit, "scope": "material_list" if material_table else "billed_line", "evidence": refs})
                record["service_quantities"].append({"item": "filter_cloth", "quantity": qty,
                    "quantity_unit": unit, "activity": "material_or_billed", "customer_supplied": item["customer_supplied"], "evidence": refs})
    # Narrative totals: extract the monetary tail, not earlier hours on the line.
    for i, raw in enumerate(lines):
        if i + 1 in table_lines:
            continue
        line = clean(raw)
        m = re.search(r"\b(?:grand total|total due|cost total|total amount due|total(?: service)? cost(?: for[^:]*?)?)(?:\s*\([^)]*\))?\s*(?::|\bis\b)\s*(.+)", line, re.I)
        if m:
            tail = m.group(1).strip().rstrip(".")
            # 'is' may follow a descriptive clause after a comma.
            tail = re.sub(r"^.*?\bis\s+(?=[$€\d]|(?:USD|EUR|CAD|AUD|SEK|ZAR)\b)", "", tail, flags=re.I)
            tailmoney = money(tail, currency(line, context)[0])
            if tailmoney["amount"] is not None:
                tailmoney["evidence"] = evidence(i + 1)
                existing = record["cost"]["reported_total"]
                if existing and (existing["amount"], existing["currency"]) != (tailmoney["amount"], tailmoney["currency"]):
                    warning(record, "conflicting_printed_totals", "Printed totals disagree; both are retained in checks.", evidence(i + 1))
                    record["cost"]["checks"].append({"kind": "alternative_printed_total", "value": tailmoney})
                else:
                    record["cost"]["reported_total"] = tailmoney
            if re.search(r"ex\.?\s*VAT", line, re.I):
                record["cost"]["tax_basis"] = "exclusive_vat"
    if record["cost"]["reported_vat"]:
        record["cost"]["tax_basis"] = "inclusive_vat"
    # Explicit narrative materials; no counts taken from part numbers or models.
    in_materials = False
    for i, raw in enumerate(lines):
        line = clean(raw)
        inline_parts = re.match(r"^(?:main )?parts(?: used)?\s*:\s*(.+)$", line, re.I)
        if inline_parts:
            for item_desc in re.split(r",\s+(?![^()]*\))", inline_parts.group(1)):
                if re.search(r"none required|no other consumables", item_desc, re.I):
                    continue
                qmatch = re.match(rf"({NUMBER})\s*(?:x\s*|({UNIT_PATTERN})\b\s*)", item_desc, re.I)
                record["materials"].append({"description": item_desc.strip(),
                    "quantity": number(qmatch.group(1)) if qmatch else None,
                    "quantity_unit": UNIT_ALIASES.get((qmatch.group(2) or "").lower()) if qmatch else None,
                    "part_number": None, "unit_rate": None, "unit_rate_currency": None, "rate_unit": None, "amount": None,
                    "currency": None, "currency_symbol": None, "customer_supplied": None,
                    "source_note": None, "evidence": evidence(i + 1)})
            continue
        if re.search(r"^(?:#+\s*(?:\d+\.\s*)?)?(?:main )?materials\s*(?:and|&|/)?\s*parts used|^(?:main )?parts used|^materials/parts used", line, re.I):
            in_materials = True
            continue
        if in_materials and (line.startswith("---") or line.startswith("#") or re.match(r"(?:findings|work completed|notes|cost summary)\s*:", line, re.I)):
            in_materials = False
        if in_materials and line.startswith("-") and i + 1 not in table_lines:
            desc = line[1:].strip()
            if re.search(r"no replacement parts|none required", desc, re.I):
                continue
            qm = re.match(rf"({NUMBER})\s*x\s*(.+)", desc, re.I)
            qty, unit = (number(qm.group(1)), None) if qm else (None, None)
            if not qm:
                qm2 = re.search(rf"(?:qty\s*:\s*|[–—]\s*)({NUMBER})(?:\s*({UNIT_PATTERN})\b)?|\(({NUMBER})\s*({UNIT_PATTERN})\)|(?:,\s*|\()({NUMBER})\s*(L|ml)\)", desc, re.I)
                if qm2:
                    groups = qm2.groups()
                    pos = next(k for k in (0, 2, 4) if groups[k])
                    qty = number(groups[pos]); unit = UNIT_ALIASES.get((groups[pos + 1] or "").lower())
            pnum = re.search(r"\bP/N\s+([\w-]+)", desc, re.I)
            record["materials"].append({"description": desc, "quantity": qty, "quantity_unit": unit,
                "part_number": pnum.group(1) if pnum else None, "unit_rate": None, "unit_rate_currency": None, "rate_unit": None, "amount": None, "currency": None,
                "currency_symbol": None, "customer_supplied": None, "source_note": None, "evidence": evidence(i + 1)})
            if "cloth" in desc.lower() and qty is not None:
                record["replacement_quantities"].append({"item": "filter_cloth", "quantity": qty,
                    "quantity_unit": unit, "scope": "material_list", "evidence": evidence(i + 1)})
                record["service_quantities"].append({"item": "filter_cloth", "quantity": qty,
                    "quantity_unit": unit, "activity": "material_list", "customer_supplied": None, "evidence": evidence(i + 1)})
        replacement = re.search(rf"(?:replaced?|replacement)\s+(?P<n>{NUMBER})(?:\s*/\s*(?P<den>{NUMBER}))?\s*(?:worn or damaged\s+)?(?:filter\s+)?cloths", line, re.I)
        if not replacement:
            replacement = re.search(rf"cloth replacement\s*\((?P<n>{NUMBER})\s*pcs\)", line, re.I)
        if replacement:
            record["replacement_quantities"].append({"item": "filter_cloth", "quantity": number(replacement.group("n")),
                "quantity_unit": "piece" if "pcs" in replacement.group(0) else None, "scope": "performed_statement",
                "planned_or_inspected_total": number(replacement.groupdict().get("den") or ""), "evidence": evidence(i + 1)})
            record["service_quantities"].append({"item": "filter_cloth", "quantity": number(replacement.group("n")),
                "quantity_unit": "piece" if "pcs" in replacement.group(0) else "cloth", "activity": "replacement",
                "planned_or_inspected_total": number(replacement.groupdict().get("den") or ""),
                "customer_supplied": None, "evidence": evidence(i + 1)})
        plate_scope = re.search(rf"(?:changed?|replaced?)\s+(?:out\s+)?filter cloths on\s+({NUMBER})\s+plates", line, re.I)
        if plate_scope:
            record["service_quantities"].append({"item": "filter_plate", "quantity": number(plate_scope.group(1)),
                "quantity_unit": "plate", "activity": "cloth_replacement", "customer_supplied": None, "evidence": evidence(i + 1)})
            warning(record, "cloth_count_not_stated", "A cloth change is counted in plates, which is not converted to a cloth count.", evidence(i + 1))
        inspected = re.search(rf"filter cloth inspection\s*\([^)]*?({NUMBER})\s*pcs\)", line, re.I)
        if inspected:
            record["service_quantities"].append({"item": "filter_cloth", "quantity": number(inspected.group(1)),
                "quantity_unit": "piece", "activity": "inspection", "customer_supplied": None, "evidence": evidence(i + 1)})
        if re.search(r"replaced cloths w/ spares \(they had\)", line, re.I) or re.search(r"seal.*stock from Plant stores", line, re.I):
            record["materials"].append({"description": line, "quantity": None, "quantity_unit": None, "part_number": None,
                "unit_rate": None, "unit_rate_currency": None, "rate_unit": None, "amount": None, "currency": None, "currency_symbol": None,
                "customer_supplied": True, "source_note": "Explicit customer/plant inventory statement", "evidence": evidence(i + 1)})
    hour_measures: dict[str, Any] = {name: measure(values, name, record) for name, values in candidates.items()}
    record["hours"] = hour_measures
    cloth_supply = [(i + 1) for i, line in enumerate(lines) if re.search(r"replaced cloths w/ spares \(they had\)", line, re.I)]
    if cloth_supply:
        for item in record["service_quantities"]:
            if item["activity"] in {"replacement", "cloth_replacement"}:
                item["customer_supplied"] = True
                item["customer_supply_scope"] = "replacement_cloths"
                item["supply_evidence"] = evidence(cloth_supply[0])
    # Break/activity notes are separate evidence, not reconstructed total hours.
    record["hours"]["context_notes"] = []
    for i, line in enumerate(lines):
        if re.search(r"lunch|half a day|actual work|total commissioning", line, re.I):
            record["hours"]["context_notes"].append({"kind": "break_or_activity_scope", "evidence": evidence(i + 1)})
    record["hours"]["on_site"]["includes_breaks"] = None
    for i, line in enumerate(lines):
        if re.search(r"(?:incl\.?|including)[^\n]{0,20}lunch", line, re.I):
            record["hours"]["on_site"]["includes_breaks"] = True
            record["field_provenance"]["hours.on_site.includes_breaks"] = evidence(i + 1)
        elif re.search(r"lunch.{0,40}not incl|not incl.{0,20}lunch", line, re.I):
            record["hours"]["on_site"]["includes_breaks"] = False
            record["field_provenance"]["hours.on_site.includes_breaks"] = evidence(i + 1)
    if "\ufffd" in text:
        warning(record, "source_encoding_replacement_character", "The source contains replacement characters; original bytes are preserved.")
    if re.search(r"\b(?:did not finish|didn't finish|partial\s+(?:cloth|job|work|diag)|tasks left|actual work .*maybe)\b", text, re.I):
        warning(record, "incomplete_or_mixed_work_scope", "The narrative includes incomplete or alternative activity scope; service tags do not mean every activity was completed.")
    if re.search(r"travel.{0,50}each way.{0,70}total just over", text, re.I):
        record["hours"]["travel"]["value"] = None
        record["hours"]["travel"]["status"] = "ambiguous"
        warning(record, "travel_combined_total_ambiguous", "Each-way travel and an unqualified combined total cannot establish round-trip travel hours.")
    for i, line in enumerate(lines):
        if re.search(r"travel.{0,50}(?:like|maybe).{0,50}each way", line, re.I):
            warning(record, "travel_vague_each_way_alternative", "Vague each-way travel is mentioned alongside other scope/context. Any separately declared total is retained, but this narrative needs review.", evidence(i + 1))
    if record["visit_date"]["status"] == "ambiguous":
        warning(record, "date_format_ambiguous", "Only an ISO date is normalized; slash/hyphen abbreviations retain the source value.", refs=record["visit_date"]["evidence"])
    nonempty = [(i + 1, clean(line)) for i, line in enumerate(lines) if clean(line)]
    if nonempty and nonempty[-1][1].startswith("|") and record["cost"]["line_items"] and record["cost"]["line_items"][-1]["amount"] is None:
        warning(record, "cost_table_ends_with_missing_amount", "The source ends within a cost-table row; no final charge is recovered.", evidence(nonempty[-1][0]))
    validate_costs(record)
    record["typed_validation_errors"] = validate_record(record)
    return record


def validate_costs(record: dict[str, Any]) -> None:
    cost = record["cost"]
    priced_materials = [item for item in record["materials"] if item["unit_rate"] is not None or item["amount"] is not None]
    for item in cost["line_items"] + priced_materials:
        q, rate, amount = item["quantity"], item["unit_rate"], item["amount"]
        check = {"kind": "material_quantity_times_unit_cost" if item in priced_materials else "quantity_times_rate", "description": item["description"], "computed_amount": None,
                 "reported_amount": amount, "currency": item["currency"], "status": "not_checkable", "evidence": item["evidence"]}
        if q is not None and rate is not None:
            check["computed_amount"] = q * rate
            uncertain_unit_basis = item["quantity_unit"] is not None and item["rate_unit"] is None and item["rate_basis"] == "not_stated" and item["quantity_unit"] != "hour"
            if uncertain_unit_basis:
                check["status"] = "ambiguous_rate_basis"
                warning(record, "rate_basis_not_stated", "Quantity has a physical unit but rate denominator is missing; multiplication is diagnostic only.", item["evidence"])
            elif amount is not None and item["unit_rate_currency"] != item["currency"]:
                check["status"] = "currency_mismatch"
                warning(record, "line_rate_currency_mismatch", "Rate and printed line amount use different currencies; no conversion is made.", item["evidence"])
            elif amount is not None:
                check["status"] = "match" if (q * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) == amount else "mismatch"
                if check["status"] == "mismatch":
                    warning(record, "line_amount_mismatch", "Printed line amount differs from quantity times rate; source values are retained.", item["evidence"])
        if amount is None:
            warning(record, "cost_line_amount_missing", "A cost-summary line lacks a numeric printed amount.", item["evidence"])
        if amount is not None and item["currency"] is None:
            warning(record, "currency_unspecified", "A numeric monetary amount lacks an unambiguous currency code or euro symbol.", item["evidence"])
        cost["checks"].append(check)
    items = cost["line_items"]
    codes = {item["currency"] for item in items if item["amount"] is not None}
    if items and len(codes) == 1:
        cost["line_item_sum"] = {"amount": sum((item["amount"] for item in items if item["amount"] is not None), Decimal(0)),
            "currency": next(iter(codes)), "complete": all(item["amount"] is not None for item in items),
            "evidence": [e for item in items for e in item["evidence"]]}
        target = cost["reported_subtotal"] or cost["reported_total"]
        if target and cost["line_item_sum"]["complete"]:
            same_currency = target["currency"] == cost["line_item_sum"]["currency"]
            status = "match" if same_currency and target["amount"] == cost["line_item_sum"]["amount"] else "mismatch" if same_currency else "currency_mismatch"
            cost["checks"].append({"kind": "line_sum_vs_printed", "status": status,
                "computed_amount": cost["line_item_sum"]["amount"], "reported_amount": target["amount"],
                "currency": target["currency"], "evidence": target["evidence"]})
            if status != "match":
                warning(record, "printed_total_mismatch", "Printed subtotal/total disagrees with the complete line-item sum or currency.", target["evidence"])
    elif items and len(codes) > 1:
        warning(record, "mixed_line_currencies", "Line items contain different currencies; no aggregate amount is computed.")
    total = cost["reported_total"]
    if total is None:
        warning(record, "reported_total_missing", "No numeric final visit total is printed; a known-component sum is not a final total.")
    elif total["currency"] is None:
        warning(record, "currency_unspecified", "Final amount has a dollar symbol without an explicit currency code.", total["evidence"])
    subtotal, vat = cost["reported_subtotal"], cost["reported_vat"]
    if subtotal and vat and total:
        same = subtotal["currency"] == vat["currency"] == total["currency"]
        computed = subtotal["amount"] + vat["amount"]
        status = "match" if same and computed == total["amount"] else "mismatch" if same else "currency_mismatch"
        cost["checks"].append({"kind": "subtotal_plus_vat", "computed_amount": computed, "reported_amount": total["amount"], "status": status, "evidence": total["evidence"]})
        if status != "match":
            warning(record, "vat_total_mismatch", "Subtotal plus VAT differs from printed final amount.", total["evidence"])
        if cost["vat_percent"] is not None:
            calculated_vat = (subtotal["amount"] * cost["vat_percent"] / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            cost["checks"].append({"kind": "vat_percentage", "computed_amount": calculated_vat, "reported_amount": vat["amount"], "status": "match" if calculated_vat == vat["amount"] else "mismatch", "evidence": vat["evidence"]})
    if record["tasks"] and all(t["hours"] is not None for t in record["tasks"]):
        task_sum = sum((t["hours"] for t in record["tasks"]), Decimal(0))
        onsite = record["hours"]["on_site"]["value"]
        if onsite is not None:
            record["hours"]["task_sum_check"] = {"computed_hours": task_sum, "reported_hours": onsite, "status": "match" if task_sum == onsite else "mismatch"}
            if task_sum != onsite:
                warning(record, "task_hours_mismatch", "Listed task durations differ from reported on-site total; neither is overwritten.", record["hours"]["on_site"]["evidence"])


def parse_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    try:
        source_path = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source_path = path.name
    return parse_text(raw.decode("utf-8-sig"), source_path=source_path,
                      source_sha256=hashlib.sha256(raw).hexdigest())


def validate_record(record: dict[str, Any]) -> list[str]:
    """Check normalized numeric fields and every recorded line reference."""
    errors: list[str] = []
    numeric_keys = {"quantity", "unit_rate", "amount", "value", "lower", "upper", "hours",
                    "computed_amount", "reported_amount", "computed_hours", "reported_hours",
                    "vat_percent", "planned_or_inspected_total"}
    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else key
                if key in numeric_keys and item is not None and not isinstance(item, (dict, Decimal, int)):
                    # Date values are strings; alternative-printed-total values are objects.
                    if not (path == "visit_date" and key == "value"):
                        errors.append(f"{child_path}: numeric field is not a JSON number/null")
                if key in numeric_keys and isinstance(item, bool):
                    errors.append(f"{child_path}: boolean is not a quantity")
                if key in {"currency", "unit_rate_currency"} and item is not None and item not in CURRENCIES:
                    errors.append(f"{child_path}: unsupported currency")
                if key == "customer_supplied" and item is not None and not isinstance(item, bool):
                    errors.append(f"{child_path}: expected boolean/null")
                if key == "line_start":
                    end = value.get("line_end")
                    if isinstance(item, bool) or not isinstance(item, int) or not isinstance(end, int) or not (1 <= item <= end <= record["source"]["line_count"]):
                        errors.append(f"{path}: source line range is invalid")
                walk(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
    walk(record)
    return errors


def quality_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    warnings = Counter(w["code"] for r in records for w in r["warnings"])
    return {"schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION, "reports_parsed": len(records),
        "invalid_typed_records": sum(bool(r["typed_validation_errors"]) for r in records),
        "with_reported_total": sum(r["cost"]["reported_total"] is not None for r in records),
        "without_reported_total": sum(r["cost"]["reported_total"] is None for r in records),
        "totals_with_unspecified_currency": sum(r["cost"]["reported_total"] is not None and r["cost"]["reported_total"]["currency"] is None for r in records),
        "with_cost_line_items": sum(bool(r["cost"]["line_items"]) for r in records),
        "with_customer_supplied_materials": sum(any(m["customer_supplied"] is True for m in r["materials"]) for r in records),
        "hour_status_counts": {key: dict(Counter(r["hours"][key]["status"] for r in records)) for key in ("on_site", "travel")},
        "country_missing": sum(r["country"] is None for r in records),
        "equipment_family_missing": sum(r["equipment"]["family"] is None for r in records),
        "date_status_counts": dict(Counter(r["visit_date"]["status"] for r in records)),
        "task_hour_checks": dict(Counter(r["hours"]["task_sum_check"]["status"] for r in records if "task_sum_check" in r["hours"])),
        "warning_counts": dict(sorted(warnings.items())),
        "checks": dict(Counter(c["kind"] + ":" + c.get("status", "retained") for r in records for c in r["cost"]["checks"])),
        "limitations": ["Rule classifications preserve source scope; tags do not certify completion.", "Missing final costs, unspecified dollars, ranges and partial travel are not resolved by inference.", "All records and report-level quality findings contain private source material."]}


def ingest_services(data_dir: str | Path, output: str | Path) -> dict[str, Any]:
    data_dir, output = Path(data_dir), Path(output)
    source_dir = data_dir / "HistoricalServices" if (data_dir / "HistoricalServices").is_dir() else data_dir
    paths = sorted(source_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No service reports in {source_dir}")
    records = [parse_report(path) for path in paths]
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.jsonl").write_text("\n".join(json_numbers(r) for r in records) + "\n", encoding="utf-8")
    quality = quality_report(records)
    detailed_quality = {**quality, "report_findings": [{"report_id": r["report_id"], "warnings": r["warnings"]} for r in records if r["warnings"]]}
    (output / "quality-report.json").write_text(json_numbers(detailed_quality) + "\n", encoding="utf-8")
    manifest = {"schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION, "operation": "deterministic_service_ingestion",
        "inputs": [r["source"] for r in records], "records_path": "records.jsonl", "records_sha256": hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest(),
        "quality_report_path": "quality-report.json", "numeric_serialization": "exact Decimal JSON number tokens", "reports_parsed": len(records)}
    (output / "manifest.json").write_text(json_numbers(manifest) + "\n", encoding="utf-8")
    return {"manifest": manifest, "quality": quality, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(".pilot-cache/assistant/services"))
    args = parser.parse_args(argv)
    result = ingest_services(args.data_dir, args.output)
    print(json.dumps(result["quality"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
