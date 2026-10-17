"""Normalize table cells without inventing model associations or missing values.

Raw parser/geometry cells remain the evidence. A candidate fact is not verified
merely because it has a numeric type; content-bound review is a separate step.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import re
from typing import Any

NORMALIZER_VERSION = "table-normalizer-1"
UNITS = {
    "m2": "m²", "m²": "m²", "m^2": "m²", "m3": "m³", "m³": "m³", "m^3": "m³",
    "mm": "mm", "m": "m", "cm": "cm", "kg": "kg", "lbs": "lb", "lb": "lb",
    "inches": "in", "inch": "in", "in": "in", "t": "t", "bar": "bar",
    "l": "L", "liter": "L", "litre": "L", "liters": "L", "litres": "L",
    "pcs": "pcs", "kw": "kW", "hp": "hp", "hz": "Hz", "rpm": "rpm",
    "r/min": "rpm", "°c": "°C", "%": "%", "pa": "Pa", "kpa": "kPa",
    "m3/h": "m³/h", "m³/h": "m³/h", "m^3/h": "m³/h", "t/m3": "t/m³",
    "t/m³": "t/m³", "μm": "µm", "µm": "µm", "pcs.": "pcs",
}
MISSING = {"", "-", "–", "—", "n/a", "na", "not stated"}


def text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def unit(value: str) -> str | None:
    token = text(value).lower().replace(" ", "")
    return UNITS.get(token)


def issue(code: str, message: str, severity: str = "blocker") -> dict:
    return {"code": code, "message": message, "severity": severity}


def numeric(raw: str, physical_unit: str | None = None) -> dict:
    """Parse unambiguous decimal/grouping forms, retaining every interpretation."""
    original = text(raw)
    if original.lower() in MISSING:
        return {"value": None, "status": "missing", "value_kind": "scalar", "issues": []}
    token = original.replace("\u202f", " ").replace("\u00a0", " ")
    token = token.replace("−", "-")
    diagnostics = []
    if re.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?", token):
        token = token.replace(" ", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+", token):
        token = token.replace(",", "")
    elif re.fullmatch(r"[+-]?\d+,\d{1,2}", token):
        token = token.replace(",", ".")
    elif re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", token):
        grouping = Decimal(token.replace(",", ""))
        decimal = Decimal(token.replace(",", ".")) if token.count(",") == 1 else None
        if physical_unit in {"mm", "kg", "lb", "pcs", "L", "rpm"} or token.count(",") > 1:
            token = token.replace(",", "")
            diagnostics.append(issue("comma_grouping_convention", "Comma grouping interpretation requires source review; printed text is retained.", "warning"))
        else:
            return {"value": None, "status": "ambiguous", "value_kind": "scalar", "numeric_candidates": [float(v) for v in (grouping, decimal) if v is not None], "issues": [issue("numeric_separator_ambiguity", "A single comma with three following digits has multiple interpretations.")]}
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token):
        return {"value": None, "status": "reported", "value_kind": "text", "issues": diagnostics}
    try:
        number = Decimal(token)
    except InvalidOperation:
        return {"value": None, "status": "invalid", "value_kind": "scalar", "issues": [issue("invalid_number", "Numeric token could not be parsed.")]}
    return {"value": int(number) if number == number.to_integral_value() else float(number), "status": "reported", "value_kind": "scalar", "issues": diagnostics}


def grid_from_cells(cells: list[dict], rows: int, columns: int) -> list[list[dict]]:
    if not (0 < rows <= 1000 and 0 < columns <= 1000 and rows * columns <= 100000):
        raise ValueError("Unsupported table dimensions")
    grid = [[{"text": "", "cell_id": f"gap-{r}-{c}", "row_index": r, "col_index": c, "row_span": 1, "col_span": 1, "structural_gap": True} for c in range(columns)] for r in range(rows)]
    occupied = set()
    for index, raw in enumerate(cells):
        cell = {**raw, "cell_id": raw.get("cell_id", f"cell-{index:04d}"), "raw_text": raw.get("text", ""), "text": text(raw.get("text"))}
        r, c = int(cell["row_index"]), int(cell["col_index"])
        rs, cs = int(cell.get("row_span", 1)), int(cell.get("col_span", 1))
        if r < 0 or c < 0 or rs < 1 or cs < 1 or r + rs > rows or c + cs > columns:
            raise ValueError("Cell span escapes table")
        for y in range(r, r + rs):
            for x in range(c, c + cs):
                if (y, x) in occupied:
                    raise ValueError("Overlapping table cells")
                occupied.add((y, x)); grid[y][x] = cell
    return grid


def unique_cells(row: list[dict], start: int = 0) -> list[dict]:
    output, seen = [], set()
    for cell in row[start:]:
        key = (cell["row_index"], cell["col_index"])
        if key not in seen:
            output.append(cell); seen.add(key)
    return output


def join_headers(cells: list[dict]) -> str:
    values = []
    for cell in cells:
        value = text(cell["text"])
        if value and value not in values:
            values.append(value)
    return " / ".join(values)


def scalar_id(cell: dict) -> str | None:
    value = text(cell["text"])
    if re.fullmatch(r"\d+(?:[.,]\d+)?", value):
        return value
    if re.fullmatch(r"[A-Za-z]{1,8}[ -]?\d+[A-Za-z0-9/-]*[ •*]*", value):
        return value.strip(" •*")
    return None


def anchor_columns(grid: list[list[dict]], start: int) -> tuple[int, list[dict], list[dict]]:
    """Use actual data-column spans, then source header labels where available."""
    preferred = []
    for r, row in enumerate(grid):
        label = text(row[0]["text"]).lower()
        rank = 0 if re.search(r"\bchambers\b", label) else 1 if re.search(r"filtration\s+area", label) else 2
        header_role = r <= 2 and (re.search(r"\b(model|series|type|pump|frame|size)\b", label)
                                      or not label and any(re.search(r"[A-Za-z]", c["text"]) and scalar_id(c) for c in unique_cells(row, start)))
        if rank == 2 and not header_role:
            continue
        cells = unique_cells(row, start)
        valid = [c for c in cells if c["col_index"] >= start and scalar_id(c)]
        if len(valid) >= 3 and len(valid) == len(cells):
            preferred.append((rank, r, valid))
    if not preferred:
        return 0, [], [issue("unrecognized_configuration_axis", "No reliable row defines the configuration columns.")]
    _, anchor_row, anchors = min(preferred, key=lambda item: (item[0], item[1]))
    notes = []
    for cell in anchors:
        label = None
        for row in grid[:anchor_row]:
            parents = unique_cells(row[cell["col_index"]:cell["col_index"] + cell.get("col_span", 1)])
            if len(parents) == 1 and parents[0]["col_index"] == cell["col_index"] and parents[0]["col_span"] == cell["col_span"]:
                label = scalar_id(parents[0]) or label
        cell["configuration"] = label or scalar_id(cell)
    # A collapsed header can contain ordered IDs. It is never silently trusted:
    # the reconstruction and its source locator remain explicit review inputs.
    if anchor_row and any(c["col_span"] >= len(grid[0]) - start for row in grid[:anchor_row] for c in unique_cells(row)):
        candidates = []
        for row in grid[:anchor_row]:
            for header in unique_cells(row):
                tail = re.search(r"(?:\bSERIES\b|\bMODEL\s+\d{4}\b)\s+([\d.,\s]+)$", header["text"], re.I)
                tokens = re.findall(r"\d+(?:\.\d+)?", tail[1]) if tail else []
                if len(tokens) == len(anchors):
                    candidates.append((header, tokens))
        if len(candidates) == 1 and len(set(candidates[0][1])) == len(anchors):
            header, labels = candidates[0]
            for cell, label in zip(anchors, labels):
                cell["configuration"] = label
                cell["header_reconstruction"] = {"source_cell": header["cell_id"], "ordered_tokens": labels}
            notes.append(issue("collapsed_configuration_header", "Ordered header reconstruction requires independent source association validation."))
        else:
            notes.append(issue("configuration_header_unresolved", "Collapsed header could not independently establish the configuration labels."))
    return anchor_row, anchors, notes


def make_fact(table: dict, configuration: str | None, parameter: str, cell: dict, printed_unit: str, locator: dict, association_issues: list[dict] | None = None) -> dict:
    physical_unit = unit(printed_unit)
    parsed = numeric(cell["text"], physical_unit)
    issues = [*parsed.pop("issues"), *(association_issues or [])]
    if cell.get("structural_gap"):
        issues.append(issue("unextracted_cell", "No extracted source cell supports this position; it is not an explicit blank."))
    if printed_unit and not physical_unit:
        issues.append(issue("unrecognized_printed_unit", "Printed unit retained without an inferred conversion.", "warning"))
    if "volume" in parameter.lower() and physical_unit == "m²":
        issues.append(issue("source_dimension_anomaly", "The source labels a volume with an area unit; no correction applied.", "warning"))
    if any(item["severity"] == "blocker" for item in issues):
        parsed.update(value=None, status="ambiguous")
    key = f"{table['table_id']}|{configuration}|{parameter}|{locator}"
    return {"fact_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:20],
            "product_family": table.get("product_family"), "configuration": configuration,
            "parameter": parameter, "raw_cell_text": cell.get("raw_text", cell["text"]), "raw_unit": printed_unit,
            "unit": physical_unit, **parsed, "issues": issues, "review_status": "pending",
            "source_refs": [{**table["source"], "table_id": table["table_id"], "cell_bbox": cell.get("bbox"), "cell_id": cell.get("cell_id"), "locator": locator}]}


def split_quantities(fact: dict) -> list[dict]:
    """Expose only explicitly paired quantities, preserving printed provenance."""
    raw, printed_unit = fact["raw_cell_text"], fact["raw_unit"]
    if fact["value_kind"] != "text" or fact["status"] == "ambiguous":
        return [fact]
    if re.fullmatch(r"\s*\d+(?:\.\d+){2,}\s*", raw):
        return [{**fact, "status": "ambiguous", "issues": fact["issues"] + [issue("malformed_numeric_punctuation", "Multiple decimal points cannot establish a scalar or a power/speed pair; printed text is retained.")]}]
    suffix_units = r"(?:t/m[³3]|m[³3]/h|kW|hp|Hz|rpm|r/min|mm|cm|kg|bar|µm|μm|m|inches|inch|in|%)"
    dimensions = re.fullmatch(rf"\s*([\d., ]+)\s*[x×]\s*([\d., ]+)\s*({suffix_units})?\s*", raw, re.I)
    if dimensions and (fact["unit"] or dimensions[3]):
        physical_unit = unit(dimensions[3]) if dimensions[3] else fact["unit"]
        components = [numeric(dimensions[i].strip(), physical_unit) for i in (1, 2)]
        if all(c["status"] == "reported" and c["value_kind"] == "scalar" for c in components):
            return [{**fact, "unit": physical_unit, "value_kind": "vector", "value_components": [c["value"] for c in components], "issues": fact["issues"] + [i for c in components for i in c["issues"]]}]
    range_match = re.fullmatch(rf"\s*([\d., ]+)\s*[-–—]\s*([\d., ]+)\s*({suffix_units})?\s*", raw, re.I)
    if range_match and not (not fact["unit"] and re.search(r"[-(]", printed_unit)):
        label = unit(range_match[3]) if range_match[3] else fact["unit"]
        bounds = [numeric(range_match[i].strip(), label) for i in (1, 2)]
        if all(b["status"] == "reported" and b["value_kind"] == "scalar" for b in bounds):
            if bounds[0]["value"] > bounds[1]["value"]:
                return [{**fact, "status": "ambiguous", "issues": fact["issues"] + [issue("reversed_range", "Printed range is reversed; no automatic swap.")]}]
            return [{**fact, "unit": label, "value_kind": "range", "value_range": {"lower": bounds[0]["value"], "upper": bounds[1]["value"]}, "issues": [i for i in fact["issues"] if i["code"] != "unrecognized_printed_unit"] + [i for b in bounds for i in b["issues"]]}]
    matches = list(re.finditer(rf"([+-]?\d[\d ,.]*?)\s*({suffix_units})(?![\w/])", raw, re.I))
    pairs = [(m[1], m[2]) for m in matches]
    operator = None
    if pairs:
        remaining = raw
        for match in reversed(matches):
            remaining = remaining[:match.start()] + remaining[match.end():]
        bound = re.match(r"\s*(up\s+to|at\s+most|at\s+least)\b", remaining, re.I)
        if bound and len(pairs) == 1:
            operator = ">=" if bound[1].lower() == "at least" else "<="
            remaining = remaining[bound.end():]
        if remaining.strip(" ,;()"):
            return [{**fact, "status": "ambiguous", "issues": fact["issues"] + [issue("unsupported_compound_semantics", "Units were found inside a choice, multiplier or qualified expression; the full printed meaning is retained without scalar promotion.")]}]
    if not pairs:
        units = re.fullmatch(r"\s*([A-Za-z]+)\s*\(\s*([A-Za-z/]+)\s*\)\s*", printed_unit)
        values = re.fullmatch(r"\s*([\d., ]+)\s*\(\s*([\d., ]+)\s*\)\s*", raw)
        if units and values and unit(units[1]) and unit(units[2]):
            pairs = [(values[1].strip(), units[1]), (values[2].strip(), units[2])]
    if not pairs and "-" in printed_unit:
        parts = printed_unit.split("-", 1)
        values = re.fullmatch(r"\s*([\d., ]+)\s*-\s*([\d., ]+)\s*", raw)
        if values and all(unit(p) for p in parts):
            pairs = [(values[1].strip(), parts[0]), (values[2].strip(), parts[1])]
    if pairs:
        result = []
        kinds = {"kW": "power", "hp": "power", "Hz": "frequency", "rpm": "rotational_speed"}
        for index, (value, label) in enumerate(pairs):
            physical_unit = unit(label)
            if not physical_unit:
                continue
            parsed = numeric(value.strip(" ,"), physical_unit)
            candidate = {**fact, **{k: v for k, v in parsed.items() if k != "issues"},
                         "fact_id": hashlib.sha256(f"{fact['fact_id']}|quantity|{index}".encode()).hexdigest()[:20],
                         "unit": physical_unit, "quantity_kind": kinds.get(physical_unit, physical_unit),
                         "issues": [i for i in fact["issues"] if i["code"] != "unrecognized_printed_unit"] + parsed["issues"]}
            if operator:
                candidate["operator"] = operator
            if printed_unit and not unit(printed_unit) and printed_unit.upper() in {"IEC", "NEMA"}:
                candidate["parameter"] += " / " + printed_unit
            result.append(candidate)
        return result or [fact]
    if fact["unit"] and re.fullmatch(r"\s*[\d., ]+\s*[x×]\s*[\d., ]+\s*", raw):
        components = [numeric(v.strip(), fact["unit"]) for v in re.split(r"[x×]", raw)]
        if all(c["status"] == "reported" and c["value_kind"] == "scalar" for c in components):
            return [{**fact, "value_kind": "vector", "value_components": [c["value"] for c in components], "issues": fact["issues"] + [i for c in components for i in c["issues"]]}]
    return [fact]


def normalize_table(table_id: str, cells: list[dict], rows: int, columns: int, source: dict, parser_fingerprint: str | dict, raw_table_sha256: str, *, product_family: str | None = None, issues: list[dict] | None = None) -> dict:
    table = {"table_id": table_id, "source": source, "product_family": product_family,
             "parser_fingerprint": parser_fingerprint, "normalizer_version": NORMALIZER_VERSION,
             "raw_table_sha256": raw_table_sha256, "facts": [], "issues": list(issues or [])}
    cells = [{**cell, "cell_id": cell.get("cell_id", f"cell-{index:04d}")} for index, cell in enumerate(cells)]
    grid = grid_from_cells(cells, rows, columns)
    # Store evidence, not review annotations or speculative repaired values.
    table["normalized_table"] = {"row_count": rows, "column_count": columns, "cells": cells, "orientation": None}
    if text(grid[0][0]["text"]).lower() == "pump" and any(re.search(r"\bframe\b", c["text"], re.I) for c in unique_cells(grid[0])):
        table["normalized_table"].update(orientation="model_options_by_frame", configuration_axis="Frame")
        for c, header in enumerate(grid[0][1:], 1):
            match = re.fullmatch(r"(?:Frame\s+)?(\d+)", header["text"], re.I)
            if not match:
                continue
            for r, row in enumerate(grid[1:], 1):
                fact = make_fact(table, match[1], row[0]["text"], row[c], "", {"row": r, "column": c})
                tokens = re.split(r"[\s/]+", row[c]["text"])
                fact.update(value=None, value_kind="model_options", model_options=[v for v in tokens if v and v.lower() not in MISSING])
                if len(tokens) == 1 and tokens[0] and len(tokens[0]) > 4:
                    fact.update(status="ambiguous", issues=fact["issues"] + [issue("concatenated_model_options", "Possible joined model identifiers cannot be treated as a large number or split without source evidence.")])
                table["facts"].append(fact)
        return table
    if columns == 2:
        table["normalized_table"]["orientation"] = "key_value"
        for r, row in enumerate(grid):
            if not row[0]["text"] or row[0]["cell_id"] == row[1]["cell_id"] or (not row[1]["text"] and r == 0):
                continue
            table["facts"].extend(split_quantities(make_fact(table, None, row[0]["text"], row[1], "", {"row": r, "column": 1})))
        return table
    model_rows = [(r, scalar_id(row[0])) for r, row in enumerate(grid) if scalar_id(row[0]) and re.search(r"[A-Za-z]", row[0]["text"]) and sum(numeric(c["text"])["value_kind"] == "scalar" for c in unique_cells(row, 1)) >= 3]
    if len(model_rows) >= 2:
        first = model_rows[0][0]
        table["normalized_table"]["orientation"] = "models_in_rows"
        for r, model in model_rows:
            for c, cell in enumerate(grid[r][1:], 1):
                headers = [row[c] for row in grid[:first]]
                units = [h["text"] for h in headers if unit(h["text"])]
                parameter = join_headers([h for h in headers if not unit(h["text"])])
                blocked = [] if parameter and units else [issue("incomplete_dimension_header", "Parameter or unit is missing from the column header.")]
                table["facts"].extend(split_quantities(make_fact(table, model, parameter or f"column {c}", cell, units[-1] if units else "", {"row": r, "column": c}, blocked)))
        return table
    votes = [(sum(bool(unit(row[c]["text"])) for row in grid), c) for c in range(min(4, columns))]
    votes.sort(reverse=True)
    unit_col = votes[0][1] if votes[0][0] >= 1 else None
    start = unit_col + 1 if unit_col is not None else (2 if columns > 2 and text(grid[0][1]["text"]).lower() == "frame" else 1)
    anchor_row, anchors, notes = anchor_columns(grid, start)
    table["issues"].extend(notes)
    table["normalized_table"].update(orientation="models_in_columns", unit_column=unit_col,
                                      configuration_axis=text(grid[anchor_row][0]["text"]) or "Model",
                                      configuration_anchors=[{k: c[k] for k in ("cell_id", "col_index", "col_span", "configuration", "header_reconstruction") if k in c} for c in anchors])
    if not anchors:
        table["issues"].append(issue("table_retained_for_review", "Raw table remains available; numeric facts were not assigned."))
        return table
    header_rows = set(range(anchor_row))
    # Configuration headers such as frame groups above a Chambers axis are data
    # only when they explicitly span groups; ordinary model titles are excluded.
    current_section = None
    for r, row in enumerate(grid):
        if r in header_rows and not re.search(r"frame\s+size", row[0]["text"], re.I):
            continue
        if r == anchor_row and re.fullmatch(r"(?:series|model|type|pump|frame|size)", row[0]["text"], re.I):
            continue
        parameter = join_headers(unique_cells(row[:unit_col if unit_col is not None else start]))
        if not parameter or any("\n" in c.get("text", "") for c in row[:start]):
            continue
        # Full-width section titles are labels, never values repeated for every
        # model. Qualifier rows inherit that title (Height / 33 mm, for example).
        data_cells = unique_cells(row, start)
        if data_cells and all(c["col_index"] < start for c in data_cells):
            current_section = parameter
            continue
        qualifier = bool(re.fullmatch(r"\d+(?:[.,]\d+)?(?:\s*(?:and|/|,)\s*\d+(?:[.,]\d+)?)*\s*(?:mm|Hz)", parameter, re.I))
        if qualifier and current_section:
            parameter = current_section + " / " + parameter
        elif not qualifier:
            frequency_row = re.fullmatch(r"(.+?)\s+\d+(?:[.,]\d+)?\s*Hz", parameter, re.I)
            current_section = frequency_row[1] if frequency_row else None
        raw_unit = row[unit_col]["text"] if unit_col is not None else ""
        if unit_col is None:
            embedded = re.search(r"\(([^()]*)\)\s*$", parameter)
            if embedded and unit(embedded[1]):
                raw_unit = embedded[1]
        if raw_unit == parameter:
            raw_unit = ""
        for anchor in anchors:
            begin, end = anchor["col_index"], anchor["col_index"] + anchor["col_span"]
            covered = unique_cells(row[begin:end])
            blocked = []
            if len({c["text"] for c in covered}) != 1 or any(c.get("structural_gap") for c in covered):
                blocked.append(issue("configuration_boundary_ambiguity", "A configuration column crosses multiple source cells or a structural gap; no model value selected."))
            if any(n["severity"] == "blocker" for n in notes):
                blocked.append(issue("unresolved_configuration_header", "The configuration label is not independently established."))
            cell = covered[0] if covered else {"text": "", "cell_id": None}
            fact = make_fact(table, anchor["configuration"], parameter, cell, raw_unit,
                             {"row": r, "configuration_columns": [begin, end]}, blocked)
            if len(covered) > 1:
                fact["candidate_cells"] = [{"cell_id": c.get("cell_id"), "text": c["text"], "bbox": c.get("bbox")} for c in covered]
            table["facts"].extend(split_quantities(fact))
    return table
