"""Shared source-bound CSV reader and descriptive statistics.

The reader owns timestamp cleaning, missing cells, duplicate hours and source
hashing. No assessment, correlation, operating target or prose answer is defined.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

IRON_COLUMN = "% Iron Concentrate"
SILICA_COLUMN = "% Silica Concentrate"
NUMBER = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?$")


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("quantile requires observations and a probability in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {"n": len(values), "min": min(values), "q25": quantile(values, .25),
            "median": median(values), "q75": quantile(values, .75), "max": max(values)}


def _number(cell: str, percent: bool) -> tuple[float | None, str | None]:
    cell = cell.strip()
    if cell.casefold() in {"", "na", "n/a", "null", "none"}:
        return None, "missing"
    if not NUMBER.fullmatch(cell):
        return None, "invalid"
    value = float(cell.replace(",", "."))
    if not math.isfinite(value) or (percent and not 0 <= value <= 100):
        return None, "invalid"
    return value, None


def _load(csv_path: str | Path) -> tuple[list[dict], dict, dict, list[str]]:
    """Load source afresh. Every statistic is bound to the bytes hashed here."""
    path = Path(csv_path)
    raw = path.read_bytes()
    source: dict[str, Any] = {"id": "P01", "source_id": "P01", "file": path.as_posix(),
              "sha256": hashlib.sha256(raw).hexdigest(), "kind": "hourly_process_csv"}
    text = raw.decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:16384], delimiters=",;\t")
    except csv.Error:
        # An empty/header-only file still receives a useful insufficient-data result.
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect, strict=True)
    original_headers = next(reader, [])
    headers = [" ".join(h.strip().split()) for h in original_headers]
    header_map = {h.casefold(): h for h in headers}
    if not all(headers) or len(header_map) != len(headers):
        raise ValueError("duplicate or empty CSV column names")
    required = ["date", IRON_COLUMN, SILICA_COLUMN]
    absent = [h for h in required if h.casefold() not in header_map]
    if absent:
        raise ValueError("missing CSV columns: " + ", ".join(absent))
    # Preserve exact source names but canonicalize only the three required roles.
    canonical = {header_map[h.casefold()]: h for h in required}
    headers = [canonical.get(h, h) for h in headers]
    numeric_headers = [h for h in headers if h != "date"]
    source["columns"] = original_headers
    issues = {h: {"missing": 0, "invalid": 0, "conflicting_duplicate_hours": 0} for h in numeric_headers}
    groups: dict[datetime, list[dict]] = defaultdict(list)
    malformed = invalid_dates = off_hour = raw_count = 0
    timezone_modes: set[str] = set()
    for physical_row, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        raw_count += 1
        if len(row) != len(headers):
            malformed += 1
            continue
        values = dict(zip(headers, row))
        try:
            stamp = datetime.fromisoformat(values["date"].strip())
        except ValueError:
            invalid_dates += 1
            continue
        timezone_modes.add("aware" if stamp.tzinfo is not None else "unspecified")
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
        if stamp.minute or stamp.second or stamp.microsecond:
            off_hour += 1
            continue
        parsed = {"timestamp": stamp, "source_rows": [physical_row]}
        for h in numeric_headers:
            value, issue = _number(values[h], percent=h.startswith("%"))
            parsed[h] = value
            if issue:
                issues[h][issue] += 1
        groups[stamp].append(parsed)
    if len(timezone_modes) > 1:
        raise ValueError("CSV mixes timezone-aware and unspecified timestamps")
    records = []
    for stamp, duplicates in sorted(groups.items()):
        record = {"timestamp": stamp, "source_rows": [r for d in duplicates for r in d["source_rows"]]}
        for h in numeric_headers:
            # Missing duplicate cells cannot supply conflicting evidence.
            distinct = {d[h] for d in duplicates if d[h] is not None}
            record[h] = next(iter(distinct)) if len(distinct) == 1 else None
            if len(distinct) > 1:
                issues[h]["conflicting_duplicate_hours"] += 1
        records.append(record)
    source["row_count"] = raw_count
    source["date_start"] = records[0]["timestamp"].isoformat(sep=" ") if records else None
    source["date_end"] = records[-1]["timestamp"].isoformat(sep=" ") if records else None
    gaps = []
    for previous, current in zip(records, records[1:]):
        hours = int((current["timestamp"] - previous["timestamp"]).total_seconds() / 3600)
        if hours > 1:
            gaps.append({"after": previous["timestamp"].isoformat(sep=" "),
                         "before": current["timestamp"].isoformat(sep=" "), "missing_hours": hours - 1})
    repeated = 0
    for previous, current in zip(records, records[1:]):
        a = (previous[IRON_COLUMN], previous[SILICA_COLUMN])
        b = (current[IRON_COLUMN], current[SILICA_COLUMN])
        if (current["timestamp"] - previous["timestamp"]).total_seconds() == 3600 and None not in a and a == b:
            repeated += 1
    paired = [(r[IRON_COLUMN], r[SILICA_COLUMN]) for r in records
              if r[IRON_COLUMN] is not None and r[SILICA_COLUMN] is not None]
    profile = {"source_rows": raw_count, "unique_valid_hours": len(records),
               "duplicate_timestamp_rows": sum(len(v) - 1 for v in groups.values()),
               "duplicate_timestamp_groups": sum(len(v) > 1 for v in groups.values()),
               "malformed_rows": malformed, "invalid_dates": invalid_dates, "non_hourly_rows": off_hour,
               "missing_hours": sum(g["missing_hours"] for g in gaps), "gaps": gaps,
               "cell_issues": issues, "complete_quality_hours": len(paired),
               "unique_quality_pairs": len(set(paired)), "adjacent_repeated_quality_pairs": repeated,
               "timestamp_timezone": next(iter(timezone_modes), "unspecified"),
               "delimiter": dialect.delimiter}
    warnings = ["Historical hourly observations are serially dependent; no independent-trial confidence interval is estimated.",
                "This is one plant's historical record; comparability with the customer's plant, sampling method and present operating conditions is unknown.",
                "Customer quality targets, feed conditions, recovery and throughput requirements are not supplied; historical frequency does not establish normal or optimal operation."]
    if profile["missing_hours"]:
        warnings.append(f"There are {profile['missing_hours']} missing hourly timestamps inside the recorded period; gaps are not filled.")
    if repeated:
        warnings.append(f"{repeated} adjacent hours have identical concentrate pairs; the CSV does not establish whether averaging, rounding or interpolation caused them.")
    if profile["duplicate_timestamp_rows"]:
        warnings.append("Duplicate timestamps are counted once; conflicting duplicate cell values are excluded, not averaged.")
    excluded = malformed + invalid_dates + off_hour
    if excluded:
        warnings.append(f"{excluded} malformed, invalid-date or non-hourly rows were excluded; source profile retains the counts.")
    if any(v["missing"] or v["invalid"] or v["conflicting_duplicate_hours"] for v in issues.values()):
        warnings.append("Missing, invalid or conflicting cells are excluded per calculation and exposed in cell_issues; no numeric imputation is performed.")
    if len(paired) < 30:
        warnings.append("Fewer than 30 complete quality hours are available; descriptive results have limited coverage.")
    return records, source, profile, warnings
