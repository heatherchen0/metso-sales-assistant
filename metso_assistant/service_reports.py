"""Bounded report discovery before inspecting individual numerical observations.

Catalogue relevance is calculated across each report's labels, so one heavily
priced report cannot consume every candidate slot. This tool registers no facts
and grants no numerical authority; callers must inspect selected reports.
"""
from contextlib import closing
import json
import re
import sqlite3
import unicodedata

from .service_evidence import _tokens


MAX_CANDIDATES = 8
MAX_MATCHING_LABELS = 4
_QUERY_WORDS = set(_tokens("compare comparison versus vs between report reports country countries historical history find show list please"))
_BOUND = "This is an incomplete, bounded report-candidate list, not an exhaustive search or a statement that other evidence is absent. Inspect selected reports before using prices, quantities or durations."


def _normal(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _named_countries(query, catalogue):
    matches = []
    countries = sorted({item["scope"].get("country") for item in catalogue
                        if isinstance(item["scope"].get("country"), str) and item["scope"]["country"].strip()},
                       key=lambda value: (-len(value), value.casefold(), value))
    text = _normal(query)
    for country in countries:
        pattern = r"(?<!\w)" + re.escape(_normal(country)) + r"(?!\w)"
        for match in re.finditer(pattern, text):
            if not any(start <= match.start() and match.end() <= end for start, end, _ in matches):
                matches.append((match.start(), match.end(), country))
    result = []
    for _, _, country in sorted(matches):
        if country not in result:
            result.append(country)
    return result


def _label(value):
    # Keep the component/task description, not duplicate quantity/rate/amount
    # labels from a single observation group. Inline numeric values are hidden.
    value = re.split(r"\s+[—–]\s+reported\b", str(value), maxsplit=1, flags=re.I)[0]
    value = re.sub(r"[-+]?\d+(?:[,.]\d+)*(?:[eE][-+]?\d+)?", "[number]", value)
    return " ".join(value.split())[:350]


def _read_catalogue(workspace):
    with closing(workspace._connect()) as connection:
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if (json.loads(metadata.get("binding", "{}")) != workspace._binding
                or workspace._content_hash(connection) != metadata.get("content_sha256")):
            raise ValueError("Service catalogue sidecar content or source binding changed")
        catalogue = []
        indexed = {}
        for row in connection.execute("SELECT report_id,json FROM reports ORDER BY report_id"):
            report = json.loads(row["json"])
            if row["report_id"] != report.get("report_id") or not isinstance(report.get("scope"), dict):
                raise ValueError("Service catalogue report identity is invalid")
            entry = {"report_id": row["report_id"], "scope": report["scope"],
                     "source": report.get("source", {}), "labels": []}
            catalogue.append(entry)
            indexed[entry["report_id"]] = entry
        for row in connection.execute("SELECT report_id,json FROM observations ORDER BY report_id,id"):
            fact = json.loads(row["json"])
            if row["report_id"] not in indexed or fact.get("report_id") != row["report_id"]:
                raise ValueError("Service catalogue observation identity is invalid")
            if isinstance(fact.get("label"), str):
                indexed[row["report_id"]]["labels"].append(fact["label"])
        return catalogue, metadata.get("content_sha256")


def find_reports(workspace, query):
    """Discover at most a bounded set of distinct reports, without price facts.

    Countries named using the actual catalogue vocabulary restrict the search.
    A comparison includes a candidate from each named country when a relevant
    candidate exists, before remaining slots are filled by lexical relevance.
    Item/action aliases and individual report IDs are not hardcoded.
    """
    warnings = [_BOUND]
    empty = {"report_candidates": [], "requested_countries": [], "complete": False}
    try:
        if not isinstance(query, str) or not query.strip() or len(query) > 6000:
            raise ValueError("Report discovery requires a bounded nonempty query")
        warnings.extend(workspace._check_binding() or [])
        catalogue, content_before = _read_catalogue(workspace)
        countries = _named_countries(query, catalogue)
        country_terms = {word for country in countries for word in _tokens(country)}
        wanted = set(_tokens(query)) - country_terms - _QUERY_WORDS
        candidates = []
        for item in catalogue:
            scope = item["scope"]
            if countries and scope.get("country") not in countries:
                continue
            label_matches = []
            covered = set()
            for raw_label in item["labels"]:
                overlap = wanted & set(_tokens(raw_label))
                covered.update(overlap)
                if overlap:
                    label_matches.append((len(overlap), _label(raw_label)))
            equipment_overlap = wanted & set(_tokens(scope.get("equipment") or ""))
            if wanted and not covered and not equipment_overlap:
                continue
            matching_labels = []
            for _, label in sorted(label_matches, key=lambda row: (-row[0], row[1])):
                if label and label not in matching_labels:
                    matching_labels.append(label)
                if len(matching_labels) == MAX_MATCHING_LABELS:
                    break
            candidate = {"report_id": item["report_id"], "country": scope.get("country"),
                         "equipment": scope.get("equipment"), "matching_labels": matching_labels,
                         "source_file": item["source"].get("path"),
                         "source_sha256": item["source"].get("sha256")}
            rank = (-len(covered), -max((row[0] for row in label_matches), default=0),
                    -len(equipment_overlap), item["report_id"])
            candidates.append((rank, candidate))
        candidates.sort(key=lambda row: row[0])
        chosen = []
        for country in countries:
            candidate = next((item for _, item in candidates if item["country"] == country), None)
            if candidate is not None and len(chosen) < MAX_CANDIDATES:
                chosen.append(candidate)
        selected = {item["report_id"] for item in chosen}
        for _, candidate in candidates:
            if len(chosen) >= MAX_CANDIDATES:
                break
            if candidate["report_id"] not in selected:
                chosen.append(candidate)
                selected.add(candidate["report_id"])
        warnings.extend(workspace._check_binding() or [])
        _, content_after = _read_catalogue(workspace)
        if content_before != content_after:
            raise ValueError("Service catalogue changed during report discovery")
        missing = [country for country in countries if not any(item["country"] == country for item in chosen)]
        if missing:
            warnings.append("Some requested countries have no candidate in this bounded lexical result; refine the query before concluding their evidence is absent.")
        if not wanted:
            warnings.append("The query supplies catalogue/country terms without a specific component or task; candidates are country-based starting points.")
        return {"facts": [], "sources": [], "warnings": sorted(set(warnings)),
                "data": {"report_candidates": chosen, "requested_countries": countries,
                         "missing_requested_countries": missing, "complete": False,
                         "numerical_authority": "None from report discovery; use read_report for source-bound observations."}}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        return {"facts": [], "sources": [], "warnings": sorted(set([*warnings, str(exc)])),
                "data": empty, "error": str(exc)}
