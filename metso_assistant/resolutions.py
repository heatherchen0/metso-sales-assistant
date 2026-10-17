"""Select preserved disagreement evidence before separate content-bound review.

Rules contain identities and a choice, never replacement answers. This stage
does not approve facts. Apply it to fresh normalized candidates, then compute
the table fingerprint and apply the independent review ledger.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Mapping

from metso_assistant.review import REVIEW_FIELDS, content_hash


COMPARISONS = ("primary_source_comparison", "geometry_crosscheck", "alternate_parser_comparison")
CROSSCHECK_CODES = {"source_geometry_disagreement", "parser_disagreement", "source_crosscheck_disagreement"}
PAYLOAD_FIELDS = ("value", "status", "value_kind", "unit", "raw_cell_text", "raw_unit",
                  "quantity_kind", "operator", "value_range", "value_components", "model_options",
                  "numeric_candidates", "source_refs")
SEMANTIC_FIELDS = ("value", "status", "value_kind", "unit", "quantity_kind", "operator",
                   "value_range", "value_components", "model_options", "numeric_candidates")


def fingerprint_conflict(source: Mapping, candidate: Mapping, alternate: Mapping) -> str:
    """Bind full immutable snapshots and their original source/field identity."""
    return content_hash([{"source": dict(source), "parameter": candidate["parameter"],
                          "configuration": candidate.get("configuration"), "candidate": dict(candidate)},
                         dict(alternate)])


def _digest(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{field} must be a SHA256 digest")
    return value.lower()


def validate_resolution_config(config: Mapping | None) -> dict:
    if config is None:
        return {"schema_version": 1, "rules": []}
    if not isinstance(config, Mapping) or type(config.get("schema_version")) is not int \
            or config["schema_version"] != 1 or not isinstance(config.get("rules"), list):
        raise ValueError("Resolution config requires schema_version=1 and a rules list")
    result = deepcopy(dict(config))
    required = {"source_sha256", "page", "table_id", "fact_id", "conflict_fingerprint",
                "decision", "reviewer", "reviewed_at_utc", "reason"}
    seen, ids = set(), set()
    for rule in result["rules"]:
        if not isinstance(rule, dict) or not required <= rule.keys() or rule.keys() - required - {"rule_id"}:
            raise ValueError("Resolution rules require guarded metadata and cannot contain replacement values")
        for field in ("source_sha256", "conflict_fingerprint"):
            rule[field] = _digest(rule[field], field)
        if type(rule["page"]) is not int or rule["page"] < 1:
            raise ValueError("Resolution page must be a positive physical page number")
        for field in ("table_id", "fact_id", "reviewer", "reviewed_at_utc", "reason"):
            if not isinstance(rule[field], str) or not rule[field].strip():
                raise ValueError(f"Resolution {field} must be a nonempty string")
        if rule["decision"] not in ("candidate", "alternate"):
            raise ValueError("Resolution decision must select candidate or alternate")
        try:
            stamp = datetime.fromisoformat(rule["reviewed_at_utc"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Resolution timestamp must be ISO8601 UTC") from exc
        if stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            raise ValueError("Resolution timestamp must include UTC timezone")
        key = tuple(rule[field] for field in ("source_sha256", "page", "table_id", "fact_id", "conflict_fingerprint"))
        if key in seen:
            raise ValueError("Duplicate resolution target/fingerprint")
        seen.add(key)
        if "rule_id" in rule:
            identifier = rule["rule_id"]
            if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
                raise ValueError("Duplicate or invalid resolution rule_id")
            ids.add(identifier)
    return result


def _payload(fact, fields=PAYLOAD_FIELDS):
    return {field: fact.get(field) for field in fields}


def _source_matches(snapshot, source):
    refs = snapshot.get("source_refs")
    return isinstance(refs, list) and bool(refs) and all(
        isinstance(ref, dict) and ref.get("sha") == source["sha"] and ref.get("page") == source["page"]
        for ref in refs)


def _identity_matches(conflict, current, candidate, alternate):
    if candidate.get("fact_id") != current["fact_id"]:
        return False
    for field in ("configuration", "unit", "quantity_kind", "value_kind", "operator", "product_family"):
        if candidate.get(field) != current.get(field) or alternate.get(field) != candidate.get(field):
            return False
    if candidate.get("parameter") != current.get("parameter"):
        return False
    if "identity" in conflict or "alternate_identity" in conflict:
        return conflict.get("identity") is not None and conflict.get("identity") == conflict.get("alternate_identity")
    canonical = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
    return canonical(candidate.get("parameter")) == canonical(alternate.get("parameter"))


def apply_resolutions(table: Mapping, config: Mapping | None = None) -> dict:
    """Choose only matched snapshots; retain unrelated blockers and raw evidence."""
    result = deepcopy(dict(table))
    rules = validate_resolution_config(config)["rules"]
    facts = result["facts"]
    by_id = {}
    for fact in facts:
        identifier = fact.get("fact_id")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("Resolution candidates require unique nonempty fact_id")
        by_id[identifier] = fact
    source = result["source"]
    _digest(source["sha"], "source sha")
    normalized = result["normalized_table"]
    if normalized.get("applied_resolutions"):
        raise ValueError("Apply resolutions to fresh candidates; regenerate before changing or reapplying choices")
    conflicts = defaultdict(list)
    for comparison in COMPARISONS:
        section = normalized.get(comparison)
        if section is None:
            continue
        if not isinstance(section, dict) or not isinstance(section.get("conflicts", []), list):
            raise ValueError("Comparison conflicts must be a list")
        for conflict in section.get("conflicts", []):
            if not isinstance(conflict, dict):
                raise ValueError("Conflict must preserve candidate and alternate snapshots")
            fingerprint = _digest(conflict.get("conflict_fingerprint"), "conflict_fingerprint")
            conflicts[fingerprint].append((comparison, conflict))
    diagnostics = {"rules_supplied": len(rules), "applied": 0, "unmatched": [], "blocked": []}
    selections = defaultdict(list)
    for rule in rules:
        fingerprint, fact_id = rule["conflict_fingerprint"], rule["fact_id"]
        context = {"fact_id": fact_id, "conflict_fingerprint": fingerprint}
        if (rule["source_sha256"], rule["page"], rule["table_id"]) != (source["sha"], source["page"], result["table_id"]):
            diagnostics["unmatched"].append({**context, "reason": "source_page_or_table_guard_mismatch"})
            continue
        if fact_id not in by_id:
            diagnostics["unmatched"].append({**context, "reason": "absent_fact_target"})
            continue
        targets = [(name, conflict) for name, conflict in conflicts.get(fingerprint, []) if conflict.get("fact_id") == fact_id]
        if not targets:
            diagnostics["unmatched"].append({**context, "reason": "stale_or_absent_conflict_fingerprint"})
            continue
        comparison, conflict = targets[0]
        candidate, alternate = conflict.get("candidate"), conflict.get("alternate")
        reason = None
        if not isinstance(candidate, dict) or not isinstance(alternate, dict):
            reason = "missing_snapshot"
        elif fingerprint_conflict(source, candidate, alternate) != fingerprint:
            reason = "conflict_snapshot_hash_mismatch"
        elif any(fingerprint_conflict(source, other["candidate"], other["alternate"]) != fingerprint for _, other in targets):
            reason = "duplicate_conflict_snapshot_mismatch"
        elif not _identity_matches(conflict, by_id[fact_id], candidate, alternate):
            reason = "semantic_identity_mismatch"
        elif not _source_matches(candidate, source) or not _source_matches(alternate, source):
            reason = "snapshot_source_mismatch"
        elif _payload(by_id[fact_id]) != _payload(candidate):
            reason = "current_candidate_payload_changed"
        elif by_id[fact_id].get("status") in {"ambiguous", "invalid"}:
            reason = "intrinsic_factual_ambiguity"
        selected = conflict.get(rule["decision"])
        if reason is None and selected.get("status") in {"ambiguous", "invalid"}:
            reason = "selected_snapshot_intrinsic_ambiguity"
        if reason is None and rule["decision"] == "alternate" and any(
                issue.get("severity") == "blocker" and issue.get("code") not in CROSSCHECK_CODES
                for issue in selected.get("issues", [])):
            reason = "alternate_intrinsic_blocker"
        corresponding = [issue for issue in by_id[fact_id].get("issues", [])
                         if issue.get("code") in CROSSCHECK_CODES and issue.get("severity") == "blocker"
                         and issue.get("conflict_fingerprint") == fingerprint]
        if reason is None and not corresponding:
            reason = "matching_crosscheck_blocker_absent"
        if reason:
            diagnostics["blocked"].append({**context, "reason": reason})
            continue
        selections[fact_id].append((rule, targets, selected))
    applied = []
    for fact_id in sorted(selections):
        choices = selections[fact_id]
        semantic_payloads = {content_hash(_payload(selected, SEMANTIC_FIELDS)) for _, _, selected in choices}
        if len(semantic_payloads) != 1:
            diagnostics["blocked"].append({"fact_id": fact_id, "reason": "inconsistent_choices_for_one_fact"})
            continue
        # Deterministic evidence selection when several matching comparisons agree.
        choices.sort(key=lambda item: (item[0]["decision"] != "alternate", item[0]["conflict_fingerprint"]))
        authoritative = choices[0][2]
        fact = by_id[fact_id]
        if choices[0][0]["decision"] == "alternate":
            for field in PAYLOAD_FIELDS:
                if field in authoritative:
                    fact[field] = deepcopy(authoritative[field])
                else:
                    fact.pop(field, None)
        resolved_hashes = {rule["conflict_fingerprint"] for rule, _, _ in choices}
        fact["issues"] = [item for item in fact.get("issues", []) if not (
            item.get("code") in CROSSCHECK_CODES and item.get("severity") == "blocker"
            and item.get("conflict_fingerprint") in resolved_hashes)]
        if choices[0][0]["decision"] == "alternate":
            for item in authoritative.get("issues", []):
                if item.get("severity") == "warning" and item not in fact["issues"]:
                    fact["issues"].append(deepcopy(item))
        for rule, targets, selected in choices:
            fact["issues"].append({"code": "resolved_source_crosscheck", "severity": "warning",
                                  "message": "Recorded source inspection selected preserved evidence; content review is still required.",
                                  "conflict_fingerprint": rule["conflict_fingerprint"], "decision": rule["decision"]})
            applied.append({**deepcopy(rule), "comparisons": sorted(name for name, _ in targets),
                            "selected_snapshot_sha256": content_hash(selected),
                            "authoritative_payload_snapshot_sha256": content_hash(authoritative)})
            diagnostics["applied"] += 1
    # A normalization decision is never an approval, including on reused objects.
    for item in [result, *facts]:
        for field in REVIEW_FIELDS:
            item.pop(field, None)
        item["review_status"] = "pending"
    result.pop("table_fingerprint", None)
    normalized["applied_resolutions"] = applied
    result["resolution_diagnostics"] = diagnostics
    return result
