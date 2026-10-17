"""Offline, content-bound review decisions for privately generated table facts.

Normalization/reconstruction happens before review, using source cell locators
and spans rather than answer constants. Keep raw extracts unchanged and include
their SHA256 in ``normalized_table.raw_table_sha256`` (or at table top level).
The parser fingerprint must record parser/version, model revisions and settings.
Use ``fingerprint_candidate(table)`` when creating a table's cached fingerprint.

Table shape: source={sha,page}, table_id, parser_fingerprint, normalizer_version,
normalized_table, facts, issues. Facts have unique fact_id, factual status
(reported/missing/ambiguous/invalid), and issues. An issue with severity=blocker
prevents verification; table issues may limit their scope with fact_ids.

Ledger: {schema_version:1, decisions:[{table_fingerprint, scope:table|fact,
fact_id (for fact scope), decision:approve|reject|ambiguous, reviewer,
reviewed_at_utc, reason}]}. Decisions cannot edit candidate values. Old review
fingerprints remain unmatched; changed evidence requires a fresh review. Review
flags describe a recorded inspection, not an automatic factual guarantee.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping


REVIEW_FIELDS = {"review_status", "review", "review_blockers", "review_diagnostics"}
FACT_STATUSES = {"reported", "missing", "ambiguous", "invalid"}
REVIEW_STATUSES = ("pending", "verified", "rejected", "ambiguous")


def _json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numbers cannot be fingerprinted")
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("Fingerprint content must be JSON-compatible with string object keys")


def content_hash(value: Any) -> str:
    """Canonical JSON SHA256; dictionary order does not change the digest."""
    _json_value(value)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_code_hash(path: str | Path) -> str:
    """Bind UTF-8 implementation text, independent of checkout line endings.

    This is deliberately narrower than syntax or whitespace normalization:
    comments, settings, literals and executable changes all alter the digest.
    Original PDFs and raw extracted artifacts continue to use byte hashes.
    """
    source = Path(path).read_bytes().decode("utf-8")
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value.lower()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def fingerprint_table(source_sha256: str, page: int, table_id: str,
                      normalized_table: Any, parser_fingerprint: Any,
                      normalizer_version: str) -> str:
    """Bind source, physical page, table content, extraction and normalization."""
    if type(page) is not int or page < 1:
        raise ValueError("page must be a positive physical page number")
    if not isinstance(parser_fingerprint, dict) or not parser_fingerprint:
        raise ValueError("parser_fingerprint must record parser/model/settings metadata")
    return content_hash({
        "review_protocol": 1,
        "source_sha256": _digest(source_sha256, "source_sha256"),
        "page": page, "table_id": _text(table_id, "table_id"),
        "normalized_table": normalized_table,
        "parser_fingerprint": parser_fingerprint,
        "normalizer_version": _text(normalizer_version, "normalizer_version"),
    })


def _unreviewed_fact(fact: Mapping) -> dict:
    return {key: value for key, value in fact.items() if key not in REVIEW_FIELDS}


def fingerprint_candidate(table: Mapping) -> str:
    """Recompute a fingerprint including raw content, facts and issue scopes."""
    source = table["source"]
    normalized = table["normalized_table"]
    raw_digest = table.get("raw_table_sha256")
    nested_digest = normalized.get("raw_table_sha256") if isinstance(normalized, dict) else None
    if raw_digest is not None and nested_digest is not None:
        if _digest(raw_digest, "raw_table_sha256") != _digest(nested_digest, "raw_table_sha256"):
            raise ValueError("Raw table digests disagree")
    raw_digest = _digest(raw_digest or nested_digest, "raw_table_sha256")
    return fingerprint_table(source["sha"], source["page"], table["table_id"], {
        "raw_table_sha256": raw_digest,
        "normalized_table": normalized,
        "facts": [_unreviewed_fact(fact) for fact in table["facts"]],
        "issues": table.get("issues", []),
    }, table["parser_fingerprint"], table["normalizer_version"])


def validate_manifest(manifest: Mapping | None) -> dict:
    """Reject conflicting decisions, malformed hashes and replacement answers."""
    if manifest is None:
        return {"schema_version": 1, "decisions": []}
    if not isinstance(manifest, Mapping) or type(manifest.get("schema_version")) is not int \
            or manifest.get("schema_version") != 1 or not isinstance(manifest.get("decisions"), list):
        raise ValueError("Review ledger requires schema_version=1 and a decisions list")
    result = deepcopy(dict(manifest))
    seen, ids = set(), set()
    for decision in result["decisions"]:
        if not isinstance(decision, dict):
            raise ValueError("Review decision must be an object")
        if any(key in decision for key in ("value", "unit", "replacement", "replacement_value", "facts", "edits", "normalized_table")):
            raise ValueError("Review decisions cannot contain replacement values or edits")
        digest = _digest(decision.get("table_fingerprint"), "table_fingerprint")
        decision["table_fingerprint"] = digest
        scope = decision.get("scope")
        if not isinstance(scope, str) or scope not in {"table", "fact"} \
                or not isinstance(decision.get("decision"), str) \
                or decision["decision"] not in {"approve", "reject", "ambiguous"}:
            raise ValueError("Unknown review scope or decision")
        fact_id = decision.get("fact_id")
        if scope == "fact":
            _text(fact_id, "fact_id")
        elif fact_id is not None:
            raise ValueError("Table-scoped decisions must not name a fact_id")
        key = (digest, scope, fact_id)
        if key in seen:
            raise ValueError("Duplicate review decision scope/fingerprint/fact_id")
        seen.add(key)
        if "decision_id" in decision:
            identifier = _text(decision["decision_id"], "decision_id")
            if identifier in ids:
                raise ValueError("Duplicate decision_id")
            ids.add(identifier)
        _text(decision.get("reviewer"), "reviewer")
        _text(decision.get("reason"), "reason")
        stamp = _text(decision.get("reviewed_at_utc"), "reviewed_at_utc")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewed_at_utc must be an ISO8601 UTC timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("reviewed_at_utc must include UTC timezone")
    return result


def _issues(issues: Any, fact_ids: set[str]) -> list[dict]:
    if not isinstance(issues, list):
        raise ValueError("issues must be a list")
    for issue in issues:
        if not isinstance(issue, dict) or not isinstance(issue.get("severity"), str) \
                or issue["severity"] not in {"info", "warning", "blocker"}:
            raise ValueError("Each issue requires explicit info/warning/blocker severity")
        _text(issue.get("code"), "issue code")
        if "fact_ids" in issue:
            scope = issue["fact_ids"]
            if not isinstance(scope, list) or not scope \
                    or any(not isinstance(item, str) or item not in fact_ids for item in scope):
                raise ValueError("Issue fact_ids must identify existing facts")
    return issues


def apply_reviews(table: Mapping, review_manifest: Mapping | None = None) -> dict:
    """Return a reviewed copy; stale approvals never modify or verify candidates."""
    result = deepcopy(dict(table))
    facts = result.get("facts")
    if not isinstance(facts, list):
        raise ValueError("Table facts must be a list")
    fact_ids: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("Candidate fact must be an object")
        identifier = _text(fact.get("fact_id"), "fact_id")
        if identifier in fact_ids:
            raise ValueError("Duplicate fact_id")
        fact_ids.add(identifier)
        if not isinstance(fact.get("status"), str) or fact["status"] not in FACT_STATUSES:
            raise ValueError("Unknown factual status")
    table_issues = _issues(result.get("issues", []), fact_ids)
    for fact in facts:
        _issues(fact.get("issues", []), fact_ids)
    fingerprint = fingerprint_candidate(result)
    if "table_fingerprint" in result and _digest(result["table_fingerprint"], "table_fingerprint") != fingerprint:
        raise ValueError("Cached table_fingerprint does not match incoming evidence")
    result["table_fingerprint"] = fingerprint
    decisions = validate_manifest(review_manifest)["decisions"]
    matched = [decision for decision in decisions if decision["table_fingerprint"] == fingerprint]
    for decision in matched:
        if decision["scope"] == "fact" and decision["fact_id"] not in fact_ids:
            raise ValueError("Matched review names an unknown fact_id")
    table_decision = next((decision for decision in matched if decision["scope"] == "table"), None)
    individual = {decision["fact_id"]: decision for decision in matched if decision["scope"] == "fact"}
    for fact in facts:
        blockers = [issue for issue in fact.get("issues", []) if issue["severity"] == "blocker"]
        blockers += [issue for issue in table_issues if issue["severity"] == "blocker"
                     and ("fact_ids" not in issue or fact["fact_id"] in issue["fact_ids"])]
        if fact["status"] in {"ambiguous", "invalid"}:
            blockers.append({"code": f"factual_status_{fact['status']}", "severity": "blocker"})
        decision = individual.get(fact["fact_id"], table_decision)
        if table_decision and table_decision["decision"] == "reject":
            decision = table_decision
        elif table_decision and table_decision["decision"] == "ambiguous":
            if not decision or decision["decision"] != "reject":
                decision = table_decision
        action = decision["decision"] if decision else None
        if action == "reject":
            review_status = "rejected"
        elif action == "ambiguous" or fact["status"] == "ambiguous":
            review_status = "ambiguous"
        elif blockers:
            review_status = "pending" if fact["status"] == "invalid" else "ambiguous"
        else:
            review_status = "verified" if action == "approve" else "pending"
        fact["review_status"] = review_status
        fact["review_blockers"] = deepcopy(blockers)
        fact["review"] = deepcopy(decision)
    statuses = [fact["review_status"] for fact in facts]
    if table_decision and table_decision["decision"] == "reject":
        result["review_status"] = "rejected"
    elif table_decision and table_decision["decision"] == "ambiguous" or "ambiguous" in statuses:
        result["review_status"] = "ambiguous"
    elif statuses and all(status == "verified" for status in statuses):
        result["review_status"] = "verified"
    elif statuses and all(status == "rejected" for status in statuses):
        result["review_status"] = "rejected"
    else:
        result["review_status"] = "pending"
    result["review"] = deepcopy(table_decision)
    result["review_diagnostics"] = {"matched_decisions": len(matched), "content_fingerprint_recomputed": True}
    return result


def review_summary(tables: list[Mapping]) -> dict:
    """Metadata-only public counts: no source values, identities or review notes."""
    table_counts = dict.fromkeys(REVIEW_STATUSES, 0)
    fact_counts = dict.fromkeys(REVIEW_STATUSES, 0)
    for table in tables:
        status = table.get("review_status", "pending")
        if status not in table_counts:
            raise ValueError("Unknown table review_status")
        table_counts[status] += 1
        for fact in table.get("facts", []):
            status = fact.get("review_status", "pending")
            if status not in fact_counts:
                raise ValueError("Unknown fact review_status")
            fact_counts[status] += 1
    return {"schema_version": 1, "tables": table_counts, "facts": fact_counts}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validate", "fingerprint", "summary"))
    parser.add_argument("path", type=Path, help="Private JSON ledger, candidate table or reviewed-table list")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if args.operation == "validate":
            ledger = validate_manifest(data)
            print(json.dumps({"schema_version": 1, "valid_decisions": len(ledger["decisions"])}))
        elif args.operation == "fingerprint":
            print(fingerprint_candidate(data))
        else:
            print(json.dumps(review_summary(data), sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Review validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
