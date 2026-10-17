"""Offline SQLite evidence store; retrieval does not recommend or estimate.

Original JSON is authoritative for numeric precision and non-scalar quantities.
REAL columns are auxiliary indexes. FTS is context retrieval, never fact approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from .services import json_numbers

SCHEMA_VERSION = 1
CHUNK_KINDS = {"narrative", "table_context", "figure_caption"}
FACT_STATUSES = {"reported", "missing", "ambiguous", "invalid"}
REVIEW_STATUSES = {"pending", "verified", "rejected", "ambiguous"}
DEFAULT_TEXT_KINDS = ("narrative", "figure_caption")

SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, source_file TEXT NOT NULL,
 source_sha256 TEXT NOT NULL, page_count INTEGER NOT NULL CHECK(page_count>0),
 json TEXT NOT NULL
);
CREATE TABLE chunks (
 id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id),
 page INTEGER NOT NULL CHECK(page>0), text TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('narrative','table_context','figure_caption')),
 source_sha256 TEXT NOT NULL,
 evidence_status TEXT NOT NULL CHECK(evidence_status IN ('source_text','raw_table')),
 citation_json TEXT NOT NULL, json TEXT NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61');
CREATE TABLE tables (
 table_key TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id),
 page INTEGER NOT NULL CHECK(page>0), table_id TEXT NOT NULL,
 source_sha256 TEXT NOT NULL, review_status TEXT NOT NULL,
 json TEXT NOT NULL, UNIQUE(doc_id,page,table_id)
);
CREATE TABLE facts (
 fact_key TEXT PRIMARY KEY, table_key TEXT NOT NULL REFERENCES tables(table_key),
 doc_id TEXT NOT NULL REFERENCES documents(id), page INTEGER NOT NULL,
 table_id TEXT NOT NULL, fact_id TEXT NOT NULL,
 product TEXT, config TEXT, parameter TEXT NOT NULL, value REAL,
 value_kind TEXT NOT NULL, unit TEXT,
 status TEXT NOT NULL CHECK(status IN ('reported','missing','ambiguous','invalid')),
 review_status TEXT NOT NULL CHECK(review_status IN ('pending','verified','rejected','ambiguous')),
 blockers INTEGER NOT NULL CHECK(blockers IN (0,1)), json TEXT NOT NULL,
 UNIQUE(doc_id,page,table_id,fact_id)
);
CREATE INDEX facts_lookup ON facts(product,config,parameter,review_status,status);
CREATE TABLE services (
 report_id TEXT PRIMARY KEY, source_file TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 country TEXT, equipment_family TEXT, on_site_hours REAL, travel_hours REAL,
 reported_total REAL, currency TEXT, json TEXT NOT NULL
);
CREATE INDEX services_lookup ON services(country,equipment_family);
"""


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value.lower()


def _page(value: Any, maximum: int, name: str = "page") -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a physical page within its document")
    return value


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file(value: str) -> str:
    return value.replace("\\", "/")


def _real(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("Scalar values must be numeric or null, never strings/booleans")
    if isinstance(value, Decimal) and not value.is_finite() or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numbers cannot be stored")
    try:
        converted = float(value)
    except OverflowError:
        return None
    # Never index an overflow/underflow as an invented infinity or zero.
    return converted if math.isfinite(converted) and (converted != 0 or value == 0) else None


def _source_matches(source: Mapping, doc: Mapping, page: int) -> None:
    if _sha(source.get("sha", source.get("source_sha256")), "table source SHA256") != doc["source_sha256"].lower():
        raise ValueError("Source hash disagrees with the referenced document")
    if source.get("page") != page:
        raise ValueError("Source physical page disagrees with table page")
    source_file = source.get("file", source.get("source_file"))
    if source_file is not None and _file(source_file) != _file(doc["source_file"]):
        raise ValueError("Source file disagrees with the referenced document")


def _prepared(documents: Iterable[Mapping], chunks: Iterable[Mapping], tables: Iterable[Mapping],
              services: Iterable[Mapping]) -> tuple[list[Mapping], list[Mapping], list[Mapping], list[Mapping]]:
    docs, chunk_rows, table_rows, service_rows = [list(items) for items in (documents, chunks, tables, services)]
    indexed_docs: dict[str, Mapping] = {}
    for doc in docs:
        identifier = _required_text(doc.get("id"), "document id")
        if identifier in indexed_docs:
            raise ValueError("Duplicate document id; documents cannot silently collide")
        _required_text(doc.get("title"), "document title")
        _required_text(doc.get("source_file"), "document source_file")
        _sha(doc.get("source_sha256"), "document source_sha256")
        if type(doc.get("page_count")) is not int or doc["page_count"] < 1:
            raise ValueError("Document page_count must be a positive integer")
        json_numbers(doc)
        indexed_docs[identifier] = doc
    chunk_ids: set[str] = set()
    for chunk in chunk_rows:
        identifier = _required_text(chunk.get("id"), "chunk id")
        if identifier in chunk_ids:
            raise ValueError("Duplicate chunk id, including cross-document collisions")
        chunk_ids.add(identifier)
        doc_id = chunk.get("doc_id")
        doc = indexed_docs.get(doc_id) if isinstance(doc_id, str) else None
        if not doc:
            raise ValueError("Chunk references an unknown document")
        _page(chunk.get("page"), doc["page_count"])
        if _sha(chunk.get("source_sha256"), "chunk source_sha256") != doc["source_sha256"].lower():
            raise ValueError("Chunk source hash disagrees with document")
        if not isinstance(chunk.get("text"), str):
            raise ValueError("Chunk text must be a string")
        kind = chunk.get("kind")
        if kind not in CHUNK_KINDS:
            raise ValueError("Unknown chunk kind")
        expected_status = "raw_table" if kind == "table_context" else "source_text"
        if chunk.get("evidence_status") != expected_status:
            raise ValueError("Raw table context must retain raw_table evidence status")
        if not isinstance(chunk.get("citation"), (str, dict)) or not chunk["citation"]:
            raise ValueError("Chunk requires source citation metadata")
        if isinstance(chunk["citation"], dict):
            cited = chunk["citation"]
            for key, expected in (("doc_id", doc["id"]), ("page", chunk["page"]),
                                  ("source_sha256", doc["source_sha256"]), ("sha", doc["source_sha256"]),
                                  ("source_file", doc["source_file"]), ("file", doc["source_file"]), ("pdf", doc["source_file"])):
                actual = cited.get(key, expected)
                if key in {"source_sha256", "sha"}:
                    actual, expected = _sha(actual, "citation sha"), expected.lower()
                elif key in {"source_file", "file", "pdf"}:
                    actual, expected = _file(actual), _file(expected)
                if actual != expected:
                    raise ValueError("Chunk citation disagrees with its source identity")
        json_numbers(chunk)
    table_keys: set[tuple] = set()
    for table in table_rows:
        doc_id = table.get("doc_id")
        doc = indexed_docs.get(doc_id) if isinstance(doc_id, str) else None
        if not doc:
            raise ValueError("Table references an unknown document")
        source = table.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Table requires source identity")
        page = _page(source.get("page"), doc["page_count"])
        local_id = _required_text(table.get("table_id"), "table_id")
        identity = (doc["id"], page, local_id)
        if identity in table_keys:
            raise ValueError("Duplicate table within the same document/page")
        table_keys.add(identity)
        _source_matches(source, doc, page)
        if table.get("review_status", "pending") not in REVIEW_STATUSES:
            raise ValueError("Unknown table review_status")
        if "table_fingerprint" in table:
            from .review import fingerprint_candidate
            if fingerprint_candidate(table) != table["table_fingerprint"]:
                raise ValueError("Stale content-bound table review fingerprint")
        facts = table.get("facts")
        if not isinstance(facts, list):
            raise ValueError("Table facts must be a list")
        identifiers: set[str] = set()
        for fact in facts:
            identifier = _required_text(fact.get("fact_id"), "fact_id")
            if identifier in identifiers:
                raise ValueError("Duplicate fact_id in a table")
            identifiers.add(identifier)
            _required_text(fact.get("parameter"), "fact parameter")
            if fact.get("status") not in FACT_STATUSES or fact.get("review_status", "pending") not in REVIEW_STATUSES:
                raise ValueError("Unknown fact/review status")
            if fact.get("value_kind", "scalar") == "scalar":
                _real(fact.get("value"))
            source_refs = fact.get("source_refs")
            if not isinstance(source_refs, list) or not source_refs:
                raise ValueError("Fact requires source references")
            for ref in source_refs:
                if not isinstance(ref, dict) or ref.get("page") != page:
                    raise ValueError("Fact source page disagrees with its table")
                if "sha" in ref and _sha(ref["sha"], "fact source sha") != doc["source_sha256"].lower():
                    raise ValueError("Fact source hash disagrees with document")
                if "file" in ref and _file(ref["file"]) != _file(doc["source_file"]):
                    raise ValueError("Fact source file disagrees with document")
                if "table_id" in ref and ref["table_id"] != local_id:
                    raise ValueError("Fact source table id disagrees")
        json_numbers(table)
    report_ids: set[str] = set()
    for service in service_rows:
        identifier = _required_text(service.get("report_id"), "service report_id")
        if identifier in report_ids:
            raise ValueError("Duplicate service report_id")
        report_ids.add(identifier)
        source = service.get("source", {})
        _required_text(source.get("path"), "service source path")
        _sha(source.get("sha256"), "service source sha256")
        if service.get("typed_validation_errors"):
            raise ValueError("Service record failed typed validation")
        if "line_count" in source:
            from .services import validate_record
            if validate_record(dict(service)):
                raise ValueError("Service numeric/provenance fields failed validation")
        for key in ("on_site", "travel"):
            _real(service.get("hours", {}).get(key, {}).get("value"))
        total = service.get("cost", {}).get("reported_total")
        if total:
            _real(total.get("amount"))
        json_numbers(service)
    return docs, chunk_rows, table_rows, service_rows


def build_store(output_path: str | Path, documents: Iterable[Mapping], chunks: Iterable[Mapping],
                tables: Iterable[Mapping], services: Iterable[Mapping]) -> dict[str, int]:
    """Build a fresh store, rejecting a nonempty existing target before any write."""
    path = Path(output_path)
    if path.is_symlink() or path.exists() and (not path.is_file() or path.stat().st_size):
        raise FileExistsError(f"Refusing to overwrite existing store: {path}")
    docs, chunks, tables, services = _prepared(documents, chunks, tables, services)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("BEGIN;\n" + SCHEMA)
        connection.execute("INSERT INTO metadata VALUES (?,?)", ("schema_version", str(SCHEMA_VERSION)))
        connection.execute("INSERT INTO metadata VALUES (?,?)", ("numeric_authority", "original JSON; REAL columns are auxiliary indexes"))
        for doc in docs:
            connection.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", (
                doc["id"], doc["title"], doc["source_file"], doc["source_sha256"].lower(), doc["page_count"], json_numbers(doc)))
        for chunk in chunks:
            connection.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", (
                chunk["id"], chunk["doc_id"], chunk["page"], chunk["text"], chunk["kind"], chunk["source_sha256"].lower(),
                chunk["evidence_status"], json_numbers(chunk["citation"]), json_numbers(chunk)))
            connection.execute("INSERT INTO chunks_fts(chunk_id,text) VALUES (?,?)", (chunk["id"], chunk["text"]))
        for table in tables:
            page, local_id = table["source"]["page"], table["table_id"]
            table_key = _key(table["doc_id"], page, local_id)
            connection.execute("INSERT INTO tables VALUES (?,?,?,?,?,?,?)", (
                table_key, table["doc_id"], page, local_id, table["source"]["sha"].lower(),
                table.get("review_status", "pending"), json_numbers(table)))
            for fact in table["facts"]:
                value_kind = fact.get("value_kind", "scalar")
                value = _real(fact.get("value")) if value_kind == "scalar" else None
                blockers = int(any(item.get("severity") == "blocker" for item in fact.get("review_blockers", [])))
                blockers |= int(any(item.get("severity") == "blocker" for item in fact.get("issues", [])))
                blockers |= int(any(item.get("severity") == "blocker" and ("fact_ids" not in item or fact["fact_id"] in item["fact_ids"]) for item in table.get("issues", [])))
                connection.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    _key(table["doc_id"], page, local_id, fact["fact_id"]), table_key, table["doc_id"], page, local_id,
                    fact["fact_id"], fact.get("product_family", table.get("product_family")), fact.get("configuration"),
                    fact["parameter"], value, value_kind, fact.get("unit"), fact["status"],
                    fact.get("review_status", "pending"), blockers, json_numbers(fact)))
        for service in services:
            total = service.get("cost", {}).get("reported_total") or {}
            connection.execute("INSERT INTO services VALUES (?,?,?,?,?,?,?,?,?,?)", (
                service["report_id"], service["source"]["path"], service["source"]["sha256"].lower(), service.get("country"),
                service.get("equipment", {}).get("family"), _real(service.get("hours", {}).get("on_site", {}).get("value")),
                _real(service.get("hours", {}).get("travel", {}).get("value")), _real(total.get("amount")),
                total.get("currency"), json_numbers(service)))
        connection.commit()
        return _counts(connection)
    except Exception:
        connection.rollback()
        connection.close()
        path.unlink(missing_ok=True)  # This fresh/empty target contains no prior data.
        raise
    finally:
        connection.close()


def _counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts = {name: connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
              for name in ("documents", "chunks", "tables", "facts", "services")}
    counts["fts_chunks"] = connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
    counts["verified_facts"] = connection.execute("SELECT count(*) FROM facts WHERE review_status='verified' AND status NOT IN ('ambiguous','invalid') AND blockers=0").fetchone()[0]
    counts["raw_table_context_chunks"] = connection.execute("SELECT count(*) FROM chunks WHERE kind='table_context'").fetchone()[0]
    return counts


def quoted_fts_query(query: str) -> str | None:
    """Literal OR terms for recall; user syntax/operators never reach MATCH."""
    if not isinstance(query, str):
        raise TypeError("Query must be text")
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+(?:-[^\W_]+)*", query.casefold(), re.UNICODE)))[:32]
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms) or None


def _limit(value: int, maximum: int = 1000) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


class KnowledgeStore:
    """A read-only retrieval handle. Opening a missing path never creates a DB."""
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")
        try:
            row = self.connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if row is None or row[0] != str(SCHEMA_VERSION):
                raise ValueError("Unsupported evidence store schema version")
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def search_text(self, query: str, limit: int = 5, kinds: tuple[str, ...] = DEFAULT_TEXT_KINDS) -> list[dict]:
        _limit(limit)
        if not kinds or any(kind not in CHUNK_KINDS for kind in kinds):
            raise ValueError("Search kinds must be known chunk kinds")
        expression = quoted_fts_query(query)
        if expression is None:
            return []
        placeholders = ",".join("?" for _ in kinds)
        rows = self.connection.execute(f"""SELECT c.json, c.evidence_status, bm25(chunks_fts) AS score
            FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.chunk_id
            WHERE chunks_fts MATCH ? AND c.kind IN ({placeholders})
            AND ((c.kind='table_context' AND c.evidence_status='raw_table')
              OR (c.kind IN ('narrative','figure_caption') AND c.evidence_status='source_text'))
            ORDER BY score,c.id LIMIT ?""", (expression, *kinds, limit)).fetchall()
        return [{**json.loads(row["json"]), "score": row["score"],
                 "authority": "raw_context_only" if row["evidence_status"] == "raw_table" else "source_text_context"} for row in rows]

    def query_facts(self, product_family: str | None = None, configuration: str | None = None,
                    parameter: str | None = None, verified_only: bool = True, limit: int = 100) -> list[dict]:
        _limit(limit)
        if type(verified_only) is not bool:
            raise TypeError("verified_only must be boolean")
        conditions, values = [], []
        for name, value in (("product", product_family), ("config", configuration), ("parameter", parameter)):
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError("Fact filters must be text")
                conditions.append(f"{name}=?"); values.append(value)
        if verified_only:
            conditions.extend(["review_status='verified'", "status NOT IN ('ambiguous','invalid')", "blockers=0"])
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self.connection.execute("SELECT * FROM facts" + where + " ORDER BY doc_id,page,table_id,fact_id LIMIT ?", (*values, limit)).fetchall()
        return [{**json.loads(row["json"], parse_float=Decimal), "store_ref": {
            "doc_id": row["doc_id"], "page": row["page"], "table_id": row["table_id"], "table_key": row["table_key"], "fact_key": row["fact_key"]}} for row in rows]

    def query_services(self, report_id: str | None = None, limit: int = 100,
                       country: str | None = None, equipment_family: str | None = None) -> list[dict]:
        _limit(limit)
        conditions, values = [], []
        for name, value in (("report_id", report_id), ("country", country), ("equipment_family", equipment_family)):
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError("Service filters must be text")
                conditions.append(f"{name}=?"); values.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self.connection.execute("SELECT json FROM services" + where + " ORDER BY report_id LIMIT ?", (*values, limit)).fetchall()
        return [json.loads(row["json"], parse_float=Decimal) for row in rows]


def search_text(path: str | Path, query: str, limit: int = 5, kinds: tuple[str, ...] = DEFAULT_TEXT_KINDS) -> list[dict]:
    with KnowledgeStore(path) as store:
        return store.search_text(query, limit, kinds)


def query_facts(path: str | Path, **kwargs: Any) -> list[dict]:
    with KnowledgeStore(path) as store:
        return store.query_facts(**kwargs)


def query_services(path: str | Path, **kwargs: Any) -> list[dict]:
    with KnowledgeStore(path) as store:
        return store.query_services(**kwargs)


def _payload_consistency(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Check that indexes still describe their authoritative stored JSON."""
    errors: list[dict[str, Any]] = []
    for table_name in ("documents", "chunks", "tables", "facts", "services"):
        for index, row in enumerate(connection.execute(f"SELECT * FROM {table_name}")):
            try:
                data = json.loads(row["json"], parse_float=Decimal,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
                pairs = []
                if table_name == "documents":
                    pairs = [(row[k], data[k]) for k in ("id", "title", "source_file", "page_count")]
                    pairs.append((row["source_sha256"], _sha(data["source_sha256"], "document sha")))
                elif table_name == "chunks":
                    pairs = [(row[k], data[k]) for k in ("id", "doc_id", "page", "text", "kind", "evidence_status")]
                    pairs.extend([(row["source_sha256"], _sha(data["source_sha256"], "chunk sha")),
                                  (json.loads(row["citation_json"], parse_float=Decimal), data["citation"])])
                elif table_name == "tables":
                    pairs = [(row[k], data[k]) for k in ("doc_id", "table_id")]
                    pairs.extend([(row["page"], data["source"]["page"]),
                                  (row["source_sha256"], _sha(data["source"]["sha"], "table sha")),
                                  (row["review_status"], data.get("review_status", "pending"))])
                    actual_facts = {r["fact_id"]: json.loads(r["json"], parse_float=Decimal) for r in connection.execute("SELECT fact_id,json FROM facts WHERE table_key=?", (row["table_key"],))}
                    pairs.append((actual_facts, {f["fact_id"]: f for f in data["facts"]}))
                    if "table_fingerprint" in data:
                        # The review hashing protocol uses JSON's original numeric
                        # types; do not replace float spelling with Decimal here.
                        from .review import fingerprint_candidate
                        original = json.loads(row["json"])
                        pairs.append((original["table_fingerprint"], fingerprint_candidate(original)))
                elif table_name == "facts":
                    pairs = [(row[k], data[k]) for k in ("fact_id", "parameter", "status")]
                    pairs.extend([(row["config"], data.get("configuration")), (row["unit"], data.get("unit")),
                                  (row["review_status"], data.get("review_status", "pending")),
                                  (row["value_kind"], data.get("value_kind", "scalar")),
                                  (row["value"], _real(data.get("value")) if data.get("value_kind", "scalar") == "scalar" else None)])
                else:
                    total = data.get("cost", {}).get("reported_total") or {}
                    pairs = [(row["report_id"], data["report_id"]), (row["source_file"], data["source"]["path"]),
                             (row["source_sha256"], _sha(data["source"]["sha256"], "service sha")),
                             (row["country"], data.get("country")), (row["equipment_family"], data.get("equipment", {}).get("family")),
                             (row["on_site_hours"], _real(data.get("hours", {}).get("on_site", {}).get("value"))),
                             (row["travel_hours"], _real(data.get("hours", {}).get("travel", {}).get("value"))),
                             (row["reported_total"], _real(total.get("amount"))), (row["currency"], total.get("currency"))]
                if any(actual != expected for actual, expected in pairs):
                    errors.append({"table": table_name, "row": index, "code": "payload_index_mismatch"})
            except (ValueError, TypeError, KeyError, AttributeError):
                errors.append({"table": table_name, "row": index, "code": "invalid_payload"})
    return errors


def validate_store(path: str | Path) -> dict[str, Any]:
    with KnowledgeStore(path) as store:
        connection = store.connection
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        foreign_key_errors = [list(row) for row in connection.execute("PRAGMA foreign_key_check")]
        fts_missing = connection.execute("SELECT count(*) FROM chunks c LEFT JOIN chunks_fts f ON c.id=f.chunk_id WHERE f.chunk_id IS NULL OR c.text!=f.text").fetchone()[0]
        fts_orphans = connection.execute("SELECT count(*) FROM chunks_fts f LEFT JOIN chunks c ON c.id=f.chunk_id WHERE c.id IS NULL").fetchone()[0]
        fts_duplicates = connection.execute("SELECT count(*) FROM (SELECT chunk_id FROM chunks_fts GROUP BY chunk_id HAVING count(*)>1)").fetchone()[0]
        authority_errors = connection.execute("SELECT count(*) FROM chunks WHERE (kind='table_context' AND evidence_status!='raw_table') OR (kind!='table_context' AND evidence_status!='source_text')").fetchone()[0]
        source_errors = connection.execute("""SELECT
            (SELECT count(*) FROM chunks c JOIN documents d ON c.doc_id=d.id WHERE c.source_sha256!=d.source_sha256 OR c.page>d.page_count) +
            (SELECT count(*) FROM tables t JOIN documents d ON t.doc_id=d.id WHERE t.source_sha256!=d.source_sha256 OR t.page>d.page_count) +
            (SELECT count(*) FROM facts f JOIN tables t ON f.table_key=t.table_key WHERE f.doc_id!=t.doc_id OR f.page!=t.page OR f.table_id!=t.table_id)
            """).fetchone()[0]
        json_errors = []
        for table_name in ("documents", "chunks", "tables", "facts", "services"):
            for index, row in enumerate(connection.execute(f"SELECT json FROM {table_name}")):
                try:
                    json.loads(row[0], parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
                except (ValueError, TypeError):
                    json_errors.append({"table": table_name, "row": index})
        review_counts = {row[0]: row[1] for row in connection.execute("SELECT review_status,count(*) FROM facts GROUP BY review_status")}
        counts = _counts(connection)
        payload_errors = _payload_consistency(connection)
        return {"schema_version": SCHEMA_VERSION, "valid": integrity == ["ok"] and not foreign_key_errors and not (fts_missing or fts_orphans or fts_duplicates or source_errors or json_errors or authority_errors or payload_errors),
            "integrity": integrity, "foreign_key_errors": foreign_key_errors, "counts": counts,
            "fts_missing_or_changed": fts_missing, "fts_orphans": fts_orphans, "fts_duplicate_ids": fts_duplicates,
            "source_identity_errors": source_errors, "json_errors": json_errors, "fact_review_status_counts": review_counts,
            "chunk_authority_errors": authority_errors,
            "payload_index_errors": payload_errors,
            "numeric_authority": "original JSON; REAL columns are auxiliary indexes"}
