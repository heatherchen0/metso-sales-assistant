"""Optional local dense retrieval over complete brochure narrative.

This module builds/searches explicit private artifacts. It does not pull models,
approve facts, change the keyword baseline, or infer numerical specifications.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import OllamaClient
from .store import KnowledgeStore

SCHEMA_VERSION = 1
METHOD_VERSION = "complete-narrative-local-dense-v2"
DEFAULT_MODEL = "qwen3-embedding:0.6b"
QUERY_INSTRUCTION = "Given a mining product question, retrieve brochure passages that answer it."
MAX_DOCUMENT_CHARACTERS = 2000
DOCUMENT_FORMAT = "product-section-page-complete-narrative-v2"
# Bounded brochure inputs stay well below this context in the local pilot. A
# smaller embedding context avoids retaining unnecessary GPU allocation beside
# the larger router/vision model; truncate:false still exposes oversized input.
EMBED_OPTIONS = {"num_ctx": 2048}


def _qualitative_filter():
    # Delayed import permits product_tool to import this optional module without
    # a cycle, while using exactly its current evidence filter at call time.
    from .product_tool import qualitative_excerpt
    return qualitative_excerpt


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _store_hash(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Checkpoint the store before embedding; an active WAL cannot bind one immutable database file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _client(client: Any | None) -> Any:
    if client is None:
        client = OllamaClient(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), 120)
    # Reuse the existing strict loopback URL check even for an injected client.
    OllamaClient(getattr(client, "endpoint", ""), getattr(client, "timeout", 120))
    if not callable(getattr(client, "json", None)):
        raise ValueError("Embedding client must provide json(route, payload=None)")
    return client


def _model_binding(client: Any, modeltag: object) -> dict:
    if (not isinstance(modeltag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", modeltag)
            or "cloud" in modeltag.casefold()):
        raise ValueError("Use an explicitly local embedding model tag; cloud tags are not allowed")
    canonical_tag = modeltag if ":" in modeltag else modeltag + ":latest"
    models = client.json("/api/tags").get("models")
    if not isinstance(models, list):
        raise ValueError("Local model inventory is invalid")
    matches = [m for m in models if isinstance(m, dict) and canonical_tag in {m.get("name"), m.get("model")}]
    if len(matches) != 1:
        raise ValueError(f"Embedding model {canonical_tag!r} must already be installed locally; no automatic pull is performed")
    model = matches[0]
    digest = model.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", digest):
        raise ValueError("Installed model has no complete digest")
    shown = client.json("/api/show", {"model": canonical_tag})
    for metadata in (model, shown):
        if metadata.get("remote_model") or metadata.get("remote_host"):
            raise ValueError("Remote-backed Ollama models are not allowed")
    capabilities = shown.get("capabilities")
    if not isinstance(capabilities, list) or "embedding" not in capabilities:
        raise ValueError("Installed model does not report the embedding capability")
    return {"tag": canonical_tag, "digest": digest.lower().removeprefix("sha256:"),
            "size_bytes": model.get("size"), "details": model.get("details", {}),
            "capabilities": capabilities, "model_info_sha256": _hash(shown.get("model_info", {}))}


def _configuration() -> dict:
    return {"method_version": METHOD_VERSION, "document_format": DOCUMENT_FORMAT,
            "max_document_characters": MAX_DOCUMENT_CHARACTERS, "query_instruction": QUERY_INSTRUCTION,
            "query_format": "Instruct: {instruction}\nQuery: {query}", "truncate": False,
            "options": EMBED_OPTIONS, "dimensions_requested": None,
            "qualitative_filter_sha256": _text_hash(inspect.getsource(_qualitative_filter()))}


def _corpus(store_path: str | Path) -> tuple[list[dict], str]:
    records = []
    qualitative_excerpt = _qualitative_filter()
    with KnowledgeStore(store_path) as store:
        documents = {r["id"]: json.loads(r["json"]) for r in store.connection.execute("SELECT id,json FROM documents")}
        rows = store.connection.execute("""SELECT * FROM chunks
            WHERE kind IN ('narrative','figure_caption') AND evidence_status='source_text'
            ORDER BY id""")
        for row in rows:
            chunk = json.loads(row["json"])
            for field in ("id", "doc_id", "page", "text", "kind", "source_sha256", "evidence_status"):
                if chunk.get(field) != row[field]:
                    raise ValueError("Store chunk JSON disagrees with its indexed source identity")
            document = documents.get(chunk["doc_id"])
            if (document is None or chunk["source_sha256"] != document.get("source_sha256")
                    or type(chunk["page"]) is not int or not 1 <= chunk["page"] <= document.get("page_count", 0)):
                raise ValueError("Store chunk source/hash/page disagrees with its brochure")
            excerpt, omitted_blocks = qualitative_excerpt(chunk)
            if not excerpt.strip():
                continue
            prefix = (f"Product: {str(document.get('title', ''))[:200]}\n"
                      f"Section: {str(chunk.get('section') or '')[:200]}\n"
                      f"Page: {chunk['page']}\nPassage:\n")
            original_input = prefix + excerpt
            embedded_text = original_input[:MAX_DOCUMENT_CHARACTERS]
            records.append({"id": chunk["id"], "chunk": chunk,
                            "source_chunk_sha256": _hash({"chunk": chunk, "document": document}),
                            "qualitative_excerpt_sha256": _text_hash(excerpt),
                            "embedding_input_sha256": _text_hash(embedded_text),
                            "embedded_text": embedded_text, "full_input_characters": len(original_input),
                            "omitted_input_characters": max(0, len(original_input) - len(embedded_text)),
                            "omitted_numeric_blocks": omitted_blocks,
                            "numeric_assertion_authority": False})
    fingerprint = _hash([{k: v for k, v in record.items() if k != "chunk"} for record in records])
    return records, fingerprint


def _unit_vector(vector: Any, dimensions: int | None = None) -> list[float]:
    if not isinstance(vector, list) or not vector or (dimensions is not None and len(vector) != dimensions):
        raise ValueError("Embedding vector dimensions are empty or inconsistent")
    values = []
    for item in vector:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("Embedding vector must contain finite numbers")
        try:
            number = float(item)
        except (ValueError, OverflowError):
            raise ValueError("Embedding vector must contain finite numbers") from None
        if not math.isfinite(number):
            raise ValueError("Embedding vector must contain finite numbers")
        values.append(number)
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding vector has an invalid or zero norm")
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = _unit_vector(left)
    b = _unit_vector(right, len(a))
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a, b))))


def _embed(client: Any, model: str, texts: list[str], dimensions: int | None = None) -> tuple[list[list[float]], dict]:
    started = time.perf_counter()
    response = client.json("/api/embed", {"model": model, "input": texts, "truncate": False,
                                         "options": dict(EMBED_OPTIONS), "keep_alive": "5m"})
    if response.get("model") != model:
        raise ValueError("Embedding response model does not match the requested local tag")
    vectors = response.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("Embedding response must return exactly one vector per input")
    validated = []
    for vector in vectors:
        unit = _unit_vector(vector, dimensions)
        dimensions = len(unit)
        validated.append(unit)
    metadata = {"inputs": len(texts), "wall_seconds": time.perf_counter() - started,
                **{name: response.get(name) for name in ("model", "total_duration", "load_duration", "prompt_eval_count")}}
    return validated, metadata


def build_embedding_index(store_path: str | Path, output: str | Path, client: Any | None = None,
                          modeltag: str = DEFAULT_MODEL, batch_size: int = 16) -> dict:
    """Build a fresh, source-bound private index; never overwrite or pull weights."""
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an embedding artifact: {output}")
    started = time.perf_counter()
    store_sha = _store_hash(store_path)
    records, corpus_sha = _corpus(store_path)
    if not records:
        raise ValueError("There are no eligible qualitative brochure passages to embed")
    client = _client(client)
    model = _model_binding(client, modeltag)
    entries, timings, dimensions = [], [], None
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        vectors, timing = _embed(client, model["tag"], [r["embedded_text"] for r in batch], dimensions)
        dimensions = len(vectors[0])
        timings.append({"offset": offset, **timing})
        entries.extend({**{k: v for k, v in record.items() if k != "chunk"}, "vector": vector}
                       for record, vector in zip(batch, vectors))
    if _store_hash(store_path) != store_sha or _corpus(store_path)[1] != corpus_sha:
        raise ValueError("Evidence store changed during the build; no index was written")
    if _model_binding(client, model["tag"]) != model:
        raise ValueError("Embedding model changed during the build; no index was written")
    configuration = _configuration()
    artifact = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "store_sha256": store_sha, "corpus_sha256": corpus_sha, "model": model,
                "configuration": configuration, "configuration_sha256": _hash(configuration),
                "dimensions": dimensions, "entries": entries, "batch_size": batch_size,
                "timings": timings, "build_wall_seconds": time.perf_counter() - started}
    artifact["content_sha256"] = _hash(artifact)
    encoded = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return {"index_path": str(output), "chunks": len(entries), "dimensions": dimensions,
            "model": model, "store_sha256": store_sha, "corpus_sha256": corpus_sha,
            "configuration_sha256": artifact["configuration_sha256"], "content_sha256": artifact["content_sha256"],
            "truncated_chunks": sum(e["omitted_input_characters"] > 0 for e in entries),
            "build_wall_seconds": artifact["build_wall_seconds"], "batch_size": batch_size}


def _validated_index(store_path: str | Path, index_path: str | Path, client: Any) -> tuple[dict, list[dict]]:
    artifact = json.loads(Path(index_path).read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite index value: {value}")))
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported embedding index schema")
    if _hash({k: v for k, v in artifact.items() if k != "content_sha256"}) != artifact.get("content_sha256"):
        raise ValueError("Embedding index content checksum disagrees")
    if _store_hash(store_path) != artifact.get("store_sha256"):
        raise ValueError("Embedding index is stale: evidence store bytes changed")
    configuration = _configuration()
    if artifact.get("configuration") != configuration or artifact.get("configuration_sha256") != _hash(configuration):
        raise ValueError("Embedding index is stale: filter, document format or query settings changed")
    records, corpus_sha = _corpus(store_path)
    if artifact.get("corpus_sha256") != corpus_sha:
        raise ValueError("Embedding index is stale: qualitative corpus changed")
    entries = artifact.get("entries")
    dimensions = artifact.get("dimensions")
    if not isinstance(entries, list) or type(dimensions) is not int or dimensions < 1 or len(entries) != len(records):
        raise ValueError("Embedding index shape or chunk count is invalid")
    for entry, record in zip(entries, records):
        if not isinstance(entry, dict) or {k: v for k, v in entry.items() if k != "vector"} != {k: v for k, v in record.items() if k != "chunk"}:
            raise ValueError("Embedding source chunk identity/text hash does not match the current corpus")
        _unit_vector(entry.get("vector"), dimensions)
    model = artifact.get("model")
    if not isinstance(model, dict) or _model_binding(client, model.get("tag")) != model:
        raise ValueError("Embedding index is stale: local model digest or metadata changed")
    return artifact, records


def check_embedding_index(store_path: str | Path, index_path: str | Path, client: Any | None = None) -> dict:
    """Check source and installed-model bindings without embedding inference."""
    artifact, records = _validated_index(store_path, index_path, _client(client))
    return {"valid": True, "chunks": len(records), "dimensions": artifact["dimensions"],
            "model": artifact["model"], "store_sha256": artifact["store_sha256"],
            "corpus_sha256": artifact["corpus_sha256"], "content_sha256": artifact["content_sha256"]}


def semantic_search(store_path: str | Path, index_path: str | Path, query: str,
                    limit: int = 60, client: Any | None = None) -> list[dict]:
    """Rank current source chunks; their rank confers no factual approval."""
    if not isinstance(query, str) or not query.strip() or len(query) > 8000:
        raise ValueError("Semantic query must be nonempty text of at most 8000 characters")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    client = _client(client)
    artifact, records = _validated_index(store_path, index_path, client)
    query_text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
    vectors, _ = _embed(client, artifact["model"]["tag"], [query_text], artifact["dimensions"])
    # Validate both bindings again after inference, so a changed model/database
    # cannot combine a fresh query vector with the earlier source snapshot.
    if _store_hash(store_path) != artifact["store_sha256"] or _model_binding(client, artifact["model"]["tag"]) != artifact["model"]:
        raise ValueError("Evidence store or model changed during semantic search")
    query_vector = vectors[0]
    qualitative_excerpt = _qualitative_filter()
    ranked = []
    for record, entry in zip(records, artifact["entries"]):
        score = max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(query_vector, _unit_vector(entry["vector"], artifact["dimensions"])))))
        ranked.append({**record["chunk"], "semantic_score": score,
                       "semantic_excerpt": qualitative_excerpt(record["chunk"])[0],
                       "semantic_truncated": bool(entry["omitted_input_characters"])})
    return sorted(ranked, key=lambda row: (-row["semantic_score"], row["id"]))[:limit]


def fuse_rankings(keyword_ids: list[str], dense_ids: list[str], k: int = 60) -> list[dict]:
    """Reciprocal rank fusion; duplicate IDs cannot add weight to one ranking."""
    if type(k) is not int or k < 0:
        raise ValueError("RRF k must be a nonnegative integer")
    rankings = []
    for identifiers in (keyword_ids, dense_ids):
        if not isinstance(identifiers, list) or any(not isinstance(i, str) or not i for i in identifiers):
            raise ValueError("Rankings must be lists of nonempty chunk IDs")
        rankings.append({identifier: rank for rank, identifier in enumerate(dict.fromkeys(identifiers), start=1)})
    keyword, dense = rankings
    rows = [{"id": identifier, "rrf_score": (1 / (k + keyword[identifier]) if identifier in keyword else 0)
             + (1 / (k + dense[identifier]) if identifier in dense else 0),
             "keyword_rank": keyword.get(identifier), "dense_rank": dense.get(identifier)}
            for identifier in keyword.keys() | dense.keys()]
    return sorted(rows, key=lambda row: (-row["rrf_score"], row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "search"))
    parser.add_argument("--store", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()
    client = OllamaClient(args.endpoint, args.timeout)
    if args.action == "build":
        result = build_embedding_index(args.store, args.index, client, args.model, args.batch_size)
    elif args.action == "check":
        result = check_embedding_index(args.store, args.index, client)
    else:
        result = semantic_search(args.store, args.index, args.query, args.limit, client)
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
