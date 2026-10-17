"""Optional local dense candidates for generic service evidence.

Vectors rank observation groups. Every returned identifier still resolves to
the source-bound service sidecar; a similarity score grants no numeric authority.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .model import OllamaClient, unload_local_model
from .semantic import (fuse_rankings, DEFAULT_MODEL, EMBED_OPTIONS, _client, _embed,
                       _model_binding, _unit_vector)
from .service_evidence import ServiceWorkspace


SCHEMA_VERSION = 1
METHOD_VERSION = "generic-service-group-dense-1"
MAX_DOCUMENT_CHARACTERS = 2000
QUERY_INSTRUCTION = "Retrieve historical service evidence for the requested tasks, components and visit context."


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _configuration():
    return {"method_version": METHOD_VERSION, "module_sha256": _file_hash(__file__),
            "embedding_helpers_sha256": _file_hash(Path(__file__).with_name("semantic.py")),
            "document_format": "service-observation-group-label-scope-units-v1",
            "max_document_characters": MAX_DOCUMENT_CHARACTERS,
            "query_instruction": QUERY_INSTRUCTION, "query_format": "Instruct: {instruction}\nQuery: {query}",
            "options": dict(EMBED_OPTIONS), "truncate": False,
            "numerical_authority": "Original sidecar facts only; vectors rank candidates."}


def _snapshot(workspace):
    workspace._check_binding()
    with closing(workspace._connect()) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        binding = json.loads(metadata["binding"])
        content = workspace._content_hash(connection)
        if binding != workspace._binding or content != metadata.get("content_sha256"):
            raise ValueError("Service sidecar content or binding changed")
    return {"store_sha256": binding["store_sha256"], "sidecar_binding": binding,
            "sidecar_content_sha256": content}


def _corpus(workspace):
    snapshot = _snapshot(workspace)
    groups = {}
    with closing(workspace._connect()) as connection:
        for row in connection.execute("SELECT json FROM observations ORDER BY id"):
            fact = json.loads(row[0])
            group_id = str(fact["scope"].get("group") or fact["id"])
            groups.setdefault(group_id, []).append(fact)
    records = []
    for group_id, facts in sorted(groups.items()):
        # Prefer a rate as group representative, while retaining every sibling ID.
        facts.sort(key=lambda f: ({"unit_price": 0, "hourly_rate": 0, "task_duration": 1}.get(f["kind"], 2), f["id"]))
        scope = facts[0]["scope"]
        labels = list(dict.fromkeys(f["label"].split(" — reported ")[0] for f in facts))
        text = (f"Equipment: {scope.get('equipment') or scope.get('equipment_family') or 'unspecified'}\n"
                f"Country: {scope.get('country') or 'unspecified'}\n"
                f"Observation: {'; '.join(labels)}\n"
                f"Kinds: {', '.join(sorted({f['kind'] for f in facts}))}\n"
                f"Literal units: {', '.join(sorted({f['unit'] for f in facts}))}")
        embedded = text[:MAX_DOCUMENT_CHARACTERS]
        records.append({"group_id": group_id, "report_id": facts[0]["report_id"],
            "fact_ids": [f["id"] for f in facts], "text_sha256": _hash(text),
            "embedded_text": embedded, "omitted_input_characters": max(0, len(text) - len(embedded))})
    if _snapshot(workspace) != snapshot:
        raise ValueError("Service evidence changed while preparing embedding inputs")
    return records, _hash(records), snapshot


def build_service_embeddings(workspace, output, client=None, modeltag=DEFAULT_MODEL, batch_size=16):
    """Write a new private group index; never pull a model or overwrite evidence."""
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a service embedding artifact")
    started = time.perf_counter()
    records, corpus_sha, snapshot = _corpus(workspace)
    if not records:
        raise ValueError("There are no source-bound service observation groups to embed")
    client = _client(client)
    model = _model_binding(client, modeltag)
    entries, timings, dimensions = [], [], None
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        vectors, timing = _embed(client, model["tag"], [record["embedded_text"] for record in batch], dimensions)
        dimensions = len(vectors[0])
        timings.append({"offset": offset, **timing})
        entries.extend({**record, "vector": vector} for record, vector in zip(batch, vectors))
    current_records, current_sha, current_snapshot = _corpus(workspace)
    if current_snapshot != snapshot or current_sha != corpus_sha or current_records != records:
        raise ValueError("Service evidence changed during embedding; no artifact was written")
    if _model_binding(client, model["tag"]) != model:
        raise ValueError("Local embedding model changed during service indexing")
    configuration = _configuration()
    artifact = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": snapshot, "corpus_sha256": corpus_sha, "model": model,
        "configuration": configuration, "configuration_sha256": _hash(configuration),
        "dimensions": dimensions, "entries": entries, "batch_size": batch_size,
        "timings": timings, "build_wall_seconds": time.perf_counter() - started}
    artifact["content_sha256"] = _hash(artifact)
    encoded = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return {"index_path": str(output), "groups": len(entries), "facts": sum(len(e["fact_ids"]) for e in entries),
            "dimensions": dimensions, "model": model, "binding": snapshot, "corpus_sha256": corpus_sha,
            "configuration_sha256": artifact["configuration_sha256"], "content_sha256": artifact["content_sha256"],
            "truncated_groups": sum(bool(e["omitted_input_characters"]) for e in entries),
            "batch_size": batch_size, "timings": timings, "build_wall_seconds": artifact["build_wall_seconds"]}


def _validated(workspace, index_path, client):
    artifact = json.loads(Path(index_path).read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite service index value")))
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported service embedding schema")
    if _hash({k: v for k, v in artifact.items() if k != "content_sha256"}) != artifact.get("content_sha256"):
        raise ValueError("Service embedding artifact checksum disagrees")
    configuration = _configuration()
    if artifact.get("configuration") != configuration or artifact.get("configuration_sha256") != _hash(configuration):
        raise ValueError("Service embedding settings or implementation changed")
    records, corpus_sha, snapshot = _corpus(workspace)
    if artifact.get("binding") != snapshot or artifact.get("corpus_sha256") != corpus_sha:
        raise ValueError("Service embedding evidence or grouped corpus changed")
    entries, dimensions = artifact.get("entries"), artifact.get("dimensions")
    if not isinstance(entries, list) or len(entries) != len(records) or type(dimensions) is not int or dimensions < 1:
        raise ValueError("Service embedding dimensions or group count is invalid")
    for entry, record in zip(entries, records):
        if not isinstance(entry, dict) or {k: v for k, v in entry.items() if k != "vector"} != record:
            raise ValueError("Service embedding group identity or text differs from evidence")
        _unit_vector(entry.get("vector"), dimensions)
    model = artifact.get("model")
    if not isinstance(model, dict) or _model_binding(client, model.get("tag")) != model:
        raise ValueError("Installed local service embedding model changed")
    return artifact, records


def check_service_embeddings(workspace, index_path, client=None):
    """Verify evidence and installed-model bindings without embedding inference."""
    artifact, records = _validated(workspace, index_path, _client(client))
    return {"valid": True, "groups": len(records), "facts": sum(len(r["fact_ids"]) for r in records),
            "dimensions": artifact["dimensions"], "model": artifact["model"],
            "binding": artifact["binding"], "corpus_sha256": artifact["corpus_sha256"],
            "content_sha256": artifact["content_sha256"]}


def search_service_embeddings(workspace, index_path, query, client=None, limit=30):
    """Return ranked source fact IDs; numeric approval stays with the workspace."""
    if not isinstance(query, str) or not query.strip() or len(query) > 6000:
        raise ValueError("Service semantic query must contain at most 6000 characters")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    client = _client(client)
    artifact, records = _validated(workspace, index_path, client)
    query_text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
    vectors, _ = _embed(client, artifact["model"]["tag"], [query_text], artifact["dimensions"])
    # Recheck complete sidecar/corpus/model bindings after the query call.
    current_records, corpus_sha, snapshot = _corpus(workspace)
    if (snapshot != artifact["binding"] or corpus_sha != artifact["corpus_sha256"] or current_records != records
            or _model_binding(client, artifact["model"]["tag"]) != artifact["model"]):
        raise ValueError("Service evidence or embedding model changed during search")
    ranked = []
    for entry in artifact["entries"]:
        vector = _unit_vector(entry["vector"], artifact["dimensions"])
        score = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(vectors[0], vector))))
        ranked.append((score, entry))
    ranked.sort(key=lambda row: (-row[0], row[1]["group_id"]))
    result = []
    for rank, (score, entry) in enumerate(ranked, 1):
        for identifier in entry["fact_ids"]:
            result.append({"id": identifier, "semantic_score": score, "group_id": entry["group_id"], "group_rank": rank})
            if len(result) == limit:
                return result
    return result


def search(workspace, query, *, country=None, equipment=None, mode="keyword",
           semantic_index=None, endpoint="http://127.0.0.1:11435", model=None):
    if mode not in {"keyword", "semantic", "hybrid"}:
        raise ValueError("Unknown service retrieval mode")
    keyword = workspace.search(query, country=country, equipment=equipment)
    if mode == "keyword":
        return {**keyword, "retrieval_mode": mode}, []
    path = Path(semantic_index or ".pilot-cache/assistant/service-semantic.json")
    if not path.is_file():
        raise ValueError("Build the optional service semantic index before selecting dense/hybrid service retrieval")
    binding = json.loads(path.read_text(encoding="utf-8"))
    embedding_tag = binding.get("model", {}).get("tag")
    if not isinstance(embedding_tag, str) or not embedding_tag:
        raise ValueError("Service semantic index has no installed-model binding")
    traces = []
    if model is not None:
        traces.append(model.unload())
    try:
        dense = search_service_embeddings(workspace, path, query, OllamaClient(endpoint, 120), limit=36)
    finally:
        traces.append(unload_local_model(endpoint, embedding_tag))
    dense_ids = list(dict.fromkeys(row["id"] for row in dense))
    keyword_ids = [fact["id"] for fact in keyword.get("facts", [])]
    ranked = ([row["id"] for row in fuse_rankings(keyword_ids, dense_ids)] if mode == "hybrid" else dense_ids)
    with closing(workspace._connect()) as connection:
        facts = workspace._read_facts(ranked[:12], connection)
    result = workspace._envelope(facts, keyword.get("warnings", []), query=query,
                                 retrieval_mode=mode, semantic_index=str(path),
                                 semantic_groups=len({row["group_id"] for row in dense}))
    return result, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "search"))
    parser.add_argument("--store", required=True)
    parser.add_argument("--sidecar")
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    workspace = ServiceWorkspace(args.store, args.sidecar)
    client = OllamaClient(args.endpoint, args.timeout)
    if args.action == "build":
        result = build_service_embeddings(workspace, args.index, client, args.model, args.batch_size)
    elif args.action == "check":
        result = check_service_embeddings(workspace, args.index, client)
    else:
        result = search_service_embeddings(workspace, args.index, args.query, client, args.limit)
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
