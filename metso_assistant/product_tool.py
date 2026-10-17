"""Read-only keyword product retrieval with source-bound evidence.

This baseline performs no model calls. Numerical answers come only from current
reviewed structured facts; narrative text and diagram crops do not approve facts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
from typing import Any

from .store import KnowledgeStore, quoted_fts_query

METHOD_VERSION = "fts5-bm25-complete-narrative-v2"
OPTIONAL_METHODS = {"semantic": "local-dense-source-filter-v2", "hybrid": "local-rrf60-lexical20-v2"}
STOP = set("a an and are as at be been but by can could do does for from has have how i if in into is it its me my of on or our please should so than that the their them there these they this to us we what when whenever which with would you your not no one give given get also instead whether explain find tell recommend suggest product products brochure brochures invent".split())
GENERIC_TITLE = STOP | set("metso brochure series filter filters pump pumps cell upgrade technical data product equipment".split())
# Interpretable language expansions, never a query-to-product answer map.
SYNONYMS = (
    ("feed", "slurry", "pulp"),
    ("grade", "composition", "characteristics", "properties"),
    ("change", "changing", "changes", "variable", "vary", "variation", "fluctuation"),
    ("performance", "efficiency", "productivity"),
    ("control", "adjust", "adjustment", "setpoint"),
    ("moisture", "wetness"),
    ("diagram", "schematic", "architecture", "system", "work"),
    ("height", "h"), ("width", "w"), ("length", "lenght"),
)
SPEC_WORDS = set("height width length dimension dimensions size weight mass volume area capacity power voltage frequency speed flow pressure temperature specification specifications spec rating footprint depth thickness recovery gain percentage percent guarantee guaranteed".split())
DIRECT_SPEC = SPEC_WORDS - {"pressure", "area", "recovery", "gain", "performance", "size"}
UNIT_WORDS = set("mm cm m inches inch in kg lb lbs kw hp hz rpm v vac vdc bar m2 m3 l litre liter litres liters percent".split())
VALUE_KEYS = ("value", "value_range", "value_components", "model_options", "operator", "quantity_kind")


def words(value: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:\.[0-9]+)?", value.casefold(), re.UNICODE)


def stem(word: str) -> str:
    if word in {"percentage", "percent"}:
        return "percent"
    if word == "mass":
        return "weight"
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        return base[:-1] if len(base) > 2 and base[-1:] == base[-2:-1] else base
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def terms(value: str) -> list[str]:
    separated = re.sub(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", " ", value)
    return [stem(w) for w in words(separated) if w not in STOP]


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", value.casefold())


def query_weights(query: str) -> dict[str, float]:
    weights = {term: 1.0 for term in terms(query)}
    for group in SYNONYMS:
        normalized = {stem(word) for word in group}
        if normalized.intersection(weights):
            for word in sorted(normalized):
                weights.setdefault(word, .4)
    return weights


def source_id(kind: str, *identity: Any) -> str:
    payload = json.dumps([kind, *identity], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "B" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def numeric_intent(query: str) -> bool:
    # A process-flow diagram names a visual subject, not a flow-rate value.
    query = re.sub(r"\bflow[- ]?(?:diagrams?|charts?|sheets?)\b", "diagram", query, flags=re.I)
    token_set = set(words(query))
    if re.search(r"\bpressure\s+filters?\b", query, re.I):
        token_set.discard("pressure")
    if token_set.intersection(DIRECT_SPEC):
        return True
    return bool(token_set.intersection(SPEC_WORDS) and re.search(r"\b(how much|how many|what is|what are|maximum|minimum|guaranteed?|rated|value|spec)\b", query, re.I))


def visual_intent(query: str) -> bool:
    return bool(re.search(r"\b(diagrams?|schematics?|architectures?|flow[- ]?charts?|flow[- ]?sheets?|layouts?|arrows?|figures?(?!\s+out\b)|drawings?)\b", query, re.I)
                or re.search(r"\b(?:show|inspect|picture|image|draw|look)\b.{0,50}\b(?:directions?|connections?|feedback loops?|control loops?)\b", query, re.I))


def document_scope(query: str, documents: dict[str, dict]) -> set[str]:
    query_words = set(words(query)) | set(terms(query))
    query_compact = compact(query)
    scores = {}
    for identifier, doc in documents.items():
        title_words = set(words(identifier.replace("-", " "))) - GENERIC_TITLE - {"a", "b"}
        shared = title_words.intersection(query_words)
        # Joined product names such as CellStation and PF-DS are recoverable.
        joined = "".join(w for w in words(identifier) if w not in GENERIC_TITLE and w not in {"a", "b"})
        if joined and len(joined) > 2 and joined in query_compact:
            shared = title_words
        score = len(shared) + (len(shared) / len(title_words) if title_words else 0)
        edition = re.search(r"\bbrochure\s*([ab])\b", query, re.I)
        if edition and re.search(r"\bbrochure[- ]" + edition[1].casefold() + r"\b", identifier):
            score += 3
        elif edition and re.search(r"\bbrochure[- ][ab]\b", identifier):
            score = 0
        if score:
            scores[identifier] = score
    if not scores:
        return set()
    maximum = max(scores.values())
    return {identifier for identifier, score in scores.items() if score == maximum}


def rank_chunks(chunks: list[dict], documents: dict[str, dict], query: str, scope: set[str]) -> list[tuple[float, dict]]:
    weights = query_weights(query)
    if not weights:
        return []
    identity_terms = {token for identifier in documents for token in terms(identifier.replace("-", " "))
                      if token not in {stem(value) for value in GENERIC_TITLE}}
    descriptive_terms = {token for token, weight in weights.items() if weight == 1 and token not in identity_terms and not token.isdecimal()}
    if not scope and len(descriptive_terms) >= 2:
        for token in identity_terms.intersection(weights):
            weights[token] *= .2
    frequencies = [Counter(terms(qualitative_excerpt(chunk)[0])) for chunk in chunks]
    df: Counter = Counter()
    for frequencies_row in frequencies:
        df.update(frequencies_row.keys())
    average = sum(sum(f.values()) for f in frequencies) / max(1, len(frequencies)) or 1
    ranked = []
    for chunk, freq in zip(chunks, frequencies):
        if scope and chunk["doc_id"] not in scope:
            continue
        length = sum(freq.values())
        score = 0.0
        for token, weight in weights.items():
            count = freq.get(token, 0)
            if count:
                inverse = math.log(1 + (len(chunks) - df[token] + .5) / (df[token] + .5))
                score += weight * inverse * count * 2.2 / (count + 1.2 * (.25 + .75 * length / average))
        # Metadata only refines already matched text, never creates a hit.
        if score:
            title_terms = set(terms(documents[chunk["doc_id"]]["title"]))
            score += .15 * sum(weight for token, weight in weights.items() if token in title_terms)
            # The bounded pool was retrieved with corpus FTS5 BM25. A small,
            # bounded tie-break retains that ordering after source-safe rerank.
            score += .05 * min(1.0, abs(chunk.get("fts_bm25", 0)))
            ranked.append((score, chunk))
    return sorted(ranked, key=lambda item: (-item[0], item[1]["id"]))


def fts_candidates(store: KnowledgeStore, query: str, scope: set[str], limit: int = 60) -> list[dict]:
    tokens = [word for word in words(query) if word not in STOP]
    normalized = {stem(token) for token in tokens}
    for group in SYNONYMS:
        if normalized.intersection(stem(word) for word in group):
            tokens.extend(group)
    expression = quoted_fts_query(" ".join(dict.fromkeys(tokens)))
    if expression is None:
        return []
    restriction, parameters = "", []
    if scope:
        restriction = " AND c.doc_id IN (" + ",".join("?" for _ in scope) + ")"
        parameters = sorted(scope)
    rows = store.connection.execute("""SELECT c.json,bm25(chunks_fts) AS score
        FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.chunk_id
        WHERE chunks_fts MATCH ? AND c.kind IN ('narrative','figure_caption')
        AND c.evidence_status='source_text'""" + restriction + " ORDER BY score,c.id LIMIT ?",
        (expression, *parameters, limit)).fetchall()
    return [{**json.loads(row["json"]), "fts_bm25": row["score"]} for row in rows]


def optional_ranked_chunks(store_path: str | Path, index_path: str | Path, query: str,
                           documents: dict[str, dict], scope: set[str], keyword: list[dict],
                           mode: str, endpoint: str, client: Any = None) -> tuple[list[tuple[float, dict]], dict]:
    """Keep optional dense signals; zero lexical overlap cannot erase a hit."""
    from .semantic import semantic_search, fuse_rankings
    from .model import OllamaClient
    client = client if client is not None else OllamaClient(endpoint, 120)
    dense = semantic_search(store_path, index_path, query, limit=60, client=client)

    def eligible(chunk: dict) -> bool:
        doc_id = chunk.get("doc_id")
        document = documents.get(doc_id) if isinstance(doc_id, str) else None
        return bool(document and (not scope or chunk["doc_id"] in scope)
                    and chunk.get("kind") in {"narrative", "figure_caption"}
                    and chunk.get("evidence_status") == "source_text"
                    and chunk.get("source_sha256") == document["source_sha256"]
                    and type(chunk.get("page")) is int and 1 <= chunk["page"] <= document["page_count"]
                    and qualitative_excerpt(chunk)[0].strip())

    dense = [chunk for chunk in dense if eligible(chunk)]
    keyword = [chunk for chunk in keyword if eligible(chunk)]
    metadata = {"mode": mode, "candidate_pool_limit": 60, "keyword_candidates": len(keyword),
                "dense_candidates": len(dense), "semantic_index_sha256": hashlib.sha256(Path(index_path).read_bytes()).hexdigest()}
    if mode == "semantic":
        output = []
        for rank, chunk in enumerate(dense, start=1):
            output.append((chunk["semantic_score"], {**chunk, "retrieval_features": {"dense_rank": rank, "semantic_score": chunk["semantic_score"]}}))
        return output, metadata
    fused = fuse_rankings([chunk["id"] for chunk in keyword], [chunk["id"] for chunk in dense], k=60)[:60]
    by_id = {chunk["id"]: chunk for chunk in dense}
    for chunk in keyword:
        by_id[chunk["id"]] = {**by_id.get(chunk["id"], {}), **chunk}
    pool = [by_id[row["id"]] for row in fused]
    lexical = {chunk["id"]: score for score, chunk in rank_chunks(pool, documents, query, scope)}
    maximum = max(lexical.values(), default=1) or 1
    output = []
    for rank, fusion in enumerate(fused, start=1):
        chunk = by_id[fusion["id"]]
        normalized_lexical = lexical.get(chunk["id"], 0) / maximum
        rank_signal = 61 / (60 + rank)
        score = .8 * rank_signal + .2 * normalized_lexical
        features = {**fusion, "fused_rank": rank, "normalized_lexical_score": normalized_lexical,
                    "rank_signal": rank_signal, "semantic_score": chunk.get("semantic_score")}
        output.append((score, {**chunk, "retrieval_features": features}))
    metadata.update({"rrf_k": 60, "fused_candidates": len(fused), "rank_weight": .8, "lexical_weight": .2})
    return sorted(output, key=lambda pair: (-pair[0], pair[1]["id"])), metadata


def qualitative_excerpt(chunk: dict) -> tuple[str, int]:
    """Keep complete source narrative for retrieval; approve no numerical facts.

    A paragraph mentioning a size, model or date can contain the strongest
    application evidence. Assertion authority is recorded separately from text
    visibility. Raw tables never reach this function through the retrieval gate.
    The second return value is retained for API compatibility.
    """
    return chunk.get("text", ""), 0


def table_label(table: dict, chunks: list[dict]) -> str:
    cells = table.get("normalized_table", {}).get("cells", [])
    for cell in cells[:4]:
        text = cell.get("text", "")
        if re.search(r"\b(series|model)\b", text, re.I):
            return text.replace("\n", " ")
    labels = [section for c in chunks if c["doc_id"] == table["doc_id"] and c["page"] == table["source"]["page"]
              and isinstance(section := c.get("section"), str) and section]
    return next((label for label in labels if re.search(r"\b(series|model)\b", label, re.I)), "")


def parameter_score(fact: dict, query: str, ignored: set[str]) -> float:
    query_terms = query_weights(query)
    label_terms = set(terms(fact["parameter"]))
    if fact.get("quantity_kind"):
        label_terms.update(terms(str(fact["quantity_kind"]).replace("_", " ")))
    # A lone dimension letter is meaningful only in its parameter label.
    for full, short in (("height", "h"), ("width", "w")):
        if short in label_terms:
            label_terms.add(full)
    return sum(weight for token, weight in query_terms.items() if token not in ignored and token in label_terms)


def match_facts(rows: list[dict], tables: dict[str, dict], documents: dict[str, dict], chunks: list[dict], query: str, scope: set[str]) -> tuple[list[dict], list[str]]:
    if not scope:
        return [], ["product"]
    selected = [row for row in rows if row["doc_id"] in scope]
    table_labels = {key: table_label(table, chunks) for key, table in tables.items() if table["doc_id"] in scope}
    query_tokens = words(query)
    query_joined = compact(query)
    configuration_tokens = {compact("".join(query_tokens[start:start + length]))
                            for start in range(len(query_tokens)) for length in range(1, 5)}
    series_numbers: set[str] = set()
    requested_series = set()
    for key, label in table_labels.items():
        for alpha, number in re.findall(r"\b([a-z]{1,8})\s*(\d+(?:\.\d+)?)", label.casefold()):
            if alpha + number in query_joined:
                requested_series.add(key)
                series_numbers.add(number)
    if requested_series:
        selected = [row for row in selected if row["table_key"] in requested_series]
    # A year is a variant selector, never a configuration quantity.
    year = re.search(r"\b(20\d{2})\b", query)
    if year:
        selected = [row for row in selected if year[1] in table_labels.get(row["table_key"], "")]
    if re.search(r"\blegacy\b", query, re.I):
        selected = [row for row in selected if not re.search(r"\bmodel\s*20\d{2}\b", table_labels.get(row["table_key"], ""), re.I)]
    all_configs = {config for row in selected if isinstance(config := row.get("config"), str)}
    configs: set[str] = set()
    for config in all_configs:
        normalized = compact(config)
        if re.search(r"[a-z]", normalized) and normalized in configuration_tokens:
            configs.add(config)
        elif normalized in query_tokens and normalized not in series_numbers:
            # Qualifier numbers (25 mm, 50 Hz, page 5) are not configurations.
            if not re.search(r"\b" + re.escape(normalized) + r"\s*(?:" + "|".join(sorted(UNIT_WORDS)) + r")\b", query, re.I) and not re.search(r"\bpage\s+" + re.escape(normalized) + r"\b", query, re.I):
                configs.add(config)
    ignored = {stem(w) for identifier in scope for w in words(identifier)} | {stem(w) for w in series_numbers} | {stem(w) for w in configs}
    # A configuration/count mentioned in the question is not the requested
    # quantity. If an explicit quantity axis has no source parameter, never
    # substitute another field merely because it shares a contextual word.
    axis_query = re.sub(r"\bpressure\s+filters?\b", "filter", query, flags=re.I)
    requested_axes = {stem(word) for word in words(axis_query)
                      if word in SPEC_WORDS - {"guarantee", "guaranteed", "spec", "specification", "specifications"}}
    available_axes = set()
    for row in selected:
        available_axes.update(terms(row["payload"]["parameter"]))
        available_axes.update(terms(str(row["payload"].get("quantity_kind", "")).replace("_", " ")))
    if "h" in available_axes:
        available_axes.add("height")
    if "w" in available_axes:
        available_axes.add("width")
    if requested_axes - available_axes:
        return [], ["supported_parameter"]
    scored = [(parameter_score(row["payload"], query, ignored), row) for row in selected]
    scored = [(score, row) for score, row in scored if score > 0]
    if not scored:
        return [], ["supported_parameter"]
    best = max(score for score, _ in scored)
    matched = [row for score, row in scored if score == best]
    if configs:
        matched = [row for row in matched if row.get("config") in configs or row.get("config") is None]
    elif any(row.get("config") is not None for row in matched):
        # Return statuses/choices for discovery, withhold all quantity values.
        for row in matched:
            row["selection_blocker"] = "configuration_not_selected"
        return sorted(matched, key=lambda row: row["fact_key"]), ["configuration"]
    units = {token for token in query_tokens if token in UNIT_WORDS - {"in"}}
    if re.search(r"\b(inch|inches)\b", query, re.I):
        units.add("in")
    if units:
        filtered = [row for row in matched if str(row["payload"].get("unit", "")).casefold() in units]
        if filtered:
            matched = filtered
    # Parameters with different chamber depths remain separately labelled; an
    # underspecified quantity must not become an arbitrary first answer.
    distinct_parameters = {row["payload"]["parameter"] for row in matched}
    qualifiers = {number for parameter in distinct_parameters for number in re.findall(r"\d+(?:\.\d+)?", parameter)}
    if len(distinct_parameters) > 1 and qualifiers and not qualifiers.intersection(terms(query)):
        for row in matched:
            row["selection_blocker"] = "parameter_qualifier_not_selected"
        return sorted(matched, key=lambda row: row["fact_key"]), ["parameter_qualifier"]
    return sorted(matched, key=lambda row: (row["doc_id"], row["page"], row["table_id"], row["fact_id"])), []


def trusted(row: dict) -> bool:
    fact = row["payload"]
    return (not row.get("selection_blocker") and row["review_status"] == fact.get("review_status") == "verified"
            and row["status"] == fact.get("status") == "reported" and row["blockers"] == 0
            and not fact.get("review_blockers")
            and not any(item.get("severity") == "blocker" for item in fact.get("issues", []) if isinstance(item, dict)))


def relative_artifact(root: Path, reference: Any) -> Path:
    if not isinstance(reference, str) or not reference or ":" in reference or PureWindowsPath(reference).drive or reference.startswith(("/", "\\")):
        raise ValueError("Invalid local artifact reference")
    parts = reference.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Artifact traversal is not allowed")
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Artifact unavailable or outside raw evidence root")
    return path


def visuals(raw_dir: Path | None, targets: list[tuple[str, int]], documents: dict[str, dict], limit: int) -> tuple[list[dict], list[dict], list[str]]:
    if raw_dir is None:
        return [], [], ["Visual evidence was requested but no raw evidence directory was provided."]
    candidates, sources, warnings = [], [], []
    for doc_id, number in list(dict.fromkeys(targets))[:limit]:
        doc = documents[doc_id]
        # Full-page render comes first; crops can omit arrows or nearby labels.
        for parser in ("docling", "mineru"):
            parser_root = raw_dir / parser
            try:
                if not parser_root.resolve().is_relative_to(raw_dir.resolve()):
                    raise ValueError("Parser root escaped selected evidence root")
                metadata_path = relative_artifact(parser_root, f"{doc_id}/page-{number:03d}/page.json")
                page = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (page.get("page") != number or page.get("source_sha256") != doc["source_sha256"] or page.get("source_file") != doc["source_file"] or page.get("status") not in {"success", "ok"}):
                    raise ValueError("Raw page source identity/status disagrees with the store")
                inventory = {item["path"]: item for item in page.get("artifacts", [])}
                items = [("full_page", page.get("source_render_file"), None)]
                items += [("figure_crop", item.get("image_file"), item.get("bbox")) for item in page.get("figures", [])]
                for kind, reference, box in items:
                    if not reference:
                        continue
                    path = relative_artifact(parser_root, reference)
                    if path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
                        raise ValueError("Visual artifact must be a local image")
                    record = inventory.get(reference)
                    if record is None or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
                        raise ValueError("Visual artifact hash differs from its raw page inventory")
                    identifier = source_id(kind, doc_id, doc["source_sha256"], number, parser, reference)
                    candidates.append({"source_id": identifier, "doc_id": doc_id, "file": doc["source_file"], "page": number, "sha": doc["source_sha256"], "parser": parser, "kind": kind, "image_file": str(path), "bbox": box, "requires_inspection": True})
                    sources.append({"id": identifier, "file": doc["source_file"], "page": number, "sha": doc["source_sha256"], "kind": kind, "excerpt": "", "image_file": str(path), "requires_inspection": True})
                break
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                warnings.append(f"{doc_id} page {number} {parser}: {exc}")
    return candidates, sources, warnings


def retrieve_product(store_path: str | Path, query: str, *, raw_dir: str | Path | None = None, top_k: int = 6,
                     retrieval_mode: str = "keyword", semantic_index: str | Path | None = None,
                     embedding_endpoint: str = "http://127.0.0.1:11434", embedding_client: Any = None,
                     scope_doc_ids: list[str] | None = None) -> dict:
    """Return bounded context and reviewed facts; never recommend or estimate."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Product query must be nonempty text")
    if len(query) > 8000:
        raise ValueError("Product query is too long")
    if type(top_k) is not int or not 1 <= top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    if scope_doc_ids is not None and (not isinstance(scope_doc_ids, list) or not 1 <= len(scope_doc_ids) <= 64
            or any(not isinstance(value, str) or not value.strip() for value in scope_doc_ids)):
        raise ValueError("scope_doc_ids must be a nonempty bounded list of actual document IDs")
    if retrieval_mode not in {"keyword", "semantic", "hybrid"}:
        raise ValueError("retrieval_mode must be keyword, semantic or hybrid")
    if retrieval_mode != "keyword" and (semantic_index is None or not Path(semantic_index).is_file()):
        raise ValueError("Optional dense retrieval requires an existing source-bound semantic index")
    numerical, visual = numeric_intent(query), visual_intent(query)
    qualitative_request = bool(re.search(r"\b(recommend\w*|suggest\w*|upgrad\w*|solutions?|help\w*|improv\w*|describ\w*|explain\w*|alternatives?|options?)\b", query, re.I))
    context_query = query
    if numerical and qualitative_request:
        # Retrieve the qualitative symptom independently of an unsupported
        # quantity request. The original query still controls fact selection.
        context_query = " ".join(word for word in words(query) if word not in SPEC_WORDS | {"improvement", "improvements"})
    with KnowledgeStore(store_path) as store:
        documents = {row["id"]: json.loads(row["json"]) for row in store.connection.execute("SELECT id,json FROM documents")}
        if scope_doc_ids is not None and set(scope_doc_ids) - documents.keys():
            raise ValueError("scope_doc_ids contains an unknown document identity")
        from .product_identity import source_product_identity
        identity_chunks = defaultdict(list)
        for row in store.connection.execute("SELECT json FROM chunks WHERE kind IN ('narrative','figure_caption') AND evidence_status='source_text' ORDER BY doc_id,page,id"):
            chunk = json.loads(row["json"])
            identity_chunks[chunk["doc_id"]].append(chunk)
        products = {doc_id: source_product_identity(doc, identity_chunks[doc_id]) for doc_id, doc in documents.items()}
        scope = set(scope_doc_ids) if scope_doc_ids is not None else document_scope(query, documents) if numerical or visual else set()
        chunks = fts_candidates(store, context_query, scope) if retrieval_mode != "semantic" else []
        tables, fact_rows, label_chunks = {}, [], chunks
        if numerical and scope:
            placeholders = ",".join("?" for _ in scope)
            table_rows = store.connection.execute(f"SELECT table_key,json FROM tables WHERE doc_id IN ({placeholders})", sorted(scope))
            tables = {row["table_key"]: json.loads(row["json"]) for row in table_rows}
            rows = store.connection.execute(f"SELECT * FROM facts WHERE doc_id IN ({placeholders})", sorted(scope))
            fact_rows = [{**dict(row), "payload": json.loads(row["json"])} for row in rows]
            label_chunks = [json.loads(row["json"]) for row in store.connection.execute(
                f"SELECT json FROM chunks WHERE doc_id IN ({placeholders}) AND kind='narrative' AND evidence_status='source_text'", sorted(scope))]
    retrieval_metadata = None
    if retrieval_mode == "keyword":
        ranked = rank_chunks(chunks, documents, context_query, scope)
    else:
        assert semantic_index is not None
        ranked, retrieval_metadata = optional_ranked_chunks(store_path, semantic_index, context_query, documents,
                                                          scope, chunks, retrieval_mode, embedding_endpoint, embedding_client)
    passages, facts, unavailable, sources, warnings, missing = [], [], [], [], [], []
    for score, chunk in ranked:
        excerpt, omitted = qualitative_excerpt(chunk)
        if not excerpt:
            continue
        doc = documents[chunk["doc_id"]]
        product = products[chunk["doc_id"]]
        identifier = source_id("passage", chunk["id"], doc["source_sha256"])
        passages.append({"source_id": identifier, "chunk_id": chunk["id"], "doc_id": chunk["doc_id"],
                         "product_id": chunk["doc_id"], "product_name": product["name"], "product_aliases": product["aliases"],
                         "page": chunk["page"], "section": chunk.get("section"), "text": excerpt, "kind": chunk["kind"],
                         "authority": "qualitative_source_text", "numeric_assertion_authority": False,
                         "contains_numeric_text": bool(re.search(r"\d", excerpt)), "score": round(score, 6), "omitted_numeric_blocks": omitted})
        if retrieval_mode != "keyword":
            passages[-1]["retrieval_features"] = chunk.get("retrieval_features", {})
        sources.append({"id": identifier, "file": doc["source_file"], "page": chunk["page"], "sha": doc["source_sha256"], "kind": chunk["kind"], "excerpt": excerpt, "section": chunk.get("section"), "chunk_id": chunk["id"], "edition_identity": doc.get("edition_identity"), "numeric_assertion_authority": False})
        if len(passages) == top_k:
            break
    if numerical:
        matched, missing = match_facts(fact_rows, tables, documents, label_chunks, query, scope)
        for row in matched[:top_k]:
            original = row["payload"]
            doc = documents[row["doc_id"]]
            identifier = source_id("fact", row["fact_key"], doc["source_sha256"])
            approved = trusted(row)
            fact = {"source_id": identifier, "fact_id": original["fact_id"], "doc_id": row["doc_id"], "page": row["page"], "table_id": row["table_id"], "configuration": original.get("configuration"), "parameter": original["parameter"], "unit": original.get("unit"), "status": original["status"], "review_status": original.get("review_status", "pending"), "trusted": approved, "authority": "reviewed_source_value" if approved else "unapproved_candidate_identity_only", "value_kind": original.get("value_kind"), "issues": original.get("issues", []), "review_blockers": original.get("review_blockers", []), "selection_blocker": row.get("selection_blocker"), "source_refs": original.get("source_refs", []), "edition_identity": doc.get("edition_identity"), "table_label": table_label(tables[row["table_key"]], label_chunks)}
            if approved:
                fact.update({key: original[key] for key in VALUE_KEYS if key in original})
            else:
                warnings.append("A matching candidate is unapproved, unavailable or underspecified; its quantity values are withheld.")
            for item in original.get("issues", []):
                if isinstance(item, dict) and item.get("severity") in {"warning", "blocker"}:
                    warnings.append(f"{doc['source_file']} page {row['page']}: {item.get('code', 'source_warning')}: {item.get('message', '')}")
            # Excerpt is a source-bound JSON snapshot, not a fabricated PDF quote.
            snapshot = {key: fact[key] for key in ("configuration", "parameter", "unit", "status", "review_status", *VALUE_KEYS) if key in fact}
            sources.append({"id": identifier, "file": doc["source_file"], "page": row["page"], "sha": doc["source_sha256"], "kind": "reviewed_fact" if approved else "candidate_identity", "excerpt": json.dumps(snapshot, sort_keys=True, ensure_ascii=False), "excerpt_kind": "stored_fact_snapshot", "fact_id": original["fact_id"], "table_id": row["table_id"], "source_refs": original.get("source_refs", []), "edition_identity": doc.get("edition_identity")})
            (facts if approved else unavailable).append(fact)
        if len(matched) > top_k:
            warnings.append(f"Matched fact results were limited to {top_k}; refine the model/configuration and parameter.")
        if not any(fact["trusted"] for fact in facts):
            warnings.append("No reviewed, reported and unblocked numerical fact answers the selected request. Qualitative passages cannot supply a numerical answer.")
    visual_candidates = []
    if visual:
        page_scores: dict[tuple[str, int], float] = defaultdict(float)
        page_cues: dict[tuple[str, int], bool] = defaultdict(bool)
        for score, chunk in ranked:
            key = (chunk["doc_id"], chunk["page"])
            # Source headings locate likely diagram pages; this is a retrieval
            # cue, not an assertion that any arrow or relationship was observed.
            section = chunk.get("section") or ""
            heading_cue = bool(set(terms(section)).intersection({"diagram", "schematic", "architecture", "work", "flowsheet", "flowchart", "layout"})
                               or re.search(r"\bflow[- ](?:sheets?|charts?)\b", section, re.I))
            page_scores[key] = max(page_scores[key], score)
            page_cues[key] = page_cues[key] or heading_cue
        targets = sorted(page_scores, key=lambda key: (-page_cues[key], -page_scores[key], key))[:min(top_k, 3)]
        visual_candidates, visual_sources, visual_warnings = visuals(Path(raw_dir) if raw_dir is not None else None, targets, documents, 3)
        crop_counts = Counter((item["doc_id"], item["page"]) for item in visual_candidates if item["kind"] == "figure_crop")
        target_order = {key: index for index, key in enumerate(targets)}
        # Prefer headed pages with broader native visual coverage for the first
        # inspection. Crop count locates evidence; it never proves semantics.
        visual_candidates.sort(key=lambda item: (-page_cues[(item["doc_id"], item["page"])],
                                                 -crop_counts[(item["doc_id"], item["page"])],
                                                 target_order[(item["doc_id"], item["page"])],
                                                 item["kind"] != "full_page"))
        sources.extend(visual_sources); warnings.extend(visual_warnings)
        warnings.append("Diagram relationships and plotted numbers require inspection of the linked original page; keyword matches and figure counts are not visual observations.")
    sufficient = any(fact["trusted"] for fact in facts) if numerical else bool(passages)
    if numerical and qualitative_request and passages:
        sufficient = True
    if visual and not visual_candidates:
        sufficient = False
    selected_documents = {row["doc_id"] for row in passages + facts + unavailable + visual_candidates}
    result = {"tool": "product", "status": "ok" if sufficient else "insufficient_evidence", "data": {"products": [products[doc_id] for doc_id in sorted(selected_documents)], "passages": passages, "facts": facts, "unavailable_facts": unavailable, "visual_candidates": visual_candidates, "needs_visual_inspection": visual, "retrieval_method": METHOD_VERSION, "candidate_pool_limit": 60, "context_query": context_query, "request_kind": "mixed" if numerical and qualitative_request else "numerical" if numerical else "visual" if visual else "qualitative", "evidence_policy": {"numerical_authority": "Only trusted reviewed_source_value facts; no values may be inferred from passages, names or images.", "narrative_numbers": "Complete source narrative, including numbers, is available for retrieval and application selection. Narrative numbers confer no numerical assertion authority.", "identity": "Product names and aliases have separate source identity evidence; identity does not establish a capability or application fit.", "candidate_values": "Pending, ambiguous, invalid, missing, blocked and underspecified quantity values are withheld in unavailable_facts.", "editions": "Each source file/hash remains a separate edition; conflicting variants must be described separately.", "visuals": "Candidates require inspection and confer no diagram or numerical authority.", "hybrid_embeddings": "Optional source-bound local retrieval; reviewed numerical and visual gates remain separate."}}, "sources": sources, "warnings": list(dict.fromkeys(warnings)), "missing_fields": missing, "question": None}
    if retrieval_metadata is not None:
        result["data"]["retrieval_mode"] = retrieval_mode
        result["data"]["retrieval_method"] = OPTIONAL_METHODS[retrieval_mode]
        result["data"]["retrieval_metadata"] = retrieval_metadata
        result["data"]["evidence_policy"]["hybrid_embeddings"] = "Optional local qualitative retrieval only; numerical gates and source-page inspection requirements are unchanged."
    result["data"]["scope_doc_ids"] = sorted(scope)
    return result
