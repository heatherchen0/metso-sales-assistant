"""Source-derived brochure identities, separate from capability evidence.

Document filenames help locate a name; they never authorize a product name that
does not occur in source text. Identity evidence does not approve a specification
or establish that a product fits the customer's application.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re


_GENERIC = set("a b the metso brochure product products technical data edition series filter filters cell cells pump pumps upgrade of horizontal".split())


def identity(value) -> str:
    return "".join(re.findall(r"[^\W_]+", str(value).casefold(), re.UNICODE))


def _words(value):
    return re.findall(r"[^\W_]+", str(value).casefold(), re.UNICODE)


def _source_rows(document, chunks):
    for chunk in chunks:
        if chunk.get("doc_id") != document.get("id") or chunk.get("kind") not in {"narrative", "figure_caption"}:
            continue
        if chunk.get("evidence_status") != "source_text":
            continue
        if document.get("source_sha256") and chunk.get("source_sha256") != document["source_sha256"]:
            continue
        if type(chunk.get("page")) is not int or not 1 <= chunk["page"] <= document.get("page_count", chunk["page"]):
            continue
        yield chunk


def _spans(chunk):
    blocks = chunk.get("blocks") or [{"text": chunk.get("text", "")}]
    for block in blocks:
        text = block.get("text", "")
        for span in re.split(r"\n+|\n\s*\n", text):
            span = span.strip(" \t\r\n#")
            if not span:
                continue
            # A title can be one extracted paragraph. Sentence subjects provide
            # a modest fallback for injected tools lacking native headings.
            yield span, block.get("label", block.get("type", ""))
            match = re.match(r"(?:The\s+)?(.{2,100}?)\s+(?:is|are|offers?|provides?|adjusts?|uses?|enables?|improves?|controls?)\b", span, re.I)
            if match:
                yield match[1].strip(), "sentence_subject"


def source_product_identity(document: dict, chunks: list[dict]) -> dict:
    """Return one source-supported document identity and its exact provenance.

    An unresolved record has ``name=None``. Callers may still retrieve its text;
    they must not turn the filename into an asserted product name.
    """
    rows = sorted(_source_rows(document, chunks), key=lambda row: (row["page"], row.get("id", "")))
    hint = set(_words(str(document.get("title", "")) + " " + str(document.get("id", "")))) - _GENERIC
    options = []
    for chunk in rows:
        for span, label in _spans(chunk):
            tokens = _words(span)
            if not 1 <= len(tokens) <= 14 or len(span) > 140:
                continue
            overlap = hint & set(tokens)
            heading = label in {"section_header", "title", "heading"}
            if not overlap and not (chunk["page"] == 1 and (heading or label == "sentence_subject")):
                continue
            coverage = len(overlap) / max(1, len(hint))
            score = 5 * coverage + 2 * bool(overlap) + 1.5 * heading + .6 * (chunk["page"] == 1) - .04 * len(tokens)
            options.append((score, chunk["page"], span, chunk))
    best = sorted(options, key=lambda row: (-row[0], row[1], len(row[2]), row[2]))[0] if options else None
    result = {"product_id": document.get("id"), "doc_id": document.get("id"), "name": None,
              "aliases": [], "title": document.get("title"), "source_file": document.get("source_file"),
              "source_sha256": document.get("source_sha256"), "identity_evidence": [],
              "authority": "source_identity_only", "status": "unresolved"}
    if best is None:
        return result
    _, _, name, chunk = best
    if name.casefold().startswith("the "):
        name = name[4:]
    aliases = [name]
    # Search aliases are actual source tokens, not invented model variants.
    # Generic nouns are excluded; shared brand aliases may remain ambiguous.
    for token in re.findall(r"[^\W_]+", name, re.UNICODE):
        if token.casefold() in hint and token.casefold() not in _GENERIC and (len(token) >= 4 or token.isupper()):
            if identity(token) not in {identity(alias) for alias in aliases}:
                aliases.append(token)
            break
    for match in re.finditer(r"\b[A-Z]{2,}(?:[- ][A-Z]{2,})+\b", name):
        if identity(match[0]) not in {identity(alias) for alias in aliases}:
            aliases.append(match[0])
    result.update(name=name, aliases=aliases, status="source_supported", identity_evidence=[{
        "doc_id": document.get("id"), "chunk_id": chunk.get("id"), "page": chunk["page"],
        "source_file": document.get("source_file"), "source_sha256": document.get("source_sha256"),
        "quote": name, "authority": "source_identity_only"}])
    return result


def catalogue_from_store(store_path) -> list[dict]:
    from .store import KnowledgeStore
    if not Path(store_path).is_file():
        return []
    with KnowledgeStore(store_path) as store:
        documents = [json.loads(row["json"]) for row in store.connection.execute("SELECT json FROM documents ORDER BY id")]
        grouped = defaultdict(list)
        for row in store.connection.execute("SELECT json FROM chunks WHERE kind IN ('narrative','figure_caption') AND evidence_status='source_text' ORDER BY doc_id,page,id"):
            chunk = json.loads(row["json"])
            grouped[chunk["doc_id"]].append(chunk)
    return [source_product_identity(document, grouped[document["id"]]) for document in documents]


def products_from_tool(tool: dict) -> list[dict]:
    """Use retrieval's catalogue, with a source-text fallback for injected tools."""
    products = tool.get("data", {}).get("products")
    if isinstance(products, list):
        return [dict(row) for row in products if isinstance(row, dict) and row.get("name") and row.get("doc_id")]
    sources = {row.get("id"): row for row in tool.get("sources", [])}
    grouped = defaultdict(list)
    for index, passage in enumerate(tool.get("data", {}).get("passages", [])):
        doc_id = passage.get("doc_id")
        if not doc_id:
            continue
        source = sources.get(passage.get("source_id"), {})
        grouped[doc_id].append({"id": passage.get("chunk_id", f"injected-{index}"), "doc_id": doc_id,
                               "page": source.get("page", passage.get("page", 1)),
                               "kind": "narrative", "evidence_status": "source_text",
                               "source_sha256": source.get("sha"), "text": source.get("excerpt", passage.get("text", ""))})
    result = []
    for doc_id, chunks in grouped.items():
        source = next((sources.get(row.get("source_id"), {}) for row in tool.get("data", {}).get("passages", []) if row.get("doc_id") == doc_id), {})
        document = {"id": doc_id, "title": doc_id.replace("-", " "), "source_file": source.get("file"),
                    "source_sha256": source.get("sha"), "page_count": max(row["page"] for row in chunks)}
        record = source_product_identity(document, chunks)
        if record["name"]:
            result.append(record)
    return result


def name_occurs(product: dict, text: str) -> bool:
    """Alias matching is a subject-discovery hint, never application evidence."""
    tokens = _words(text)
    haystack = " " + " ".join(tokens) + " "
    return any(" " + " ".join(_words(alias)) + " " in haystack
               or len(_words(alias)) > 1 and identity(alias) in identity(text)
               for alias in [product.get("name", ""), *product.get("aliases", [])] if alias)
