"""Generic, source-bound service observations and dimensional arithmetic.

The sidecar is a disposable FTS index, not a replacement for prepared evidence.
It contains no product-specific part aliases, price constants or job workflows.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import tempfile
from typing import Any

from .services import CURRENCIES, json_numbers
from .store import KnowledgeStore


METHOD_VERSION = "generic-service-observations-3"
SCHEMA_VERSION = 1
MAX_SEARCH_FACTS = 12
MAX_REPORT_FACTS = 20
_INDIVIDUAL = {"each", "piece", "unit", "pcs", "pc", "cloth"}
_COUNT_UNITS = {"each", "pair", "tube", "set", "kit", "bottle", "roll", "box", "lot"}
_QUANTITY_UNITS = _COUNT_UNITS | {"ml", "litre"}
_STOP = set("a an and are as at be by for from in into is it of on or the their this to with "
            "customer needs need what should budget service visit replace replaced replacement "
            "install installed installation change changed removal remove task performed item "
            "quantity qty rate total cost hours hour hrs hr units unit pieces piece pcs each "
            "kit kits set sets lot lots part parts work system equipment".split())
_DURATION_ROLES = {"travel_hours", "labour_hours"}
_COUNT_WORDS = {word: index for index, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_LITERAL_COUNT = re.compile(r"(?<![\w.\-])(?:[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)|"
                            + "|".join(_COUNT_WORDS) + r")(?![\w.]|\.\d)", re.I)


def _unit(raw):
    """Literal unit normalization; never a numerical unit conversion."""
    text = str(raw or "").casefold().strip()
    if text in _INDIVIDUAL | {"ea"}:
        return "each"
    if text in {"h", "hr", "hrs", "hour", "hours"}:
        return "hour"
    if text in {"l", "liter", "liters", "litre", "litres"}:
        return "litre"
    if text in {"ml", "milliliter", "milliliters", "millilitre", "millilitres"}:
        return "ml"
    if text.endswith("s") and text[:-1] in _COUNT_UNITS:
        return text[:-1]
    return text if text in _QUANTITY_UNITS else None


def _description_hours(description, quantity):
    """One literal same-row hour quantity can supply a missing unit label."""
    matches = list(re.finditer(r"(?<![\w.-])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", description, re.I))
    return (len(matches) == 1 and quantity is not None and quantity > 0
            and Decimal(matches[0][1].replace(",", "")) == quantity)


def _independent_hour_refs(record, category, quantity, row_refs):
    """Prove a row's missing hour unit from an exact independent visit-hour statement."""
    field = record.get("hours", {}).get("travel" if category == "travel" else "on_site", {})
    if (category not in {"travel", "labour"} or quantity is None or quantity <= 0
            or field.get("status") != "reported" or field.get("approximate")
            or field.get("range") is not None or _number(field.get("value")) != quantity):
        return []
    occupied = _lines(row_refs)
    independent = []
    for candidate in field.get("candidates", []):
        refs = _lines(candidate.get("evidence", []))
        if (candidate.get("scope") != "visit_total" or candidate.get("approximate")
                or candidate.get("range") is not None or _number(candidate.get("value")) != quantity or not refs):
            continue
        if any(left["line_start"] <= right["line_end"] and right["line_start"] <= left["line_end"]
               for left in refs for right in occupied):
            continue
        independent.extend(refs)
    return _lines(independent)


def _literal_count_spans(fact):
    text = fact.get("user_text", "")
    if not isinstance(text, str) or not text:
        return []
    matches = []
    for match in _LITERAL_COUNT.finditer(text):
        token = match.group().casefold()
        value = Decimal(_COUNT_WORDS[token]) if token in _COUNT_WORDS else Decimal(token.replace(",", ""))
        if value == fact["value"]:
            matches.append(match.span())
    span = fact.get("scope", {}).get("span")
    if isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span):
        return [item for item in matches if item == tuple(span)]
    return matches


_REPETITION = re.compile(r"\b(?:(?:for\s+)?each|(?:for\s+)?every|per)\b", re.I)
_CLAUSE_BREAK = re.compile(r"[;!?\n]|(?<!\d)\.(?!\d)")
_REQUEST_VERB = re.compile(r"\b(?:needs?|requires?|wants?|provides?|replace|replacing|replaced|install|installing|installed|fit|supply|use|repair|like)\b", re.I)
_NOUN_STOP = {"and", "or", "but", "need", "needs", "require", "requires", "want", "wants", "have", "has",
              "get", "gets", "at", "for", "with", "on", "per", "each", "every", "price", "budget", "cost",
              "provide", "provides", "replace", "replacing", "install", "installing", "fit", "supply", "use", "repair"}
_UNSAFE_LIST = re.compile(r"[$\u20ac\u00a3]|\b(?:USD|EUR|AUD|CAD|SEK|ZAR|rate|price|priced|charge|charged|overall|altogether|combined|or|plus|but|instead|however)\b|\bin\s+total\b|\bat\s+\d", re.I)


def _noun_words(text):
    words = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.casefold()):
        if word in _NOUN_STOP:
            break
        if word in {"of", "the", "these", "those"} or word in _COUNT_WORDS:
            continue
        words.append(word[:-1] if len(word) > 3 and word.endswith("s") else word)
        if len(words) == 5:
            break
    return set(words)


def _clause_bounds(text, at):
    before = [match.end() for match in _CLAUSE_BREAK.finditer(text, 0, at)]
    following = _CLAUSE_BREAK.search(text, at)
    return before[-1] if before else 0, following.start() if following else len(text)


def _safe_component_list(text, *, allow_counts=False):
    """Conjunctive quantities may share a modifier; alternatives never do."""
    if _CLAUSE_BREAK.search(text) or _UNSAFE_LIST.search(text):
        return False
    counts = list(_LITERAL_COUNT.finditer(text))
    if not counts:
        return True
    if not allow_counts:
        return False
    previous = 0
    for count in counts:
        if not re.search(r"\band\b", text[previous:count.start()], re.I):
            return False
        previous = count.end()
    return True


def _antecedent_before(text, equipment_span, boundary):
    if equipment_span[1] > boundary:
        return False
    bridge = text[equipment_span[1]:boundary]
    # A bare Each may refer to an earlier counted subject in the same clause
    # or the immediately preceding sentence, never across another count.
    return (len(list(_CLAUSE_BREAK.finditer(bridge))) <= 1
            and not _LITERAL_COUNT.search(bridge) and not _UNSAFE_LIST.search(bridge))


def _repetition_relation(per_item, equipment_count):
    """Conservative literal proof for a component list repeated per subject."""
    text, equipment_text = per_item.get("user_text", ""), equipment_count.get("user_text", "")
    per_spans, equipment_spans = _literal_count_spans(per_item), _literal_count_spans(equipment_count)
    for equipment_span in equipment_spans:
        tail = re.split(r"[;,.!?\n:]", equipment_text[equipment_span[1]:equipment_span[1] + 100], maxsplit=1)[0]
        nouns = _noun_words(tail)
        if not nouns:
            continue
        for per_span in per_spans:
            if text == equipment_text and per_span == equipment_span:
                continue
            for relation in _REPETITION.finditer(text):
                start, end = _clause_bounds(text, relation.start())
                if not start <= per_span[0] < end:
                    continue
                target = re.split(r"[,;.!?\n:]", text[relation.end():end], maxsplit=1)[0]
                if _UNSAFE_LIST.search(target):
                    continue
                target_nouns = _noun_words(target)
                named_subject = bool(nouns & target_nouns)
                after_count = relation.start() >= per_span[1]
                if after_count:
                    verbs = list(_REQUEST_VERB.finditer(text, start, per_span[0]))
                    list_start = verbs[-1].end() if verbs else start
                    body = text[list_start:relation.start()]
                    # Check the complete shared list, including an OR or total
                    # qualifier before the selected final component.
                    if _UNSAFE_LIST.search(body):
                        continue
                    between = text[per_span[1]:relation.start()]
                    if not _safe_component_list(between, allow_counts=relation.group().casefold() != "per"):
                        continue
                    if named_subject:
                        return True
                    bare_each = relation.group().casefold() in {"each", "for each"} and not target.strip()
                    if bare_each and (text != equipment_text or
                                      equipment_span[0] >= start and _antecedent_before(text, equipment_span, list_start)):
                        return True
                else:
                    body = text[relation.end():end]
                    if _UNSAFE_LIST.search(body):
                        continue
                    # Leading Each governs the whole following conjunction,
                    # so an earlier component count does not break its scope.
                    bridge = text[relation.end():per_span[0]]
                    counts = list(_LITERAL_COUNT.finditer(bridge))
                    if counts:
                        last = counts[-1]
                        if not re.search(r"\band\b", bridge[last.end():], re.I):
                            continue
                    if named_subject:
                        return True
                    if (relation.group().casefold() == "each" and text == equipment_text
                            and _REQUEST_VERB.search(body)
                            and _antecedent_before(text, equipment_span, relation.start())):
                        return True
    return False


def quantity_relations(inputs):
    """Return only proven literal user-count pairs; never source-price facts.

    This is a conservative text proof, not a general language parser. Missing
    or ambiguous relationships remain unavailable to the calculation planner.
    """
    if not isinstance(inputs, list):
        return []
    eligible = []
    for fact in inputs:
        if not isinstance(fact, dict) or fact.get("kind") != "user_input" or not isinstance(fact.get("id"), str):
            continue
        value = _number(fact.get("value"))
        if (value is None or value <= 0 or value != value.to_integral_value() or fact.get("unit") not in _COUNT_UNITS
                or not isinstance(fact.get("scope", {}), dict)
                or fact.get("scope", {}).get("role") in {"analytical_parameter", "superseded_input"}):
            continue
        eligible.append({**fact, "value": value})
    pairs = set()
    for per_item in eligible:
        for equipment in eligible:
            if per_item["id"] != equipment["id"] and equipment["unit"] == "each" and _repetition_relation(per_item, equipment):
                pairs.add((per_item["id"], equipment["id"]))
    return [{"per_item_ref": per_item, "equipment_count_ref": equipment} for per_item, equipment in sorted(pairs)]


def has_quantity_repetition(inputs, messages=None):
    """Literal repetition intent when the equipment count is still unknown.

    This weaker signal authorizes no arithmetic; quantity_relations supplies
    the separate proof needed for nonnull count references.
    """
    texts = [fact.get("user_text") for fact in inputs if isinstance(fact, dict)] if isinstance(inputs, list) else []
    if isinstance(messages, list):
        texts.extend(message for message in messages if isinstance(message, str))
    for text in dict.fromkeys(text for text in texts if isinstance(text, str)):
        for relation in _REPETITION.finditer(text):
            start, end = _clause_bounds(text, relation.start())
            clause = text[start:end]
            qualifier = re.search(r"\b(?:but|however)\b", text[relation.end():end], re.I)
            if qualifier:
                clause = text[start:relation.end() + qualifier.start()]
            if _UNSAFE_LIST.search(clause):
                continue
            target = text[relation.end():end].strip()
            words = _noun_words(target)
            if words - {"hour", "hr", "h", "minute", "second", "day"}:
                return True
            if relation.group().casefold() in {"each", "for each"} and (_REQUEST_VERB.search(clause) or not target):
                return True
    return False


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _digest(value: Any) -> str:
    return hashlib.sha256(json_numbers(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(text: str) -> Any:
    return json.loads(text, parse_float=Decimal, parse_int=Decimal,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite evidence")))


def _tokens(text: Any) -> list[str]:
    # Generic normalization only: no equipment/part aliases or source answers.
    result = []
    for word in re.findall(r"[^\W_]+", str(text).casefold(), re.UNICODE):
        if word.isdecimal() or word in _STOP:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word not in result:
            result.append(word)
    return result


_ENTITY_BOUNDARY = re.compile(r"\b(?:and|or|for|per|each|every|on|in|at|with|from|to|using|across|plus|but|please|what|how|cost|budget|overall)\b|[,;.!?\n]", re.I)
_GENERIC_MATERIAL_WORDS = _COUNT_UNITS | {"ml", "l", "litre", "liter", "millilitre", "milliliter", "ea", "pc", "boxe", "of", "spare", "material", "component", "number", "none",
                                        "new", "old", "some", "all", "both", "only", "additional", "necessary", "needed", "requested"}


def _material_tokens(text):
    return set(_tokens(text)) - _GENERIC_MATERIAL_WORDS


def _literal_material_phrase(quantity):
    """Read only the noun phrase attached to this actual numeric operand."""
    text = quantity.get("user_text", "")
    spans = _literal_count_spans(quantity)
    if not spans:
        return None
    scope = quantity.get("scope", {})
    claimed = scope.get("entity_text")
    if claimed is not None and not isinstance(claimed, str):
        return None
    phrases = []
    for _, end in spans:
        tail = text[end:end + 140].lstrip()
        if claimed:
            # entity_text is an extraction, never a replacement supplied by a
            # model: it must occur directly after the literal numeric operand.
            if not tail.casefold().startswith(claimed.strip().casefold()):
                continue
            tail = claimed.strip()
        phrase = _ENTITY_BOUNDARY.split(tail, maxsplit=1)[0].strip()
        if phrase:
            phrases.append(phrase)
    if not phrases:
        return None
    # A missing span cannot join unrelated same-valued operands into one broad
    # bag of words. An explicit narrow extraction can select its literal span.
    if len({frozenset(_material_tokens(phrase)) for phrase in phrases}) != 1:
        return None
    return phrases[0]


def _component_quantity_matches(evidence, quantity, facts=None, *, allow_customer_supplied=False):
    if not isinstance(evidence, dict) or not isinstance(quantity, dict):
        return False
    if quantity.get("kind") == "request_quantity":
        if not isinstance(quantity.get("scope"), dict):
            return False
        reference = quantity.get("scope", {}).get("per_item_ref")
        if not isinstance(reference, str) or not isinstance(facts, dict) or reference not in quantity.get("parents", []) or reference not in facts:
            return False
        literal = facts[reference]
        if not isinstance(literal, dict) or literal.get("unit") != quantity.get("unit"):
            return False
    else:
        literal = quantity
    if (literal.get("kind") != "user_input" or _number(literal.get("value")) is None or not isinstance(literal.get("scope", {}), dict)
            or literal.get("scope", {}).get("role") in {"analytical_parameter", "superseded_input"}):
        return False
    if not allow_customer_supplied and literal.get("scope", {}).get("customer_supplied") is True:
        return False
    phrase = _literal_material_phrase(literal)
    scope = evidence.get("scope", {})
    if not phrase or not isinstance(scope, dict) or not isinstance(scope.get("entity_tokens"), list):
        return False
    equipment = _material_tokens(str(scope.get("equipment") or "") + " " + str(scope.get("equipment_family") or ""))
    source = _material_tokens(" ".join(str(token) for token in scope["entity_tokens"])) - equipment
    requested = _material_tokens(phrase) - equipment
    # A shared model/catalogue qualifier cannot identify a requested component:
    # different parts fitted to one model can carry the same identifier.
    # Identifier-only requests remain ambiguous rather than pricing an accessory.
    return any(not any(character.isdigit() for character in token)
               for token in source & requested)


def material_quantity_matches(rate, quantity, facts=None):
    """Conservative literal grounding for a purchase price and user quantity.

    Equipment/unit words, broad labels and request history cannot provide a
    component match. This does not certify equipment compatibility or price.
    """
    if not isinstance(rate, dict) or not isinstance(quantity, dict):
        return False
    if rate.get("kind") != "unit_price" or not isinstance(rate.get("unit"), str) or not rate["unit"].startswith("money/"):
        return False
    denominator = rate["unit"].split("/", 1)[1]
    return (denominator != "hour" and quantity.get("unit") == denominator
            and _component_quantity_matches(rate, quantity, facts))


def duration_scaling_matches(task, requested, observed, facts=None):
    """Whether this reported task/count can support a conditional work duration.

    Only a single task and a positive count from that same report qualify.
    The requested literal component must match the observed component; owned
    materials remain eligible because installation work is distinct from a
    purchase. Source checking and immutable registry checks remain backend-owned.
    """
    if any(not isinstance(fact, dict) for fact in (task, requested, observed)):
        return False
    task_value, request_value, observed_value = (_number(fact.get("value")) for fact in (task, requested, observed))
    if (task.get("kind") != "task_duration" or task.get("unit") != "hour" or task_value is None or task_value < 0
            or requested.get("kind") not in {"user_input", "request_quantity"} or requested.get("unit") not in _COUNT_UNITS
            or request_value is None or request_value < 0 or request_value != request_value.to_integral_value()
            or observed.get("kind") != "quantity" or observed.get("unit") != requested.get("unit")
            or observed_value is None or observed_value <= 0
            or not task.get("report_id") or task.get("report_id") != observed.get("report_id")
            or any(not isinstance(fact.get("scope"), dict) for fact in (task, observed))):
        return False
    required, available = observed["scope"].get("entity_tokens"), task["scope"].get("entity_tokens")
    if (not isinstance(required, list) or not isinstance(available, list) or not required
            or any(not isinstance(token, str) for token in required + available)
            or not set(required).issubset(available)):
        return False
    if "inspect" in str(observed["scope"].get("activity", "")) and "inspect" not in str(task.get("label", "")).casefold():
        return False
    literal = requested
    if requested.get("kind") == "request_quantity":
        reference = requested.get("scope", {}).get("per_item_ref")
        literal = facts.get(reference, {}) if isinstance(facts, dict) else {}
    if not isinstance(literal, dict):
        return False
    requested_modes = set()
    for start, _ in _literal_count_spans(literal):
        left, right = _clause_bounds(literal["user_text"], start)
        requested_modes.update(_work_modes(literal["user_text"][left:right]))
    task_modes = _work_modes(str(task.get("label", "")))
    if len(requested_modes) == 1 and task_modes and not requested_modes.intersection(task_modes):
        return False
    row_quantities = _task_row_quantity_ids(task, facts)
    if row_quantities and observed.get("id") not in row_quantities:
        return False
    return _component_quantity_matches(observed, requested, facts, allow_customer_supplied=True)


def _work_modes(text):
    """Only explicit generic inspection/replacement words establish intent."""
    modes = set()
    if re.search(r"\breplac(?:e|es|ed|ing|ement|ements)\b", text, re.I):
        modes.add("replacement")
    if re.search(r"\binspect(?:s|ed|ing|ion|ions)?\b", text, re.I):
        modes.add("inspection")
    return modes


def _task_row_quantity_ids(task, facts):
    """A quantity printed on the task row takes precedence over bill counts."""
    if not isinstance(facts, dict):
        return set()
    task_sources = set(task.get("source_ids", []))
    return {fact["id"] for fact in facts.values()
            if isinstance(fact, dict) and isinstance(fact.get("scope"), dict)
            and fact.get("kind") == "quantity" and fact.get("unit") not in {"money", "hour"}
            and fact.get("report_id") == task.get("report_id")
            and fact["scope"].get("type") == "reported_quantity"
            and (value := _number(fact.get("value"))) is not None and value > 0
            and task_sources.intersection(fact.get("source_ids", []))}


def _lines(refs: list[dict]) -> list[dict]:
    found = set()
    for ref in refs:
        start, end = ref.get("line_start"), ref.get("line_end")
        if (isinstance(start, (int, Decimal)) and isinstance(end, (int, Decimal))
                and int(start) == start and int(end) == end and 0 < start <= end):
            found.add((int(start), int(end)))
    return [{"line_start": start, "line_end": end} for start, end in sorted(found)]


def _source(record: dict, refs: list[dict]) -> dict | None:
    refs = _lines(refs)
    raw = record.get("source", {})
    sha = raw.get("sha256", "")
    if not refs or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        return None
    maximum = _number(raw.get("line_count"))
    if maximum is not None and any(ref["line_end"] > maximum for ref in refs):
        return None
    item = {"report_id": record["report_id"], "source_file": raw.get("path"),
            "source_sha256": sha, "lines": refs, "roles": ["service_observation"]}
    item["id"] = "SE" + _digest(item)[:16]
    return item


def _base_scope(record: dict) -> dict:
    equipment = record.get("equipment", {})
    return {"country": record.get("country"), "site": record.get("site"),
            "equipment_family": equipment.get("family"), "equipment": equipment.get("description"),
            "reported_visit_date": record.get("visit_date", {}).get("value")}


def _visit_activity(label):
    """Readable groups for original task descriptions, without changing scope."""
    matches = [name for name, pattern in (
        ("Preparation and safety", r"induct|safety|briefing|lock[ -]?out|\bloto\b|isolat|prepar"),
        ("Testing", r"test|commission|operational\s+check|functional|calibrat|tuning"),
        ("Inspections", r"inspect|assessment"),
        ("Cleanup and documentation", r"clean[ -]?up|documentation|debrief|report\s+writ|handover"),
    ) if re.search(pattern, label, re.I)]
    # One reported row remains one quantity. Mixed activities cannot be split
    # or assigned wholly to whichever label happened to match first.
    return matches[0] if len(matches) == 1 else "Other work"


def _observations(record: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Normalize existing reported numbers; leave unsupported bases unavailable."""
    facts, sources, warnings = {}, {}, []
    scope = _base_scope(record)
    equipment_tokens = set(_tokens(str(scope.get("equipment")) + " " + str(scope.get("equipment_family"))))

    def add(label, value, unit, kind, refs, *, currency=None, extra=None, assumptions=None):
        value = _number(value)
        source = _source(record, refs)
        if value is None or value < 0 or source is None:
            return None
        metadata = {**scope, **(extra or {})}
        item = {"label": label, "value": value, "unit": unit, "currency": currency,
                "kind": kind, "report_id": record["report_id"], "source_ids": [source["id"]],
                "parents": [], "assumptions": list(assumptions or []), "scope": metadata}
        item["id"] = "SF" + _digest(item)[:18]
        facts[item["id"]] = item
        sources[source["id"]] = source
        return item

    def entity(description):
        return sorted(set(_tokens(description)) - equipment_tokens) or _tokens(description)

    # A printed visit total remains distinct from components and task durations.
    for field, label in (("reported_total", "Reported whole-visit total"),
                         ("reported_subtotal", "Reported whole-visit subtotal"),
                         ("reported_vat", "Reported visit tax amount")):
        item = record.get("cost", {}).get(field)
        if isinstance(item, dict):
            add(label, item.get("amount"), "money", "visit_total", item.get("evidence", []),
                currency=item.get("currency"), extra={"type": field, "context": "original_visit",
                    "tax_basis": record.get("cost", {}).get("tax_basis", "not_stated"),
                    "validated": not any(c.get("status") in {"mismatch", "currency_mismatch"}
                        for c in record.get("cost", {}).get("checks", [])
                        if c.get("kind") in {"line_sum_vs_printed", "subtotal_plus_vat"})})
    for field, role in (("on_site", "labour_hours"), ("travel", "travel_hours")):
        item = record.get("hours", {}).get(field, {})
        if item.get("status") == "reported":
            add("Reported visit " + field.replace("_", " ") + " hours", item.get("value"), "hour", "quantity",
                item.get("evidence", []), extra={"type": "visit_duration", "role": role, "context": "original_visit"})

    # Keep every task row; only an individual task can be scaled explicitly.
    for item in record.get("tasks", []):
        add(item.get("description", "Reported task"), item.get("hours"), "hour", "task_duration",
            item.get("evidence", []), extra={"type": "task_duration", "role": "labour_hours",
                "entity_tokens": entity(item.get("description", "")), "context": "original_visit",
                "visit_task_sum_status": record.get("hours", {}).get("task_sum_check", {}).get("status", "unknown")})

    rows = record.get("cost", {}).get("line_items", []) + record.get("materials", [])
    material_rows = {id(row) for row in record.get("materials", [])}
    conflicts = {}
    for row in rows:
        key = (tuple(_tokens(row.get("description", ""))), _number(row.get("quantity")), row.get("currency"))
        amount, rate = _number(row.get("amount")), _number(row.get("unit_rate"))
        if amount is not None and rate is not None:
            conflicts.setdefault(key, set()).add((amount, rate))
    for row in rows:
        description = str(row.get("description", "Reported component"))
        refs = row.get("header_evidence", []) + row.get("evidence", [])
        q, rate, amount = (_number(row.get(key)) for key in ("quantity", "unit_rate", "amount"))
        currency = row.get("currency")
        qunit, runit = _unit(row.get("quantity_unit")), _unit(row.get("rate_unit"))
        category = row.get("category")
        key = (tuple(_tokens(description)), q, currency)
        contradictory = len(conflicts.get(key, ())) > 1
        rate_matches = (q is not None and q > 0 and rate is not None and rate >= 0 and amount is not None
                        and (q * rate).quantize(Decimal(".01"), rounding=ROUND_HALF_UP) == amount)
        valid_currency = currency in CURRENCIES and row.get("unit_rate_currency") == currency
        # The parser populates unit_rate from its recognized rate/cost column.
        # A literal quantity unit plus a coherent priced amount establishes the
        # denominator even when that column is labelled simply "Rate".
        no_rate_denominator = not str(row.get("rate_unit") or "").strip()
        inferred_quantity_basis = bool(no_rate_denominator and qunit and row.get("header_evidence")
                                       and rate_matches and valid_currency and not contradictory)
        denominator = runit or (qunit if inferred_quantity_basis else None)
        description_hour_basis = bool(not denominator and no_rate_denominator
            and not str(row.get("quantity_unit") or "").strip()
            and category in {"travel", "labour"} and row.get("header_evidence")
            and _description_hours(description, q)
            and rate_matches and valid_currency and not contradictory)
        independent_hour_refs = []
        if (not denominator and not description_hour_basis and no_rate_denominator
                and not str(row.get("quantity_unit") or "").strip()
                and row.get("rate_basis") == "unit_rate_header" and row.get("header_evidence")
                and rate_matches and valid_currency and not contradictory):
            independent_hour_refs = _independent_hour_refs(record, category, q, refs)
        implied_item_basis = bool(not denominator and no_rate_denominator
            and not str(row.get("quantity_unit") or "").strip()
            and row.get("rate_basis") == "unit_rate_header" and row.get("header_evidence")
            and (category == "parts" or id(row) in material_rows)
            and q is not None and q > 0 and q == q.to_integral_value()
            and rate is not None and rate > 0 and row.get("customer_supplied") is not True
            and rate_matches and valid_currency and not contradictory)
        row_assumptions = []
        if description_hour_basis:
            denominator = "hour"
            row_assumptions.append("The hourly denominator is established by the matching duration in this same row's description and its coherent priced amount.")
        if independent_hour_refs:
            denominator = "hour"
            refs = refs + independent_hour_refs
            row_assumptions.append("The hourly denominator is established by an exact independent reported visit-hour statement matching this row's quantity, with a coherent explicit-currency unit-rate amount.")
        if implied_item_basis:
            denominator = "each"
            row_assumptions.append("The unit-rate/cost header and coherent whole-number quantity are interpreted as each listed item, not the members inside a set or package.")
        # An explicit per-each rate counts the item described by its quantity
        # column (for example one whole set), not the members inside that item.
        if runit == "each" and qunit in _COUNT_UNITS:
            denominator = qunit
        compatible = (not (qunit and denominator and qunit != denominator)
            and not (str(row.get("quantity_unit") or "").strip() and qunit is None)
            and not (str(row.get("rate_unit") or "").strip() and runit is None))
        hourly = denominator == "hour"
        valid = bool(rate_matches and valid_currency and not contradictory and denominator and compatible)
        if not hourly and (rate is None or rate <= 0 or row.get("customer_supplied") is True):
            valid = False
        if denominator in _COUNT_UNITS and q is not None and q != q.to_integral_value():
            valid = False
        group = _digest([record["report_id"], entity(description), q, currency])[:18]
        extra = {"type": "component", "description": description, "category": category,
                 "entity_tokens": entity(description), "group": group, "context": "original_visit",
                 "quantity_unit_reported": row.get("quantity_unit"), "rate_unit_reported": row.get("rate_unit"), "validated": valid,
                 "unit_basis": denominator or "unconfirmed", "denominator_compatible": compatible}
        if implied_item_basis:
            extra["denominator_basis"] = "coherent_whole_item_quantity_under_unit_rate_header"
        if description_hour_basis:
            extra["denominator_basis"] = "matching_same_row_hour_quantity"
        if independent_hour_refs:
            extra["denominator_basis"] = "matching_reported_hour_quantity"
        if row.get("customer_supplied") is True:
            extra["customer_supplied"] = True
        quantity = None
        quantity_unit = qunit or denominator
        if q is not None and q >= 0:
            quantity = add(description + " — reported quantity", q, quantity_unit or "number", "quantity", refs,
                           extra={**extra, "type": "reported_quantity", "role": "travel_hours" if category == "travel" else "labour_hours" if hourly else "observed_quantity"},
                           assumptions=row_assumptions)
        if rate is not None:
            add(description + " — reported unit rate", rate, "money/" + denominator if denominator else "money",
                "hourly_rate" if hourly else "unit_price", refs, currency=row.get("unit_rate_currency") or currency,
                extra={**extra, "quantity_ref": quantity["id"] if quantity else None}, assumptions=row_assumptions)
        if amount is not None:
            add(description + " — reported component amount", amount, "money", "quantity", refs, currency=currency,
                extra={**extra, "type": "reported_component_amount", "validated": not contradictory,
                       "quantity_ref": quantity["id"] if quantity else None}, assumptions=row_assumptions)
        if contradictory:
            warnings.append("Conflicting duplicate component prices in " + record["report_id"] + ": " + description)
        elif rate is not None and not valid:
            warnings.append("Reported rate is not an eligible individual/hourly calculation basis in " + record["report_id"] + ": " + description)

    # Parser-normalized service quantities can also support duration comparisons.
    # Their activity and unit are retained; plate/set counts are never cloth counts.
    for item in record.get("service_quantities", []) + record.get("replacement_quantities", []):
        quantity_unit = _unit(item.get("quantity_unit"))
        label = str(item.get("item") or "Reported item").replace("_", " ")
        add(label + " — reported activity quantity", item.get("quantity"), quantity_unit or "number", "quantity", item.get("evidence", []),
            extra={"type": "reported_quantity", "role": "observed_quantity", "description": label,
                   "entity_tokens": entity(label), "activity": item.get("activity"), "context": "original_visit",
                   "quantity_unit_reported": item.get("quantity_unit")})
    return list(facts.values()), list(sources.values()), sorted(set(warnings))


class ServiceWorkspace:
    """Per-conversation evidence registry backed by a source-bound FTS sidecar."""

    def __init__(self, store_path, index_path=None, inputs=None, source_dir=None):
        self.store_path = Path(store_path).resolve()
        self.source_dir = Path(source_dir).resolve() if source_dir is not None else None
        self.index_path = Path(index_path or self.store_path.with_name("service-evidence.sqlite")).resolve()
        if self.index_path == self.store_path:
            raise ValueError("The service index must be a separate sidecar")
        self.facts: dict[str, dict] = {}
        self.sources: dict[str, dict] = {}
        self._fact_hashes = {}
        self._last_query = ""
        self._binding = self._current_binding()
        self._ensure_index()
        self._base_warnings = self._check_originals()
        for index, item in enumerate(inputs or [], 1):
            if not isinstance(item, dict):
                raise ValueError("Request facts must be objects")
            value = _number(item.get("value"))
            if value is None or value < 0 or item.get("unit") not in _QUANTITY_UNITS | {"hour", "number", "percent"}:
                raise ValueError("Request facts require finite nonnegative values and explicit units")
            if item.get("unit") in _COUNT_UNITS and value != value.to_integral_value():
                raise ValueError("Individual item counts must be whole numbers")
            identifier = item.get("id") or f"U{index:03}"
            if not isinstance(identifier, str) or not re.fullmatch(r"U[A-Za-z0-9_-]{1,60}", identifier) or identifier in self.facts:
                raise ValueError("Request facts require unique U-prefixed identifiers")
            fact = {"id": identifier, "label": str(item.get("label", "User-provided input")), "value": value,
                    "unit": item["unit"], "currency": None, "kind": "user_input", "report_id": None,
                    "source_ids": [], "parents": [], "assumptions": list(item.get("assumptions", [])),
                    "scope": deepcopy(item.get("scope", {})), "user_text": str(item.get("user_text", ""))}
            self._register(fact)

    def _current_binding(self):
        wal = Path(str(self.store_path) + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Checkpoint the prepared store before indexing")
        return {"schema_version": SCHEMA_VERSION, "method_version": METHOD_VERSION,
                "store_sha256": _file_hash(self.store_path), "method_sha256": _file_hash(Path(__file__))}

    def _check_binding(self):
        if self._current_binding() != self._binding:
            raise ValueError("Prepared service evidence changed; start a new workspace")
        return self._check_originals()

    def _connect(self):
        connection = sqlite3.connect(self.index_path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _content_hash(connection):
        digest = hashlib.sha256()
        for table, fields in (("reports", "report_id,json"), ("observations", "id,json"),
                              ("sources", "id,json"), ("observations_fts", "id,text")):
            for row in connection.execute(f"SELECT {fields} FROM {table} ORDER BY 1"):
                digest.update(json_numbers(list(row)).encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()

    def _ensure_index(self):
        valid = False
        if self.index_path.is_file():
            try:
                with closing(self._connect()) as connection:
                    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
                    valid = (_load(metadata.get("binding", "{}")) == self._binding
                             and self._content_hash(connection) == metadata.get("content_sha256")
                             and connection.execute("SELECT count(*) FROM observations").fetchone()[0]
                             == connection.execute("SELECT count(*) FROM observations_fts").fetchone()[0])
            except (sqlite3.Error, ValueError, OSError, KeyError):
                valid = False
        if valid:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.index_path.name + ".", suffix=".tmp", dir=self.index_path.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            with closing(sqlite3.connect(temporary)) as connection, KnowledgeStore(self.store_path) as store:
                connection.executescript("""
                    CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                    CREATE TABLE reports(report_id TEXT PRIMARY KEY,json TEXT NOT NULL);
                    CREATE TABLE sources(id TEXT PRIMARY KEY,json TEXT NOT NULL);
                    CREATE TABLE observations(id TEXT PRIMARY KEY,report_id TEXT NOT NULL,json TEXT NOT NULL);
                    CREATE INDEX observations_report ON observations(report_id);
                    CREATE VIRTUAL TABLE observations_fts USING fts5(id UNINDEXED,text,tokenize='unicode61');
                """)
                for raw in store.connection.execute("SELECT report_id,source_file,source_sha256,json FROM services ORDER BY report_id"):
                    record = _load(raw[3])
                    if (raw[0] != record.get("report_id") or raw[1] != record.get("source", {}).get("path")
                            or raw[2] != record.get("source", {}).get("sha256")):
                        raise ValueError("Prepared service source identity disagrees with indexed columns")
                    facts, sources, warnings = _observations(record)
                    manifest = {"report_id": record["report_id"], "source": record.get("source", {}),
                                "scope": _base_scope(record), "warnings": warnings, "fact_ids": [f["id"] for f in facts]}
                    connection.execute("INSERT INTO reports VALUES(?,?)", (record["report_id"], json_numbers(manifest)))
                    connection.executemany("INSERT OR IGNORE INTO sources VALUES(?,?)", [(s["id"], json_numbers(s)) for s in sources])
                    for fact in facts:
                        connection.execute("INSERT INTO observations VALUES(?,?,?)", (fact["id"], record["report_id"], json_numbers(fact)))
                        search_text = " ".join(str(v) for v in (record["report_id"], record.get("country"), record.get("site"),
                            record.get("equipment", {}).get("description"), record.get("equipment", {}).get("family"), fact["label"]))
                        connection.execute("INSERT INTO observations_fts VALUES(?,?)", (fact["id"], " ".join(_tokens(search_text))))
                connection.execute("INSERT INTO metadata VALUES('binding',?)", (json_numbers(self._binding),))
                connection.execute("INSERT INTO metadata VALUES('content_sha256',?)", (self._content_hash(connection),))
                connection.commit()
            if self._current_binding() != self._binding:
                raise ValueError("Prepared store changed during service index build")
            os.replace(temporary, self.index_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _check_originals(self):
        warnings = []
        missing = []
        with closing(self._connect()) as connection:
            for row in connection.execute("SELECT json FROM reports ORDER BY report_id"):
                report = _load(row[0]); source = report["source"]
                path = Path(source.get("path") or "")
                if self.source_dir is not None:
                    path = self.source_dir / PureWindowsPath(str(path)).name
                if path.is_file():
                    if _file_hash(path) != source.get("sha256"):
                        raise ValueError("Original service source changed: " + report["report_id"])
                else:
                    missing.append(report["report_id"])
        if missing:
            warnings.append(f"Original sources unavailable for {len(missing)} reports; observations remain bound to prepared source hashes.")
        return warnings

    def _register(self, fact):
        self.facts[fact["id"]] = deepcopy(fact)
        self._fact_hashes[fact["id"]] = _digest(fact)

    def _envelope(self, facts, warnings=None, **extra):
        ids = {sid for fact in facts for sid in fact.get("source_ids", [])}
        return {"facts": deepcopy(facts), "sources": [deepcopy(self.sources[sid]) for sid in sorted(ids) if sid in self.sources],
                "warnings": sorted(set(self._base_warnings + list(warnings or []))), **extra}

    def _read_facts(self, ids, connection):
        result = []
        for identifier in ids:
            row = connection.execute("SELECT json FROM observations WHERE id=?", (identifier,)).fetchone()
            if row is None:
                continue
            fact = _load(row[0]); self._register(fact); result.append(fact)
            for sid in fact["source_ids"]:
                source = connection.execute("SELECT json FROM sources WHERE id=?", (sid,)).fetchone()
                if source:
                    decoded = _load(source[0])
                    decoded["lines"] = _lines(decoded["lines"])
                    self.sources[sid] = decoded
        return result

    def search(self, query, country=None, equipment=None, limit=8):
        try:
            self._base_warnings = self._check_binding()
            if not isinstance(query, str) or not query.strip() or len(query) > 6000:
                raise ValueError("Search needs a nonempty bounded query")
            if type(limit) is not int or not 1 <= limit <= 30:
                raise ValueError("Search limit must be between 1 and 30")
            self._last_query = query
            words = _tokens(query)
            if not words:
                return self._envelope([], ["No meaningful service search terms were provided."])
            match = " OR ".join('"' + word.replace('"', '""') + '"' for word in words[:60])
            with closing(self._connect()) as connection:
                rows = connection.execute("SELECT o.json,bm25(observations_fts) AS rank FROM observations_fts "
                    "JOIN observations o ON o.id=observations_fts.id WHERE observations_fts MATCH ? ORDER BY rank,o.id LIMIT ?",
                    (match, max(60, limit * 12))).fetchall()
                candidates = [_load(row[0]) for row in rows]
                candidates.sort(key=lambda f: self._rank(f, query, country, equipment))
                selected, groups = [], set()
                for fact in candidates:
                    group = fact["scope"].get("group", fact["id"])
                    if group in groups:
                        continue
                    groups.add(group)
                    related = [f for f in candidates if f["scope"].get("group", f["id"]) == group]
                    related.sort(key=lambda f: ({"unit_price": 0, "hourly_rate": 0, "quantity": 1}.get(f["kind"], 2), f["id"]))
                    available = MAX_SEARCH_FACTS - len(selected)
                    selected.extend(related[:available])
                    if len(groups) >= limit or len(selected) >= MAX_SEARCH_FACTS:
                        break
                facts = self._read_facts([f["id"] for f in selected], connection)
                warnings = []
                for report_id in sorted({f["report_id"] for f in facts}):
                    row = connection.execute("SELECT json FROM reports WHERE report_id=?", (report_id,)).fetchone()
                    warnings.extend(_load(row[0])["warnings"])
            return self._envelope(facts, warnings, query=query)
        except (ValueError, OSError, sqlite3.Error, InvalidOperation) as exc:
            return self._envelope([], [str(exc)], error=str(exc))

    def travel_rates(self, country):
        """Load the complete validated country cohort, independent of relevance ranking."""
        data = {"country": country.strip() if isinstance(country, str) else None,
                "rate_ids": [], "selection": "country_metadata"}
        try:
            self._base_warnings = self._check_binding()
            if not isinstance(country, str) or not country.strip() or len(country) > 200:
                raise ValueError("A nonempty country is required for historical travel rates")
            wanted = country.strip().casefold()
            with closing(self._connect()) as connection:
                identifiers = []
                for row in connection.execute("SELECT id,json FROM observations ORDER BY report_id,id"):
                    fact = _load(row[1]); scope = fact.get("scope", {})
                    if (str(scope.get("country") or "").strip().casefold() == wanted
                            and fact.get("kind") == "hourly_rate" and fact.get("unit") == "money/hour"
                            and scope.get("category") == "travel" and scope.get("validated") is True):
                        identifiers.append(row[0])
                facts = self._read_facts(identifiers, connection)
            data["rate_ids"] = [fact["id"] for fact in facts]
            warnings = [] if facts else ["No validated historical hourly travel rates were found for this country."]
            return self._envelope(facts, warnings, status="ok" if facts else "insufficient_evidence", data=data)
        except (ValueError, OSError, sqlite3.Error, InvalidOperation) as exc:
            return self._envelope([], [str(exc)], status="error", error=str(exc), data=data)

    @staticmethod
    def _rank(fact, query, country=None, equipment=None):
        wanted = set(_tokens(query)); label = set(_tokens(fact["label"]))
        scope = fact["scope"]
        score = 10 * len(wanted & label)
        score += 2 if country and str(scope.get("country", "")).casefold() == str(country).casefold() else 0
        score += 2 * len(set(_tokens(equipment or "")) & set(_tokens(str(scope.get("equipment")))))
        score += {"unit_price": 2, "hourly_rate": 1, "task_duration": 1}.get(fact["kind"], 0)
        return -score, fact["report_id"], fact["id"]

    def read_report(self, report_id):
        try:
            self._base_warnings = self._check_binding()
            with closing(self._connect()) as connection:
                row = connection.execute("SELECT json FROM reports WHERE report_id=?", (report_id,)).fetchone()
                if row is None:
                    raise ValueError("Unknown service report identifier")
                report = _load(row[0])
                # Register every observation; show a compact selected context.
                facts = self._read_facts(report["fact_ids"], connection)
            facts.sort(key=lambda f: self._rank(f, self._last_query))
            anchors = [f for f in facts if f["scope"].get("type") in {"reported_total", "visit_duration"}]
            anchor_ids = {f["id"] for f in anchors}
            chosen = anchors + [f for f in facts if f["id"] not in anchor_ids][:MAX_REPORT_FACTS - len(anchors)]
            warnings = report["warnings"][:]
            if len(facts) > len(chosen):
                warnings.append(f"Showing {len(chosen)} of {len(facts)} observations; refine search for other components.")
            return self._envelope(chosen, warnings, report_id=report_id, available_fact_count=len(facts), scope=report["scope"])
        except (ValueError, OSError, sqlite3.Error, InvalidOperation) as exc:
            return self._envelope([], [str(exc)], error=str(exc))

    def _ref(self, identifier):
        fact = self.facts.get(identifier)
        if fact is None or _digest(fact) != self._fact_hashes.get(identifier):
            raise ValueError("Unknown or modified evidence reference: " + str(identifier))
        if _number(fact.get("value")) is None:
            raise ValueError("Evidence value is not finite")
        return fact

    def visit_labour_comparison(self, task_ref, rate_ref):
        """Original same-report labour alternatives, never a customer subtotal.

        The complete listed task population is read independently of retrieval
        limits. Missing task hours prevent a whole-visit itemized calculation.
        Both itemized and reported/billed bases retain their original scope.
        """
        try:
            self._base_warnings = self._check_binding()
            task, rate = self._ref(task_ref), self._ref(rate_ref)
            report_id = task.get("report_id")
            if (task.get("kind") != "task_duration" or not report_id or rate.get("report_id") != report_id
                    or rate.get("kind") != "hourly_rate" or rate.get("unit") != "money/hour"
                    or rate.get("scope", {}).get("category") != "labour" or rate["scope"].get("validated") is not True
                    or rate.get("currency") not in CURRENCIES):
                raise ValueError("Historical visit labour requires a task and validated labour rate from the same report")
            with KnowledgeStore(self.store_path) as store:
                row = store.connection.execute("SELECT json FROM services WHERE report_id=?", (report_id,)).fetchone()
                record = _load(row[0]) if row else {}
            rows = record.get("tasks", [])
            if len(rows) < 2:
                return self._envelope([], status="unavailable", data={"reason": "no_additional_tasks"})
            if any(value is None or value < 0 for value in (_number(row.get("hours")) for row in rows)):
                return self._envelope([], ["Some original task hours are missing; no complete itemized visit total is calculated."],
                                      status="unavailable", data={"reason": "incomplete_task_hours"})
            with closing(self._connect()) as connection:
                facts = self._read_facts([row[0] for row in connection.execute(
                    "SELECT id FROM observations WHERE report_id=? ORDER BY id", (report_id,))], connection)
            tasks = [fact for fact in facts if fact["kind"] == "task_duration"]
            if len(tasks) != len(rows) or task_ref not in {fact["id"] for fact in tasks}:
                return self._envelope([], ["The complete original task population is unavailable."],
                                      status="unavailable", data={"reason": "incomplete_task_population"})
            generated = []
            def derived(label, value, unit, parents):
                parents = [self._ref(reference) for reference in dict.fromkeys(parents)]
                fact = {"label": label, "value": value, "unit": unit,
                        "currency": rate["currency"] if unit == "money" else None,
                        "kind": "calculation", "report_id": report_id,
                        "source_ids": sorted({sid for parent in parents for sid in parent["source_ids"]}),
                        "parents": [parent["id"] for parent in parents],
                        "assumptions": ["Original source-country visit labour comparison only; not resized, added to, or substituted for the requested service subtotal."],
                        "scope": {**_base_scope(record), "type": "historical_labour_comparison",
                                  "operation": "visit_labour_comparison", "context": "original_visit", "comparison_only": True}}
                if unit == "hour": fact["scope"]["role"] = "labour_hours"
                fact["id"] = "C" + _digest(fact)[:18]
                self._register(fact); generated.append(fact)
                return fact
            grouped = {}
            for item in tasks:
                label = "Matched task" if item["id"] == task_ref else _visit_activity(item["label"])
                grouped.setdefault(label, []).append(item)
            with localcontext() as context:
                context.prec = 48
                groups = [{"label": label, "hours_ref": derived(label + " original hours",
                    sum((item["value"] for item in items), Decimal(0)), "hour", [item["id"] for item in items])["id"],
                    "task_refs": [item["id"] for item in items]} for label, items in grouped.items()]
                itemized = derived("Itemized original visit hours", sum((item["value"] for item in tasks), Decimal(0)),
                                   "hour", [item["id"] for item in tasks])
                other_tasks = [item for item in tasks if item["id"] != task_ref]
                other_hours = derived("Other original visit task hours", sum((item["value"] for item in other_tasks), Decimal(0)),
                                      "hour", [item["id"] for item in other_tasks])
                itemized_cost = derived("Itemized original visit labour cost",
                    (itemized["value"] * rate["value"]).quantize(Decimal(".01"), rounding=ROUND_HALF_UP),
                    "money", [itemized["id"], rate_ref])
                hours = [fact for fact in facts if fact.get("unit") == "hour" and fact.get("kind") == "quantity"
                         and fact.get("scope", {}).get("type") == "visit_duration" and self._role(fact) == "labour_hours"]
                billed = self.facts.get(rate.get("scope", {}).get("quantity_ref"), {})
                if billed.get("unit") == "hour" and billed.get("report_id") == report_id:
                    hours.append(billed)
                reported = []
                for value in sorted({fact["value"] for fact in hours}):
                    matching = [fact for fact in hours if fact["value"] == value]
                    kinds = {"reported" if fact["scope"].get("type") == "visit_duration" else "billed" for fact in matching}
                    label = "/".join(name for name in ("reported", "billed") if name in kinds)
                    hour_fact = derived(label.capitalize() + " original visit hours", value, "hour", [fact["id"] for fact in matching])
                    cost = derived(label.capitalize() + " original visit labour cost",
                        (value * rate["value"]).quantize(Decimal(".01"), rounding=ROUND_HALF_UP), "money", [hour_fact["id"], rate_ref])
                    reported.append({"basis": label, "hours_ref": hour_fact["id"], "amount_ref": cost["id"],
                                     "matches_itemized": value == itemized["value"]})
            self._check_binding()
            return self._envelope(generated, status="available", data={
                "report_id": report_id, **_base_scope(record), "matched_task_ref": task_ref, "rate_ref": rate_ref,
                "groups": groups, "task_refs": [item["id"] for item in tasks],
                "itemized_hours_ref": itemized["id"], "itemized_amount_ref": itemized_cost["id"],
                "other_hours_ref": other_hours["id"],
                "reported_bases": reported, "comparison_only": True})
        except (ValueError, OSError, sqlite3.Error, InvalidOperation, KeyError, TypeError) as exc:
            return self._envelope([], [str(exc)], error=str(exc))

    @staticmethod
    def _role(fact):
        role = fact.get("scope", {}).get("role")
        if role in _DURATION_ROLES:
            return role
        if fact.get("kind") == "user_input" and fact.get("unit") == "hour":
            words = set(_tokens(fact.get("label", "")))
            if "travel" in words or "journey" in words:
                return "travel_hours"
        return "labour_hours" if fact.get("kind") == "task_duration" else None

    def _ancestors(self, fact):
        result = set()
        for parent in fact.get("parents", []):
            result.add(parent); result.update(self._ancestors(self._ref(parent)))
        return result

    def _sum_overlap(self, facts):
        if any(f["kind"] == "visit_total" for f in facts):
            raise ValueError("A whole-visit total/subtotal cannot be added to components or alternative visits")
        for i, left in enumerate(facts):
            for right in facts[i + 1:]:
                if left["id"] == right["id"] or left["id"] in self._ancestors(right) or right["id"] in self._ancestors(left):
                    raise ValueError("Overlapping calculation parents cannot be summed")
                lscope, rscope = left["scope"], right["scope"]
                if (lscope.get("type") == "reported_component_amount" and lscope.get("group") in rscope.get("origin_groups", [])
                        or rscope.get("type") == "reported_component_amount" and rscope.get("group") in lscope.get("origin_groups", [])):
                    raise ValueError("An original component amount overlaps a calculation from its unit rate")
                if left["unit"] == "money" and {lscope.get("context"), rscope.get("context")} == {"original_visit", "conditional_scenario"}:
                    raise ValueError("Original component amounts cannot be silently added to a new conditional scenario")
                if lscope.get("coverage") and set(lscope["coverage"]) & set(rscope.get("coverage", [])):
                    raise ValueError("Overlapping charge/duration provenance cannot be summed")
                if left.get("report_id") and left["report_id"] == right.get("report_id"):
                    if (lscope.get("type") == "visit_duration" or rscope.get("type") == "visit_duration"):
                        raise ValueError("Visit hours overlap their individual task hours")
                    if lscope.get("group") and lscope.get("group") == rscope.get("group"):
                        raise ValueError("Duplicate component provenance cannot be summed")

    def calculate(self, operation, refs):
        try:
            self._base_warnings = self._check_binding()
            if not isinstance(refs, list) or not refs or len(refs) > 30 or any(not isinstance(x, str) for x in refs):
                raise ValueError("Calculations require a bounded list of registered fact IDs")
            if len(refs) != len(set(refs)):
                raise ValueError("Duplicate evidence references cannot be calculated together")
            facts = [self._ref(identifier) for identifier in refs]
            if any(fact.get("scope", {}).get("comparison_only") for fact in facts):
                raise ValueError("Historical comparison-only values cannot be reused in a requested budget calculation")
            assumptions = sorted({a for f in facts for a in f.get("assumptions", [])})
            scope = {"type": "calculation", "operation": operation, "context": "conditional_scenario"}
            scope["origin_groups"] = sorted({group for fact in facts for group in
                fact["scope"].get("origin_groups", [fact["scope"].get("group")]) if group})
            with localcontext() as context:
                context.prec = 48
                if operation == "quantity_product":
                    if len(facts) != 2:
                        raise ValueError("Quantity composition requires a per-item count and an equipment count")
                    per_item, equipment_count = facts
                    if (any(f["kind"] != "user_input" or f["value"] <= 0 or f["value"] != f["value"].to_integral_value() for f in facts)
                            or per_item["unit"] not in _COUNT_UNITS or equipment_count["unit"] != "each"):
                        raise ValueError("Quantity composition requires two positive whole literal user counts in compatible count units")
                    if not _repetition_relation(per_item, equipment_count):
                        raise ValueError("The actual user text must establish a per/each/every equipment relationship between these literal counts")
                    context.prec = max(context.prec, len(per_item["value"].as_tuple().digits) + len(equipment_count["value"].as_tuple().digits) + 2)
                    value = per_item["value"] * equipment_count["value"]
                    unit, currency = per_item["unit"], None
                    label = "Requested aggregate quantity: " + per_item["label"]
                    turns = [f.get("scope", {}).get("turn", -1) for f in facts]
                    turns = [int(turn) for turn in turns if isinstance(turn, (int, Decimal)) and not isinstance(turn, bool)]
                    scope.update({"type": "request_quantity", "context": "user_request", "role": "requested_quantity",
                                  "turn": max(turns, default=-1), "per_item_ref": per_item["id"], "equipment_count_ref": equipment_count["id"]})
                    assumptions.append("The literal user counts are combined under the user's explicit per/each/every equipment relationship; package contents are not expanded.")
                elif operation == "multiply":
                    if len(facts) != 2:
                        raise ValueError("Multiply requires exactly two references")
                    rate = next((f for f in facts if f["unit"].startswith("money/")), None)
                    other = next((f for f in facts if f is not rate), None)
                    if rate is None or other is None or not rate["scope"].get("validated") or rate["currency"] not in CURRENCIES:
                        raise ValueError("Multiplication needs a validated explicit-currency unit/hourly rate")
                    currency, unit = rate["currency"], "money"
                    denominator = rate["unit"].split("/", 1)[1]
                    if denominator != "hour":
                        if other["kind"] not in {"user_input", "request_quantity"} or other["unit"] != denominator:
                            raise ValueError("Price multiplication requires a user quantity in the exact reported denominator unit; no conversion is inferred")
                        if other["kind"] == "request_quantity":
                            self._ref(other["scope"].get("per_item_ref"))
                        if not material_quantity_matches(rate, other, self.facts):
                            raise ValueError("The literal user quantity does not identify the component priced by this historical material rate")
                        quantity_origin = other["scope"].get("per_item_ref", other["id"]) if other["kind"] == "request_quantity" else other["id"]
                        scope["coverage"] = ["material:" + quantity_origin + ":" + " ".join(rate["scope"].get("entity_tokens", []))]
                        assumptions.append("The historical price per reported unit is assumed applicable to the requested quantity; package contents, compatibility, current price and quantity discounts are unconfirmed.")
                    else:
                        if other["unit"] != "hour" or other["kind"] not in {"user_input", "task_duration", "quantity", "calculation"}:
                            raise ValueError("Hourly cost requires an explicit duration in hours")
                        if other["kind"] == "quantity" and other["scope"].get("type") not in {"visit_duration", "reported_quantity"}:
                            raise ValueError("Unknown duration scope")
                        role = self._role(other)
                        rate_role = "travel_hours" if rate["scope"].get("category") == "travel" else "labour_hours" if rate["scope"].get("category") == "labour" else None
                        if role and rate_role and role != rate_role:
                            raise ValueError("Travel and onsite labour hours cannot exchange billing rates")
                        if rate_role in _DURATION_ROLES and other.get("source_ids") and any(
                                fact.get("kind") == "user_input" and fact.get("unit") == "hour"
                                and fact.get("scope", {}).get("role") == rate_role
                                and self._ref(fact["id"])["value"] >= 0 for fact in self.facts.values()):
                            raise ValueError("Actual user-provided billing hours supersede original or source-derived hours for the same billing role")
                        scope["coverage"] = ["billed:" + str(rate_role) + ":" + token for token in other["scope"].get("coverage", ["fact:" + other["id"]])]
                        assumptions.append("The selected historical hourly rate is assumed applicable to the selected duration; this does not establish a current local quote.")
                    value = (rate["value"] * other["value"]).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
                    label = f"{rate['label']} x {other['label']}"
                elif operation == "scale_duration":
                    if len(facts) != 3:
                        raise ValueError("Duration scaling needs task hours, requested count and reported count")
                    task, requested, observed = facts
                    if requested.get("kind") == "request_quantity":
                        self._ref(requested["scope"].get("per_item_ref"))
                    if not duration_scaling_matches(task, requested, observed, self.facts):
                        raise ValueError("Duration scaling requires one reported task, its same-report component count, and a matching literal requested component")
                    value = task["value"] * requested["value"] / observed["value"]
                    currency, unit = None, "hour"
                    label = "Conditional scaled task duration: " + task["label"]
                    quantity_origin = requested["scope"].get("per_item_ref", requested["id"]) if requested["kind"] == "request_quantity" else requested["id"]
                    scope.update({"role": "labour_hours", "coverage": ["task:" + task["id"], "task:" + task["id"] + ":" + quantity_origin]})
                    assumptions.append("Linear scaling of this single reported task by individual count is hypothetical; fixed visit tasks and travel are excluded, and equipment/work-scope comparability is unconfirmed.")
                    if not _task_row_quantity_ids(task, self.facts):
                        assumptions.append("No task-row count is available; the selected same-report component count is assumed to be this task's denominator and that work-scope relationship is unconfirmed.")
                    if task["scope"].get("visit_task_sum_status") == "mismatch":
                        assumptions.append("The source visit's task total does not reconcile; this isolated reported task is a qualified observation only.")
                elif operation == "sum":
                    if len(facts) < 2 or len({f["unit"] for f in facts}) != 1 or facts[0]["unit"] not in {"money", "hour"}:
                        raise ValueError("Sum requires at least two compatible monetary amounts or durations")
                    unit = facts[0]["unit"]
                    currencies = {f["currency"] for f in facts}
                    if len(currencies) != 1 or (unit == "money" and next(iter(currencies)) not in CURRENCIES):
                        raise ValueError("Money sums require one explicit currency; currency conversion is unavailable")
                    if any(f["scope"].get("validated") is False for f in facts):
                        raise ValueError("Conflicting or unvalidated observations cannot form a subtotal")
                    self._sum_overlap(facts)
                    if unit == "hour" and len({self._role(f) for f in facts} - {None}) > 1:
                        raise ValueError("Different billing-hour scopes cannot form one cost duration")
                    currency = next(iter(currencies))
                    value = sum((f["value"] for f in facts), Decimal(0))
                    label = "Subtotal of " + ", ".join(f["label"] for f in facts)
                    scope["coverage"] = sorted({token for f in facts for token in f["scope"].get("coverage", ["fact:" + f["id"]])})
                    if unit == "hour":
                        scope["role"] = next((self._role(f) for f in facts if self._role(f)), None)
                else:
                    raise ValueError("Unsupported arithmetic operation; no exchange rates or unit conversion are available")
            if not value.is_finite():
                raise ValueError("Arithmetic produced a non-finite result")
            fact = {"label": label, "value": value, "unit": unit, "currency": currency,
                    "kind": "request_quantity" if operation == "quantity_product" else "calculation",
                    "report_id": None, "source_ids": sorted({sid for f in facts for sid in f["source_ids"]}),
                    "parents": refs[:], "assumptions": sorted(set(assumptions)), "scope": scope}
            if operation == "quantity_product":
                fact["user_text"] = "\n".join(dict.fromkeys(f["user_text"] for f in facts))
            fact["id"] = "C" + _digest(fact)[:18]
            self._register(fact)
            return self._envelope([fact])
        except (ValueError, OSError, sqlite3.Error, InvalidOperation, KeyError, TypeError) as exc:
            return self._envelope([], [str(exc)], error=str(exc))
