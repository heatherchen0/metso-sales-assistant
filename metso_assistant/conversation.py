"""Bounded conversation memory and provenance checks around model routing.

This module supplies no product facts, travel estimates, or service prices. It
preserves the customer's actual words for referential questions and accepts the
two easily defaulted service inputs only when the user has supplied them.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re


SERVICE_FIELDS = ("country", "equipment", "site", "task", "cloth_count",
                  "cloths_supplied", "round_trip_travel_hours")
_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
          "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
          "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19, "twenty": 20, "half": .5}
_NUMBER = r"(?:\d+(?:[.,]\d+)?|" + "|".join(_WORDS) + r")"
_HOURS = re.compile(rf"(?<![\w.\-])({_NUMBER})\s*(?:hours?|hrs?|h)\b", re.I)
_UNKNOWN = re.compile(r"\b(?:don['’]?t\s+know|do\s+not\s+know|not\s+sure|unsure|unknown|no\s+idea|cannot\s+(?:say|confirm)|can['’]?t\s+(?:say|confirm))\b", re.I)
_TRAVEL = re.compile(r"\b(?:travel|journey|driv\w*|trip|hours?|hrs?|distance)\b", re.I)
_SUPPLY = re.compile(r"\b(?:cloths?|spares?)\b", re.I)
_MODEL = re.compile(r"\b(?:model|equipment|machine|press|filter\s+type)\b", re.I)
_NEW = re.compile(r"\b(?:another|different|new)\s+(?:customer|visit|job|request|estimate|machine|equipment|plant|site|process|circuit)\b|\b(?:change|switch)\s+(?:the\s+)?(?:topic|subject)\b", re.I)


def _text(value) -> str:
    return " ".join(str(value).casefold().replace("’", "'").split())


def _latest(history: list[dict], role: str) -> str:
    return next((str(item.get("content", "")) for item in reversed(history)
                 if item.get("role") == role), "")


def _number(value: str):
    value = value.casefold().replace(",", ".")
    number = _WORDS[value] if value in _WORDS else float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return int(number) if int(number) == number else number


def product_intent(message: str, *, followup: bool = False) -> str:
    """A display/generation hint, never a source of an answer or a diagnosis."""
    text = _text(message)
    if re.search(r"\b(?:recommend|suggest)\b", text) and (not re.search(r"\b(?:why|explain|reason|cause)\b", text)
            or re.search(r"\b(?:and|also|plus)\b.{0,70}\b(?:recommend|suggest)\b", text)):
        return "recommend"
    if re.search(r"\b(?:specs?|specifications?|dimensions?|weight|capacity|technical\s+data|rated|rating|voltage|power|length|width|height|filter\s+area)\b", text):
        return "specification"
    if re.search(r"\b(?:why|reason|cause|explain|what\s+(?:went\s+wrong|happened))\b", text):
        return "explain"
    if re.search(r"\b(?:recommend|suggest|choose|select|which|best|suitable|solution)\b", text):
        return "recommend"
    if followup or re.search(r"\b(?:what\s+is|what\s+does|how\s+does|describe|tell\s+me\s+(?:more\s+)?about)\b", text):
        return "describe"
    return "recommend"


def _referential_product(message: str, previous: dict, routing: dict) -> bool:
    if not previous or _NEW.search(message):
        return False
    text = _text(message)
    old_equipment = _text(previous.get("equipment") or "")
    equipment = routing.get("known_fields", {}).get("equipment")
    # A different named subject actually present in the user's new question
    # takes precedence over the router's sometimes stale conversation summary.
    if isinstance(equipment, str) and _text(equipment) != old_equipment and _text(equipment) in text:
        recommendation = _text(previous.get("recommendation") or "")
        if _text(equipment) != recommendation:
            return False
    if re.search(r"\b(?:instead|another|different)\b.{0,25}\b(?:machine|equipment|customer|problem|topic|subject)\b|\bwhat\s+about\s+(?!it\b|that\b|its\b|this\b)", text):
        return False
    return bool(re.search(r"\b(?:it|its|that|this|they|their|those|these|the\s+(?:recommended\s+)?(?:product|optimizer|solution))\b", text)
                or re.search(r"^(?:and\s+|but\s+)?(?:why\b|what\s+happened\b|what\s+specs?\b|explain\b)", text)
                or re.search(r"\b(?:other|additional|alternative|more)\s+(?:\w+\s+)?(?:options?|recommendations?|products?|solutions?)\b", text))


def _supply_value(message: str, question: str):
    """Return (value, reason); None is deliberate rather than a false default."""
    text = _text(message)
    # The subject must be cloths, not an unrelated customer-provided spare part.
    if _SUPPLY.search(text):
        if re.search(r"\b(?:customer|client|they)\b.{0,30}\b(?:has|have)\s+no\b.{0,25}\bcloths?\b|\bno\s+customer[ -]supplied\s+cloths?\b|\bcloths?\b.{0,20}\bnot\s+customer[ -]supplied\b", text):
            return False, "explicit absence of customer-supplied cloths"
        if re.search(r"\b(?:customer|client|they|we)\s+(?:already\s+)?(?:do\s+not|don't|does\s+not|doesn't)\s+have\b.{0,45}\bcloths?\b", text):
            return False, "explicit absence of customer cloth stock"
        if re.search(r"\b(?:customer|client)\b.{0,35}\b(?:cannot|can't|won't|will\s+not)\s+(?:supply|provide)\b.{0,25}\bcloths?\b", text):
            return False, "explicit customer non-supply"
        if re.search(r"\b(?:cloths?|spares?)\b.{0,25}\b(?:not\s+(?:available|supplied)|need\s+to\s+be\s+(?:bought|purchased|supplied))\b", text):
            return False, "explicit cloth supply requirement"
        if re.search(r"\b(?:service|technician|supplier|metso)\b.{0,35}\b(?:must|will|should|needs?\s+to)\s+(?:supply|provide|bring|buy|purchase)\b.{0,30}\bcloths?\b", text):
            return False, "explicit service-supplied cloths"
        if re.search(r"\b(?:need|have|want)\s+to\s+(?:buy|purchase|order|supply)\b.{0,30}\bcloths?\b|\binclude\b.{0,20}\b(?:buying|purchasing|supplying)\b.{0,20}\bcloths?\b", text):
            return False, "explicit cloth purchase in requested scope"
        if re.search(r"\b(?:customer|client|they)\b.{0,35}\b(?:has|have)\b.{0,15}\b(?:some|a\s+few)\b.{0,25}\b(?:cloths?|spares?)\b", text):
            return None, "partial cloth stock does not establish supply of all replacements"
        if re.search(r"\b(?:customer|client|they)\b.{0,35}\b(?:has|have|holds?|provides?|supplies|will\s+(?:provide|supply)|already\s+(?:bought|purchased))\b.{0,35}\b(?:cloths?|spares?)\b", text):
            return True, "explicit customer cloth supply"
        if re.search(r"\b(?:cloths?|spares?)\b.{0,25}\b(?:customer[ -]supplied|provided\s+by\s+(?:the\s+)?customer|already\s+(?:available|on\s+site))\b", text):
            return True, "explicit cloths available from customer"
        if re.search(r"\bcustomer[ -]supplied\s+cloths?\b", text):
            return True, "explicit customer-supplied cloths"
    yes_no = re.match(r"^(yes|yeah|yep|no|nope)\b", text)
    q = _text(question)
    if yes_no and _SUPPLY.search(q):
        positive = yes_no[1] in {"yes", "yeah", "yep"}
        if re.search(r"\b(?:customer|client|you)\b.{0,35}\b(?:have|has|provide|supply)\b|\bcloths?\s+(?:already\s+)?(?:supplied|available)\b", q):
            return positive, "direct answer to customer-supply question"
        if re.search(r"\b(?:we|service|technician)\b.{0,25}\b(?:supply|provide|buy|purchase)\b|\binclude\b.{0,25}\bcloth", q):
            return not positive, "direct answer to service-supply question"
    return None, None


def _travel_value(message: str, pending: list[str], question: str):
    """Expose only the shared resolver's confirmed travel operand to memory."""
    from .service_request import resolve_duration_inputs, duration_literal_spans, _TOTAL, _ONEWAY

    text = _text(message)
    if re.search(r"\b(?:no\s+travel(?!\s+(?:budget|cost|data|information|estimate))|travel\s+(?:is\s+)?not\s+(?:required|needed)|(?:technician|engineer|we|i)\s+(?:is\s+|are\s+|am\s+)?already\s+on[ -]?site)\b", text):
        return 0, "explicit no travel"
    if re.search(r"\bone\s+trip\b|\bper\s+trip\b", text) and not _TOTAL.search(text):
        return None, "one trip does not establish one-way or round-trip travel"
    only_travel = pending == ["round_trip_travel_hours"]
    spans = duration_literal_spans(message)
    # An explicit travel basis can identify an otherwise bare hour expression;
    # a pending bundled question cannot turn a bare number into travel hours.
    explicit_basis = bool(_TOTAL.search(text) or _ONEWAY.search(text))
    fields = ["round_trip_travel_hours"] if only_travel or (spans and explicit_basis) else []
    resolved_question = question if only_travel else ""
    record = next(record for record in resolve_duration_inputs([message], pending_fields=fields,
                  pending_question=resolved_question) if record.role == "travel_hours")
    if record.status in {"missing", "unknown"}:
        return None, None
    reasons = {"range": "ambiguous travel duration range",
               "alternatives": "alternative travel durations require clarification",
               "multiple": "multiple hour quantities require travel clarification",
               "negated": "negated travel duration is not confirmed",
               "qualified": "qualified travel duration requires confirmation"}
    if record.uncertainty:
        return None, reasons[record.uncertainty]
    if _TOTAL.search(text) and _ONEWAY.search(text):
        return None, "conflicting one-way and round-trip wording"
    if record.status != "known":
        return None, "hour quantity lacks a round-trip or one-way basis"
    # Do not infer zero travel merely because some unrelated object is already
    # on site. Explicit no-travel statements are handled above.
    if not spans and not only_travel:
        return None, None
    value = record.total_hours
    value = int(value) if value == value.to_integral_value() else float(value)
    if record.basis == "oneway":
        reason = "explicit one-way duration doubled for return journey"
    elif _TOTAL.search(text):
        reason = "explicit round-trip travel"
    else:
        reason = ("duration reply to round-trip-only clarification" if spans else
                  "numeric reply to round-trip-only clarification")
    if record.estimated:
        reason += "; user-estimated nominal duration"
    return value, reason


def _unknown_fields(message: str, pending: list[str]) -> set[str]:
    if not _UNKNOWN.search(message):
        return set()
    fields = set()
    # Scope each uncertainty clause, avoiding 'I don't know travel, but the
    # customer has cloths' incorrectly erasing the explicit supply statement.
    clauses = re.split(r"[.;!?]|\bbut\b|\bhowever\b", message, flags=re.I)
    for clause in clauses:
        if not _UNKNOWN.search(clause):
            continue
        if _TRAVEL.search(clause):
            fields.add("round_trip_travel_hours")
        if _SUPPLY.search(clause):
            fields.add("cloths_supplied")
        if _MODEL.search(clause):
            fields.add("equipment")
    return fields or set(pending)


def _explicit_cloth_job_count(message: str):
    """Recognize a counted replacement job, not a brochure/specification query.

    This narrow correction requires independent evidence in the actual user
    message. A count left in the model's typed fields is insufficient.
    """
    text = _text(message)
    if re.search(r"\b(?:brochure|datasheet|specs?|specifications?|suitable|suitability|recommend)\b|\bcloth\s+(?:length|width|dimensions?)\b|\bwhich\s+(?:product|filter|machine|equipment)\b", text):
        return None
    if re.search(r"\bhow\s+(?:to|do\s+(?:i|we|you)|should\s+(?:i|we))\s+(?:replace|change|install)\b", text):
        return None
    if re.search(r"\d\s*(?:-|–|to)\s*\d\s+(?:filter\s+)?cloths?\b", text):
        return None
    counts = [_number(match[1]) for match in re.finditer(
        rf"(?<![\w.\-])({_NUMBER})\s+(?:(?:new|replacement)\s+)?(?:filter\s+)?cloths?\b", text)]
    if len(counts) != 1 or not isinstance(counts[0], int) or counts[0] <= 0:
        return None
    action = re.search(r"\b(?:replace|replaced|replacement|replacing|change|changed|changing|install|installed|installing)\b", text)
    job = re.search(r"\b(?:visit|service|technician|budget|estimate|cost|quotation|quote)\b|\b(?:customer|client)\b.{0,50}\b(?:needs?|requires?|wants?)\b|\b(?:we|i)\s+need\b", text)
    imperative = re.match(r"^(?:please\s+)?(?:replace|change|install)\b", text)
    return counts[0] if action and (job or imperative) else None


def _name_present(value, message: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    compact = lambda item: re.sub(r"[^\w]", "", _text(item))
    return compact(value) in compact(message)


def reconcile_route(routing: dict, history: list[dict], context: dict | None = None, *, interpretation: dict | None = None):
    """Return ``(schema-compatible routing, next_context, diagnostic_trace)``.

    ``history`` includes the latest user message. Memory is returned as plain
    dictionaries and is private runtime state, not extra router schema fields.
    Use :func:`update_context` after a tool call to record its actual question.
    """
    route = deepcopy(routing)
    state = deepcopy(context or {})
    trace = []
    message = _latest(history, "user")
    # Literal updates and explicit references to verified active state are
    # stronger evidence of continuation than a model's transition label.
    # Do this before retiring state, so a short input cannot erase its job.
    active_route = state.get("last_route")
    self_contained_job = bool(re.search(r"\b(?:a|the)\s+customer\b.{0,90}\b(?:needs?|requires?|wants?)\b", _text(message)))
    if not _NEW.search(message) and not self_contained_job:
        dependency = (interpretation or {}).get("initial_context_requirement", (interpretation or {}).get("context_requirement"))
        if active_route == "service_cost" and dependency != "self_contained":
            from .service_request import service_continuation_proof
            pending_inputs = list(state.get("pending_fields", []))
            pending_question = str(state.get("pending_question") or "")
            supply, supply_reason = _supply_value(message, pending_question)
            travel, travel_reason = _travel_value(message, pending_inputs, pending_question)
            proof = service_continuation_proof(history, message, pending_fields=pending_inputs,
                pending_question=pending_question, service_fields=state.get("service", {}).get("fields", {}))
            update = supply_reason is not None or travel_reason is not None or bool(_unknown_fields(message, pending_inputs)) or proof["proven"]
            if (interpretation or {}).get("context_requirement") == "self_contained":
                # A contextual transition label cannot erase a literal update
                # that was initially recognized as dependent. Only the shared
                # proof may override it; pending-hour fallback is insufficient.
                update = route["route"] == "service_cost" and proof["proven"]
            if update:
                interpretation = {**(interpretation or {}), "relation": "followup"}
                route["route"], route["query"] = "service_cost", message
                trace.append({"operation": "current_request_reconciliation", "reason": "literal service-input update retains its verified job", "proof": proof["reasons"]})
        elif active_route == "product":
            explicit_reference = bool(re.search(r"\b(?:it|its|that|this|those|these|recommended|alternatives|(?:other|additional|alternative|more)\s+(?:\w+\s+)?(?:options?|recommendations?|products?|solutions?))\b", _text(message)))
            # A reference-only request can retain a subject; local pronouns in
            # a newly stated customer problem cannot undo its new-request
            # boundary. In particular, declaration + 'that problem' refers to
            # the declaration in this message, not an older customer need.
            reference_request = bool(re.match(r"^(?:(?:please|and|also|but)\s+)*(?:why|what|how|which|can|could|would|do|does|is|are|tell|describe|explain|give|show)\b", _text(message)))
            self_contained = (interpretation or {}).get("context_requirement") == "self_contained"
            if self_contained and (interpretation or {}).get("initial_context_requirement") == "needs_context":
                from .product_request import _reference_only_request
                previous = state.get("product", {})
                names = [name for name in (previous.get("recommendation"), previous.get("discussed_subject")) if name]
                if _reference_only_request(message, previous, names):
                    self_contained = False
            if not self_contained and explicit_reference and reference_request and _referential_product(message, state.get("product", {}), route):
                interpretation = {**(interpretation or {}), "relation": "followup"}
                route["route"], route["query"] = "product", message
                trace.append({"operation": "current_request_reconciliation", "reason": "explicit reference retains the verified product subject"})
    if interpretation and interpretation.get("relation") == "new_request":
        state = {}
        trace.append({"operation": "current_request_boundary", "new_request": True,
                      "reason": "current request starts a new subject; prior operands and needs retired"})
    previous_route = state.get("last_route")
    pending = list(state.get("pending_fields", []))
    question = str(state.get("pending_question") or _latest(history[:-1], "assistant"))
    known = route["known_fields"]
    unknown = _unknown_fields(message, pending)
    previous_service = state.get("service", {}) if previous_route == "service_cost" else {}
    previous_product = state.get("product", {}) if previous_route == "product" else {}
    job_count = _explicit_cloth_job_count(message)
    if job_count is not None and route["route"] != "service_cost":
        trace.append({"operation": "conversation_route_reconciliation",
                      "reason": "actual user message specifies a counted cloth replacement job",
                      "original_route": route["route"], "cloth_count": job_count})
        route["route"] = "service_cost"
        route["query"] = message
        known["cloth_count"] = job_count
        known["task"] = "cloth replacement"
    # Bare replies can be misrouted because they contain no equipment or task.
    # Correct only actual answers to a pending service question, never a new
    # product/process request merely because the previous estimate is unfinished.
    service_reply = bool(previous_service and pending and not _NEW.search(message)
                         and (unknown or re.fullmatch(rf"\s*(?:yes|no|about\s+)?\s*{_NUMBER}?\s*(?:hours?|hrs?)?\s*[.!]?\s*", _text(message))))
    if service_reply and route["route"] in {"clarify", "unsupported"}:
        route["route"] = "service_cost"
        trace.append({"operation": "conversation_route_reconciliation", "reason": "reply to pending service inputs", "original_route": routing["route"]})

    if route["route"] == "product":
        for name in ("country", "site", "task", "cloth_count", "cloths_supplied", "round_trip_travel_hours", "iron_pct", "silica_pct"):
            known[name] = None
        followup = (bool(previous_product) and interpretation["relation"] == "followup"
                    if interpretation else _referential_product(message, previous_product, route))
        intent = product_intent(message, followup=followup)
        if followup:
            subject = previous_product.get("subject_query", "")
            candidate = previous_product.get("recommendation")
            # Specification/description follow-ups address the selected
            # candidate. Including the customer's original machine here could
            # accidentally make its numerical facts answer 'its specs'. The
            # original question remains in state for later explanatory turns.
            pieces = ([] if candidate and intent in {"specification", "describe"}
                      else ["Customer's previous question: " + subject])
            if candidate:
                pieces.append("Previously discussed candidate product: " + str(candidate))
            pieces.append("Current question: " + message)
            route["query"] = "\n".join(pieces)[:6000]
            known["equipment"] = (candidate if intent in {"specification", "describe"} and candidate
                                  else previous_product.get("equipment"))
            product = deepcopy(previous_product)
            trace.append({"operation": "product_context_reconciliation", "intent": intent,
                          "reason": "referential follow-up preserves actual customer question", "previous_candidate_used": bool(candidate)})
        else:
            # A new product question must not inherit a service request or an
            # old machine embedded in the model's standalone query. Preserve
            # the actual user's question; only verified follow-ups above add
            # conversational context. Named equipment remains optional because
            # retrieval can use the complete current question directly.
            if route["query"] != message:
                trace.append({"operation": "product_context_reconciliation",
                              "intent": intent, "reason": "new product question uses actual current user text"})
            route["query"] = message
            if known.get("equipment") is not None and not _name_present(known["equipment"], message):
                known["equipment"] = None
            product = {"subject_query": message[:3000], "equipment": known.get("equipment"), "recommendation": None}
        product.update(intent=intent, followup=followup)
        broad_specs = intent == "specification" and bool(re.search(r"\b(?:specs?|specifications?|technical\s+(?:overview|details))\b", message, re.I)) and not re.search(
            r"\b(?:voltage|power|capacity|height|length|width|weight|dimensions?|pressure|temperature|frequency|flow|speed|depth|thickness|area|volume|rating|rated|how\s+(?:much|many))\b", message, re.I)
        product["broad_specification"] = broad_specs
        if broad_specs:
            route["query"] += "\nPlease help describe the documented technical features, distinguishing them from numerical ratings not established by the available evidence."
        state = {"last_route": "product", "product": product, "pending_fields": [],
                 "pending_question": "", "scenario_mode": False}
        return route, state, trace

    if route["route"] != "service_cost":
        relevant = {"iron_pct", "silica_pct"} if route["route"] == "flotation" else set()
        for name in known:
            if name not in relevant:
                known[name] = None
        return route, {"last_route": route["route"], "pending_fields": [], "pending_question": "", "scenario_mode": False}, trace

    known["iron_pct"] = known["silica_pct"] = None

    text = _text(message)
    self_contained_request = bool(
        re.search(r"\b(?:a|the)\s+customer\b.{0,90}\b(?:needs?|requires?|wants?)\b", text)
        or re.search(r"\b(?:estimate|budget|quote)\b.{0,90}\b(?:visit|replace|repair|service)\b", text))
    if re.search(r"\b(?:same|that|this|it|instead\s+of)\b", text):
        self_contained_request = False
    new_request = not previous_service or bool(_NEW.search(message)) or self_contained_request
    trace.append({"operation": "service_request_boundary", "new_request": new_request})
    fields = {} if new_request else deepcopy(previous_service.get("fields", {}))
    provenance = {} if new_request else deepcopy(previous_service.get("provenance", {}))
    unknowns = set() if new_request else set(previous_service.get("unknown_fields", []))
    numeric_travel_reply = (not new_request and pending == ["round_trip_travel_hours"]
                            and bool(re.fullmatch(rf"(?:about\s+)?{_NUMBER}\s*[.!]?", text)))
    for name in SERVICE_FIELDS:
        if name in {"cloths_supplied", "round_trip_travel_hours"}:
            continue
        value = known.get(name)
        if numeric_travel_reply and name == "cloth_count":
            value = fields.get(name)
        if value is not None:
            # A newly requested visit must not inherit identical old values
            # merely because the model left its previous typed fields filled.
            old = previous_service.get("fields", {}).get(name)
            if new_request and old is not None and value == old and name in {"country", "equipment", "site", "cloth_count"}:
                literal = str(value) if name == "cloth_count" else _text(value)
                if literal not in _text(message):
                    value = None
            if value is not None:
                fields[name] = value
        known[name] = fields.get(name)

    supply, supply_reason = _supply_value(message, question if not new_request else "")
    travel, travel_reason = _travel_value(message, pending if not new_request else [], question)
    for name, value, reason in (("cloths_supplied", supply, supply_reason),
                                ("round_trip_travel_hours", travel, travel_reason)):
        if name in unknown:
            fields[name] = None
            provenance.pop(name, None)
            unknowns.add(name)
            trace.append({"operation": "service_input_provenance", "field": name, "action": "unknown", "reason": "user explicitly does not know this input"})
        elif value is not None:
            fields[name] = value
            provenance[name] = {"user_text": message[:2000], "basis": reason}
            unknowns.discard(name)
            trace.append({"operation": "service_input_provenance", "field": name, "action": "accepted", "basis": reason, "value": value})
        elif reason:
            fields[name] = None
            provenance.pop(name, None)
            trace.append({"operation": "service_input_provenance", "field": name, "action": "withheld", "reason": reason})
        elif name not in provenance:
            fields[name] = None
            if known.get(name) is not None:
                trace.append({"operation": "service_input_provenance", "field": name, "action": "withheld", "reason": "router value has no explicit user provenance"})
        known[name] = fields.get(name)
    for name in unknown - {"cloths_supplied", "round_trip_travel_hours"}:
        if name in SERVICE_FIELDS:
            fields[name] = known[name] = None
            unknowns.add(name)
    for name in tuple(unknowns):
        if name not in unknown and known.get(name) is not None:
            unknowns.discard(name)
    state = {"last_route": "service_cost", "service": {"fields": fields,
             "provenance": provenance, "unknown_fields": sorted(unknowns)},
             "pending_fields": [] if new_request else pending,
             "pending_question": "" if new_request else question,
             "scenario_mode": bool(unknowns)}
    if state["scenario_mode"]:
        trace.append({"operation": "service_unknown_input_handling", "unknown_fields": sorted(unknowns),
                      "scenario_mode": True, "reason": "offer bounded historical scenarios rather than repeat the same question"})
    return route, state, trace


def update_context(context: dict, routing: dict, tool_result: dict | None = None,
                   recommendation: str | None = None) -> dict:
    """Record only an actual tool question and a validated recommendation.

    ``recommendation`` must come from the checked product answer, never from
    raw model routing. The caller owns that validation.
    """
    state = deepcopy(context)
    if tool_result is not None:
        state["pending_fields"] = list(tool_result.get("missing_fields", []))
        state["pending_question"] = str(tool_result.get("question") or "")
    if routing.get("route") == "product" and recommendation:
        product = state.setdefault("product", {})
        product["recommendation"] = product["discussed_subject"] = recommendation
        product.setdefault("recommendations", [])
        if recommendation not in product["recommendations"]:
            product["recommendations"].append(recommendation)
    return state
