"""Immutable, literal duration inputs for an active service request.

An unspecified journey is retained as hours, but is never a billing operand
until its total/one-way basis is resolved. No source prices or travel guesses
are supplied by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from decimal import Decimal
import re


_WORDS = {word: index for index, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUM = r"(?:[-+]?\d+(?:[.,]\d+)?|" + "|".join(_WORDS) + r")"
_HOUR_UNIT = r"(?:hours?|hrs?|h)\b"
_HOUR_MODIFIER = r"(?:total|estimated|approximate|billable|billed|chargeable|round[ -]?trip|one[ -]?way|on[- ]?site|travel|journey|trip|driving|labou?r|installation|repair|work)"
_HOUR_BODY = rf"(?:{_HOUR_MODIFIER}\s+)*{_HOUR_UNIT}"
_HOURS = re.compile(rf"(?<![\w.-])(?P<number>{_NUM})\s*(?P<modifiers>(?:{_HOUR_MODIFIER}\s+)*){_HOUR_UNIT}", re.I)
_GROUP_HOURS = re.compile(rf"(?P<number>{_NUM})\s*(?P<modifiers>(?:{_HOUR_MODIFIER}\s+)*){_HOUR_UNIT}", re.I)
_SHARED_HOURS = re.compile(
    rf"(?<![\w.-])(?P<first>{_NUM})\s*(?:{_HOUR_BODY})?\s*"
    rf"(?P<link>[-\u2013\u2014]|to|and|or)\s*(?P<second>{_NUM})\s*{_HOUR_BODY}", re.I)
_ROLE_WORD = r"(?:travel|journey|trip|driving|labou?r|on[- ]?site|installation|repair|work)"
_ROLE = re.compile(r"\b(?:(?P<travel>travel|journey|trip|driving)|(?P<labour>labou?r|on[- ]?site|installation|repair|work))\b", re.I)
_BILLING_MODIFIER = r"(?:total|estimated|approximate|billable|billed|chargeable|round[ -]?trip|one[ -]?way)"
_AFTER_HOURS_ROLE = re.compile(
    rf"\s*(?:(?:of|for)\s+)?(?:{_BILLING_MODIFIER}\s+)*(?P<role>{_ROLE_WORD})\b", re.I)
_DURATION_BOUNDARY = re.compile(r"[;!?\n]|(?<!\d)[.,](?!\d)|\b(?:and|but|plus)\b", re.I)
_ROLE_LINK = re.compile(
    r"\s*(?:(?:time|duration|hours?|is|are|was|were|will|would|be|takes?|needs?|requires?|lasts?|"
    r"of|for|total|billed|billable|chargeable|estimated|approximate|about|around|roughly|approximately|"
    r"may|might|perhaps|maybe|not|at|least|most|up|to|more|less|than|between|either)\s+)*", re.I)
_UNKNOWN = re.compile(r"\b(?:don['’]t\s+know|do\s+not\s+know|not\s+sure|unsure|unknown|no\s+idea|cannot\s+(?:say|confirm))\b", re.I)
_TOTAL = re.compile(r"round[ -]?trip|there\s+and\s+back|(?:both|two)\s+ways|including\s+(?:the\s+)?return|total\s+(?:(?:billed|billable|chargeable|return|estimated|approximate)\s+)*(?:travel|journey|trip|driving|hours)|(?:travel|journey|trip|driving|hours?)\s+(?:in\s+)?total", re.I)
_ONEWAY = re.compile(r"one[ -]?way|each\s+way|per\s+way|hours?\s+(?:to\s+get\s+)?there\b(?!\s+and\s+back)", re.I)
_NOMINAL = re.compile(r"\b(?:about|around|approximately|approximate|roughly|estimated)\b", re.I)
_BARE_DURATION = re.compile(rf"(?:(?:about|around|approximately|roughly|estimated|total)\s+)?(?P<number>{_NUM})(?:\s+(?:in\s+total|total|round[ -]?trip|including\s+(?:the\s+)?return|one[ -]?way|each\s+way))?[.!]?", re.I)


def _shared_hour_groups(text):
    for group in _SHARED_HOURS.finditer(text):
        first = text[group.start():group.start("link")]
        second = text[group.start("second"):group.end()]
        first_roles = {bool(role.group("travel")) for role in _ROLE.finditer(first)}
        second_roles = {bool(role.group("travel")) for role in _ROLE.finditer(second)}
        # Separately labelled conjunctive durations are not one range. Keep
        # explicit bounds and alternatives unresolved even if labels conflict.
        if (group["link"].casefold() == "and" and first_roles and second_roles
                and first_roles.isdisjoint(second_roles)
                and not re.search(r"\bbetween\s*$", text[:group.start()], re.I)):
            continue
        yield group


def duration_literal_spans(message):
    """Literal numeric spans belonging to hours, including shared-unit bounds.

    A range's first number is a duration candidate, never a material count.
    This helper classifies text only; it does not authorize any billing value.
    """
    if not isinstance(message, str):
        return ()
    spans = {match.span("number") for match in _HOURS.finditer(message)}
    for match in _shared_hour_groups(message):
        spans.update((match.span("first"), match.span("second")))
    return tuple(sorted(spans))


def _uncertainty(text, match):
    """Retain uncertainty rather than selecting one literal bound as hours."""
    for group in _shared_hour_groups(text):
        if group.start() <= match.start() < group.end():
            reason = "alternatives" if group["link"].casefold() == "or" else "range"
            return reason, (_value(group["first"]), _value(group["second"]))
    # Bound negation to the current clause so an unrelated negative assertion
    # about another component cannot negate a later affirmative duration.
    left, _ = _duration_bounds(text, match)
    prefix = text[left:match.start()]
    suffix = text[match.end():]
    if (re.search(r"\b(?:not|never|no\s+longer|isn['’]t|aren['’]t|won['’]t|wasn['’]t)\b", prefix, re.I)
            or re.match(r"\s*(?:is|was|would\s+be)\s+not\s+(?:correct|right|confirmed)\b", suffix, re.I)):
        return "negated", (_value(match["number"]),)
    if re.search(r"\b(?:perhaps|maybe|might|may|up\s+to|at\s+(?:least|most)|more\s+than|less\s+than)\b", prefix, re.I):
        return "qualified", (_value(match["number"]),)
    return None, ()


def _value(token):
    token = token.casefold()
    return Decimal(_WORDS[token]) if token in _WORDS else Decimal(token.replace(",", "."))


def _display(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True)
class DurationInput:
    role: str
    value: Decimal | None = None
    unit: str = "hour"
    basis: str = "unspecified"
    status: str = "missing"
    quote: str = ""
    turn: int = -1
    origin_quote: str = ""
    uncertainty: str | None = None
    candidates: tuple[Decimal, ...] = ()
    estimated: bool = False

    def __post_init__(self):
        if self.role not in {"travel_hours", "labour_hours"} or self.unit != "hour":
            raise ValueError("A duration needs an explicit billing role and hours unit")
        if self.basis not in {"total", "oneway", "unspecified"} or self.status not in {"known", "ambiguous", "unknown", "missing"}:
            raise ValueError("Unsupported duration basis or resolution status")
        if self.value is not None and (not isinstance(self.value, Decimal) or not self.value.is_finite()):
            raise ValueError("Duration values must be finite Decimals")
        if self.status == "known" and (self.value is None or self.value < 0 or self.basis == "unspecified"):
            raise ValueError("Known durations need nonnegative hours and a resolved basis")
        if self.uncertainty not in {None, "range", "alternatives", "negated", "qualified", "multiple"}:
            raise ValueError("Unsupported duration uncertainty")
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in self.candidates):
            raise ValueError("Duration candidates must be finite Decimals")
        if self.status == "known" and self.uncertainty is not None:
            raise ValueError("Uncertain durations cannot be known billing operands")
        if not isinstance(self.estimated, bool):
            raise ValueError("The duration estimate flag must be boolean")

    @property
    def total_hours(self):
        if self.status != "known" or self.value is None:
            return None
        return self.value * (2 if self.role == "travel_hours" and self.basis == "oneway" else 1)

    def as_dict(self):
        return {**asdict(self), "total_hours": self.total_hours}

    def to_fact(self, identifier):
        if self.total_hours is None:
            return None
        return {"id": identifier, "label": ("User-estimated total " if self.estimated else "Confirmed total ") + self.role.replace("_", " "),
                "value": self.total_hours, "unit": "hour", "currency": None, "kind": "user_input",
                "user_text": self.origin_quote + "\n" + self.quote if self.origin_quote else self.quote, "source_ids": [], "parents": [],
                "assumptions": (["Explicit one-way hours are doubled to include the return journey."] if self.basis == "oneway" else [])
                    + (["User-supplied hours are an approximate nominal estimate; actual billable time may differ."] if self.estimated else []),
                "scope": {"role": self.role, "turn": self.turn, "duration_basis": self.basis,
                          "duration_status": self.status, "nominal_hours": self.value, "input_quote": self.quote,
                          "input_value_quote": self.origin_quote or self.quote,"duration_estimated":self.estimated}}


def _pending_role(fields):
    fields = set(fields or ())
    if fields in ({"round_trip_travel_hours"}, {"travel_hours"}): return "travel_hours"
    if fields in ({"labour_hours"}, {"on_site_hours"}): return "labour_hours"
    return None


def _duration_bounds(text, match):
    """Keep billing cues local without splitting a range or 'there and back'."""
    protected = [group.span() for group in _shared_hour_groups(text)]
    protected.extend(group.span() for group in re.finditer(r"\bthere\s+and\s+back\b", text, re.I))
    left, right = 0, len(text)
    for boundary in _DURATION_BOUNDARY.finditer(text):
        if any(start <= boundary.start() < end for start, end in protected):
            continue
        if boundary.end() <= match.start():
            left = boundary.end()
        elif boundary.start() >= match.end():
            right = boundary.start()
            break
    return left, right


def _role_options(text, match):
    """Bind a duration to its local role, retaining explicit contradictions.

    Billing adjectives may sit between hours and the following role. A role
    before a separate clause must not win merely because it appeared first.
    """
    left, right = _duration_bounds(text, match)
    phrases = [match["modifiers"]]
    after = _AFTER_HOURS_ROLE.match(text[match.end():right])
    if after:
        phrases.append(after["role"])
    before = list(_ROLE.finditer(text[left:match.start()]))
    if before:
        last = before[-1]
        tail = text[left + last.end():match.start()]
        explicit_field = (last.group("travel") or re.fullmatch(r"labou?r|on[- ]?site", last.group(), re.I)
                          or re.match(r"\s+(?:hours?|time|duration)\b", tail, re.I))
        if not any(_ROLE.search(phrase) for phrase in phrases) or explicit_field and _ROLE_LINK.fullmatch(tail):
            # A named billing field immediately governing the number can
            # conflict. General job context ('repair with', 'work needs')
            # cannot override a directly attached travel/labour phrase.
            phrases.append(last.group())
    return {"travel_hours" if role.group("travel") else "labour_hours"
            for phrase in phrases for role in _ROLE.finditer(phrase)}


def _basis_confirmation(text, pending_fields=(), pending_question=""):
    affirmative = bool(re.fullmatch(r"(?:yes|correct|that's\s+right|that\s+is\s+right)[.!]?",text,re.I))
    explicit = bool(re.fullmatch(r"(?:yes[, ]*)?(?:(?:(?:it|that)\s+is\s+)?(?:total|round[ -]?trip|including\s+(?:the\s+)?return(?:\s+(?:journey|trip))?|both\s+ways|one[ -]?way|each\s+way)|(?:it|that)\s+includes?\s+(?:the\s+)?return(?:\s+(?:journey|trip))?)[.!]?",text,re.I))
    if explicit:
        return "oneway" if _ONEWAY.search(text) else "total"
    if affirmative and _pending_role(pending_fields) == "travel_hours" and _TOTAL.search(pending_question):
        return "total"
    return None


def resolve_duration_inputs(messages, pending_fields=(), pending_question="", clarifications=()):
    """Return latest labour/travel records without discarding unresolved hours.

    Only a reply to an actual pending total-travel question can use a bare
    number. A basis-only confirmation can resolve an earlier literal duration;
    its quote is retained as the confirmation evidence.
    """
    current = {role: DurationInput(role) for role in ("labour_hours", "travel_hours")}
    if not isinstance(messages, (list, tuple)):
        return tuple(current.values())
    for turn, message in enumerate(messages):
        if not isinstance(message, str): continue
        text = message.strip().replace("’", "'")
        clarification = (clarifications.get(turn) if isinstance(clarifications,Mapping)
                         else clarifications[turn] if isinstance(clarifications,(list,tuple)) and turn < len(clarifications) else None)
        if isinstance(clarification,Mapping):
            turn_fields, turn_question = clarification.get("fields",()),clarification.get("question","")
        else:
            turn_fields, turn_question = (pending_fields,pending_question) if turn == len(messages)-1 else ((),"")
        if not isinstance(turn_question,str): turn_question = ""
        pending = _pending_role(turn_fields)
        updates = {}
        if re.search(r"\bno\s+travel(?!\s+(?:budget|cost|data|information|estimate))\b|\btravel\s+(?:is\s+)?not\s+(?:required|needed)\b|\balready\s+on[- ]?site\b", text, re.I):
            updates["travel_hours"] = DurationInput("travel_hours", Decimal(0), basis="total", status="known", quote=message, turn=turn)
        for clause in re.split(r"[;!?\n]|(?<!\d)[.,](?!\d)|\b(?:and|but)\b", text, flags=re.I):
            if not _UNKNOWN.search(clause): continue
            roles = {"travel_hours" if match.group("travel") else "labour_hours" for match in _ROLE.finditer(clause)}
            if not roles and pending: roles.add(pending)
            for role in roles:
                updates[role] = DurationInput(role, status="unknown", quote=message, turn=turn)
        candidates = {}
        matches = list(_HOURS.finditer(text))
        # A compact hyphenated range has no whitespace before its final
        # number. Read it only inside the already-proven range expression.
        for group in _shared_hour_groups(text):
            if not any(match.start("number") == group.start("second") for match in matches):
                match = _GROUP_HOURS.match(text, group.start("second"))
                if match is not None:
                    matches.append(match)
        matches.sort(key=lambda match: match.start())
        for match in matches:
            options = _role_options(text, match)
            if len(options) > 1:
                # A pending field cannot override contradictory literal roles.
                for role in options:
                    candidates.setdefault(role, []).append(DurationInput(role, status="ambiguous", quote=message,
                        turn=turn, uncertainty="multiple", candidates=(_value(match["number"]),)))
                continue
            role = next(iter(options)) if options else None
            if role is None and pending: role = pending
            if role is None and len(matches) == 1 and current["travel_hours"].status == "ambiguous" and re.fullmatch(rf"\s*{_NUM}\s*(?:hours?|hrs?|h)\s*(?:in\s+total|total|round[ -]?trip)?[.!]?", text, re.I):
                role = "travel_hours"
            if role is None: continue
            if updates.get(role, DurationInput(role)).status == "unknown": continue
            value = _value(match["number"])
            # Scope basis cues to the nearest conjunction, so a separate
            # labour figure does not supply a journey's billing interpretation.
            left, right = _duration_bounds(text, match)
            context = text[left:right]
            total, oneway = bool(_TOTAL.search(context)), bool(_ONEWAY.search(context))
            if role == "labour_hours": total, oneway = True, False
            if pending == "travel_hours" and _TOTAL.search(turn_question) and not oneway: total = True
            basis = "unspecified" if total == oneway else "total" if total else "oneway"
            status = "known" if basis != "unspecified" and value >= 0 else "ambiguous"
            uncertainty, alternatives = _uncertainty(text, match)
            if uncertainty:
                value, status = None, "ambiguous"
            candidates.setdefault(role, []).append(DurationInput(role,value,basis=basis,status=status,quote=message,turn=turn,
                uncertainty=uncertainty,candidates=alternatives,estimated=bool(_NOMINAL.search(context))))
        for role, values in candidates.items():
            alternatives = tuple(dict.fromkeys(value for record in values for value in
                (record.candidates or ((record.value,) if record.value is not None else ()))))
            updates[role] = values[0] if len(values) == 1 else DurationInput(role,status="ambiguous",quote=message,turn=turn,
                uncertainty="multiple",candidates=alternatives)
        if not matches and pending:
            bare = _BARE_DURATION.fullmatch(text)
            if bare:
                value = _value(bare["number"])
                oneway = pending == "travel_hours" and bool(_ONEWAY.search(text))
                known = pending == "labour_hours" or oneway or bool(_TOTAL.search(turn_question) or re.search(r"\btotal\b",text,re.I))
                updates[pending] = DurationInput(pending,value,basis="oneway" if oneway else "total" if known else "unspecified",
                                                  status="known" if known and value >= 0 else "ambiguous",quote=message,turn=turn,
                                                  estimated=bool(_NOMINAL.search(text)))
        prior = current["travel_hours"]
        if "travel_hours" not in updates and prior.value is not None and prior.status == "ambiguous" and prior.uncertainty is None:
            basis = _basis_confirmation(text, turn_fields, turn_question)
            if basis:
                updates["travel_hours"] = DurationInput("travel_hours",prior.value,basis=basis,status="known",quote=message,turn=turn,origin_quote=prior.origin_quote or prior.quote,estimated=prior.estimated)
        current.update(updates)
    return tuple(current.values())


def bind_duration_facts(inputs, messages, records, *, pending_fields=(), pending_question="", clarifications=()):
    """Replace raw duration operands with their typed, resolved records.

    A bare reply has no unit in its own text. Only the shared resolver's proof
    from the actual pending question can consume that exact literal span, so
    an equally valued component count remains a separate material request.
    Earlier replies are checked against their recorded clarification context
    even when a later turn updates the duration or adds another component.
    """
    bound_spans = set()
    for turn, message in enumerate(messages):
        if not isinstance(message, str):
            continue
        bare = _BARE_DURATION.fullmatch(message.strip())
        if bare is None:
            continue
        last = turn == len(messages) - 1
        resolved = resolve_duration_inputs(messages[:turn + 1],
            pending_fields=pending_fields if last else (),
            pending_question=pending_question if last else "", clarifications=clarifications)
        if any(record.turn == turn and record.quote == message and record.value == _value(bare["number"])
               and record.status in {"known", "ambiguous"} for record in resolved):
            offset = len(message) - len(message.lstrip())
            bound_spans.add((turn, tuple(position + offset for position in bare.span("number"))))
    output = []
    for fact in inputs:
        scope = fact.get("scope", {})
        if scope.get("role") in {"ambiguous_duration", "labour_hours", "travel_hours"}:
            continue
        if (fact.get("kind") == "user_input" and scope.get("role") == "requested_quantity"
                and (scope.get("turn"), tuple(scope.get("span", ()))) in bound_spans
                and fact.get("user_text") == messages[scope["turn"]]):
            continue
        output.append(fact)
    for record in records:
        fact = record.to_fact("U_duration_" + record.role)
        if fact is not None:
            output.append(fact)
    return output


def duration_question(records, categories=None):
    requested = set(("labour", "travel") if categories is None else categories)
    for record in sorted(records,key=lambda record:(-record.turn,record.role != "travel_hours")):
        if record.role == "travel_hours" and "travel" in requested and record.status == "ambiguous":
            mentioned = "You mentioned " + _display(record.value) + " hours. " if record.value is not None else ""
            return mentioned + "How many total billable travel hours should I use, including the return journey?"
        if record.role == "labour_hours" and "labour" in requested and record.status == "ambiguous":
            return "How many total on-site labour hours should I use?"
    if "travel" in requested and any(record.role == "travel_hours" and record.status == "missing" for record in records):
        return "How many total billable travel hours should I use, including the return journey?"
    if "labour" in requested and any(record.role == "labour_hours" and record.status == "missing" for record in records):
        return "How many total on-site labour hours should I use?"
    return None


def service_continuation_proof(active_messages, current_message, *, pending_fields=(), pending_question="",
                               service_fields=None):
    """Prove a literal update to a caller-established active service request.

    This classifies continuation only. It neither supplies a numeric operand
    nor approves a price, and cannot turn a basis acknowledgement into hours.
    Actual user text and existing quantity-supersession provenance are reused.
    """
    from .adaptive import request_facts
    from .service_evidence import _literal_material_phrase, _material_tokens, _QUANTITY_UNITS, _COUNT_UNITS

    no = {"proven":False,"reasons":[]}
    if not isinstance(active_messages,(list,tuple)) or not isinstance(current_message,str) or not current_message.strip(): return no
    prior = [item if isinstance(item,str) else item.get("content","")
             for item in active_messages if isinstance(item,str) or isinstance(item,dict) and item.get("role")=="user"]
    prior = [text for text in prior if isinstance(text,str) and text.strip()]
    if prior and prior[-1] == current_message: prior.pop()
    if not prior: return no
    text = current_message.strip()
    if re.search(r"\b(?:another|different|new)\s+(?:customer|client|visit|job|request|machine|equipment)\b|\b(?:change|switch)\s+(?:the\s+)?(?:topic|subject)\b|\b(?:a|the)\s+(?:customer|client)\b.{0,90}\b(?:needs?|requires?|wants?)\b",text,re.I):
        return no
    if re.search(r"\b(?:our|my|their)\b[^.!?]{0,80}\b(?:loses?|losing|drops?|declines?|unstable|oscillat\w*|fluctuat\w*)\b",text,re.I):
        return no
    fields = service_fields if isinstance(service_fields,dict) else {}
    anchors = _material_tokens(" ".join(str(fields.get(key) or "") for key in ("equipment","task")))
    if anchors and not any(anchors & _material_tokens(message) for message in prior): return no

    messages, turn = [*prior,current_message], len(prior)
    before = resolve_duration_inputs(prior)
    after = resolve_duration_inputs(messages,pending_fields=pending_fields,pending_question=pending_question)
    # The resolver may borrow a pending billing role for an otherwise unlabelled
    # hours phrase. That is appropriate for a field answer, but does not prove
    # that a self-contained equipment observation continues this service job.
    explicit_duration = any(_role_options(text, match) for match in _HOURS.finditer(text))
    journey_basis = bool(_ONEWAY.search(text) or re.search(
        r"round[ -]?trip|there\s+and\s+back|(?:both|two)\s+ways|including\s+(?:the\s+)?return", text, re.I))
    field_answer = _pending_role(pending_fields) is not None and bool(
        _BARE_DURATION.fullmatch(text) or re.fullmatch(
            rf"(?:(?:(?:it|that)\s+(?:is|takes?|will\s+(?:be|take))|use|allow|estimate)\s+)?"
            rf"(?:(?:about|around|approximately|roughly|estimated)\s+)?{_NUM}\s*{_HOUR_BODY}"
            r"(?:\s+(?:in\s+total|total|round[ -]?trip|including\s+(?:the\s+)?return(?:\s+(?:journey|trip))?|one[ -]?way|each\s+way))?[.!]?",
            text, re.I))
    if any(record.turn == turn and record.quote == current_message and (
            explicit_duration or journey_basis or field_answer or
            record.origin_quote and _basis_confirmation(text,pending_fields,pending_question))
            for record in after):
        return {"proven":True,"reasons":["literal duration update to the active service request"]}
    pending_travel = bool(set(pending_fields or ()) & {"round_trip_travel_hours","travel_hours"})
    if pending_travel and _basis_confirmation(text,pending_fields,pending_question) and any(
            record.role == "travel_hours" and record.status == "ambiguous" for record in before):
        return {"proven":True,"reasons":["literal reply to the pending travel-basis question; hours remain resolver-owned"]}

    facts = request_facts(messages,"service_cost")
    def literal_quantity(fact):
        scope = fact.get("scope",{})
        value = fact.get("value")
        span = scope.get("span")
        if (scope.get("turn") != turn or scope.get("role") != "requested_quantity" or
                fact.get("unit") not in _QUANTITY_UNITS or not isinstance(value,Decimal) or not value.is_finite() or value <= 0
                or fact.get("unit") in _COUNT_UNITS and value != value.to_integral_value()
                or not isinstance(span,list) or len(span)!=2): return False
        # Literal physical measurements do not become service component counts.
        suffix = current_message[span[1]:]
        if re.match(r"\s*(?:%|percent\b|microns?\b|micromet(?:er|re)s?\b|[µμ]m\b|mm\b|cm\b|met(?:er|re)s?\b|kg\b|ppm\b|bar\b|kpa\b)",suffix,re.I): return False
        return bool(_material_tokens(_literal_material_phrase(fact) or ""))
    current = {fact["id"]:fact for fact in facts if literal_quantity(fact)}
    if any(fact.get("scope",{}).get("supersession_provenance",{}).get("turn") == turn
           and fact["scope"]["supersession_provenance"].get("replaced_by_ref") in current for fact in facts):
        return {"proven":True,"reasons":["proven correction of an existing literal service quantity"]}
    addition = re.match(r"^(?:(?:please|and|also|then|can\s+you|could\s+you|would\s+you)\s+)*(?:add|include)\b|^(?:please\s+)?(?:also|additionally)\s+(?:need|require)\b",text,re.I)
    if addition and current:
        return {"proven":True,"reasons":["explicit addition of literal component quantities to the active service scope"]}
    return no
