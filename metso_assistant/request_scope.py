"""Source-free annotations from explicit material-supply statements by users.

This is a conservative literal binding, not a general ownership classifier.
Silence, pronouns, partial stock and hypothetical statements establish nothing.
"""
from __future__ import annotations

from copy import deepcopy
import re


_CUSTOMER = re.compile(
    r"\b(?P<subject>(?:the\s+)?(?:customer|client)|we)\s+"
    r"(?P<before>(?:(?:already|now|still|will|do|does|not|don't|doesn't|no\s+longer|never)\s+){0,5})"
    r"(?P<verb>has|have|provides?|suppl(?:y|ies)|owns?)\s+(?P<object>[^.;!?\n]{1,140})", re.I)
_PREDICATE = re.compile(
    r"(?P<object>[^.;!?\n]{1,120}?)\s+(?:are|is|were|will\s+be)\s+(?P<negative>not\s+)?"
    r"(?:customer|client)[ -]supplied\b", re.I)
_SERVICE = re.compile(
    r"\b(?:(?:the\s+)?(?:service\s+provider|technician|engineer|supplier|you)\s+"
    r"(?P<before>(?:(?:will|must|should|can|to|not|never|needs?\s+to)\s+){0,4})"
    r"|please\s+)(?:suppl(?:y|ies)|provides?|purchases?|buys?)\s+(?P<object>[^.;!?\n]{1,140})", re.I)
_NEGATIVE = re.compile(r"\b(?:not|never|no\s+longer)\b|\b(?:don't|doesn't)\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:if|whether|might|may|could|possibly|perhaps|assuming|assume|unsure|unknown)\b|\b(?:don't\s+know|do\s+not\s+know|not\s+sure|cannot\s+confirm)\b", re.I)
_PARTIAL = re.compile(r"\b(?:some|several|only|half)\b|\ba\s+few\b|\bpart\s+of\b", re.I)


def _statements(messages):
    for turn, text in enumerate(messages):
        if not isinstance(text, str):
            continue
        found = []
        for pattern, customer_owned in ((_CUSTOMER, True), (_PREDICATE, True), (_SERVICE, False)):
            for match in pattern.finditer(text):
                if (pattern is _CUSTOMER and match.group("subject").casefold() == "we"
                        and match.group("verb").casefold() in {"provide", "provides", "supply", "supplies"}):
                    # The speaker may be the servicing company. This pronoun
                    # does not establish customer-supplied material ownership.
                    continue
                before = match.groupdict().get("before") or ""
                negative = bool(_NEGATIVE.search(before) or match.groupdict().get("negative"))
                # A service provider refusing to supply does not establish
                # that the customer possesses the material.
                if not customer_owned and negative:
                    continue
                prefix = re.split(r"[.;!?\n]|\bbut\b", text[:match.start()], flags=re.I)[-1]
                if _UNCERTAIN.search(prefix) or _UNCERTAIN.search(match.group()) or text[match.end():].lstrip().startswith("?"):
                    continue
                supplied = customer_owned and not negative
                noun = match.group("object").strip()
                if customer_owned and re.match(r"(?:no|none\s+of\s+the)\b", noun, re.I):
                    supplied = False
                if supplied and _PARTIAL.search(noun):
                    continue
                found.append((match.start(), supplied, noun, match.group().strip()))
        for _, supplied, noun, quote in sorted(found):
            yield turn, supplied, noun, quote


def annotate_material_supply(inputs, messages):
    """Deep-copy literal operands and bind explicit matching supply statements.

    The latest applicable literal statement wins. ``customer_supplied=False``
    records an explicit lack of ownership or request for service procurement;
    it does not establish a current quote. Numeric partial stock is not treated
    as ownership of an entire differently sized requested quantity.
    """
    from .service_evidence import (_COUNT_WORDS, _ENTITY_BOUNDARY, _LITERAL_COUNT,
                                  _literal_material_phrase, _material_tokens, _number, _tokens)

    result = deepcopy(inputs) if isinstance(inputs, list) else []
    statements = list(_statements(messages if isinstance(messages, list) else []))
    for fact in result:
        if not isinstance(fact, dict):
            continue
        scope = fact.get("scope")
        if not isinstance(scope, dict):
            scope = {}; fact["scope"] = scope
        scope.pop("customer_supplied", None)
        scope.pop("supply_provenance", None)
        if (fact.get("kind") != "user_input" or fact.get("unit") in {"hour", "number", "percent"}
                or _number(fact.get("value")) is None):
            continue
        phrase = _literal_material_phrase(fact)
        requested = _material_tokens(phrase) if phrase else set()
        if not requested:
            continue
        head = next((word for word in reversed(_tokens(phrase)) if word in requested), None)
        for turn, supplied, noun, quote in statements:
            narrow_noun = _ENTITY_BOUNDARY.split(noun, maxsplit=1)[0]
            affirmed = _material_tokens(narrow_noun)
            # Operating equipment is not ownership of its replacement parts:
            # 'owns pumps' cannot establish possession of 'pump seals'.
            if head not in affirmed:
                continue
            counts = []
            for match in _LITERAL_COUNT.finditer(narrow_noun):
                token = match.group().casefold()
                counts.append(_COUNT_WORDS[token] if token in _COUNT_WORDS else token.replace(",", ""))
            if supplied and counts and not all(str(value) == str(fact["value"]) for value in counts):
                # Decimal equality handles actual numeric operands without
                # introducing a guessed stock-to-request conversion.
                from decimal import Decimal
                if any(Decimal(str(value)) != _number(fact["value"]) for value in counts):
                    continue
            scope["customer_supplied"] = supplied
            scope["supply_provenance"] = {"turn": turn, "quote": quote}
    return result


def annotate_quantity_supersession(inputs, messages):
    """Retire older literal quantities only after an explicit user correction.

    A shared head and compatible noun-token subsequence can connect a shorter
    correction to an earlier specific noun phrase. Additive requests, uncertain
    wording and matches spanning distinct old entities remain unresolved.
    IDs, values and source text are preserved for the conversation trace.
    """
    from .service_evidence import (_LITERAL_COUNT, _QUANTITY_UNITS, _clause_bounds,
                                  _literal_count_spans, _literal_material_phrase,
                                  _material_tokens, _number, _tokens)

    result = deepcopy(inputs) if isinstance(inputs, list) else []
    if not isinstance(messages, list):
        return result
    eligible = []
    for fact in result:
        if not isinstance(fact, dict) or not isinstance(fact.get("id"), str) or fact.get("kind") != "user_input" or fact.get("unit") not in _QUANTITY_UNITS:
            continue
        scope = fact.get("scope", {})
        value = _number(fact.get("value"))
        if not isinstance(scope, dict) or value is None or value < 0 or scope.get("role") in {"analytical_parameter", "superseded_input"}:
            continue
        text = fact.get("user_text")
        positions = [turn for turn, message in enumerate(messages) if isinstance(message, str) and message == text]
        turn = scope.get("turn")
        if type(turn) is not int or turn not in positions:
            turn = positions[0] if len(positions) == 1 else None
        if turn is None:
            continue
        spans = _literal_count_spans(fact)
        phrase = _literal_material_phrase(fact)
        if len(spans) != 1 or not phrase:
            continue
        phrase = re.split(r"\b(?:instead\s+of|rather\s+than)\b", phrase, maxsplit=1, flags=re.I)[0]
        material = _material_tokens(phrase)
        tokens = tuple(word for word in _tokens(phrase) if word in material)
        if not tokens:
            continue
        eligible.append((turn, spans[0][0], fact, tokens))

    def subsequence(left, right):
        remaining = iter(right)
        return all(word in remaining for word in left)

    for turn, start, new, tokens in sorted(eligible, key=lambda item: item[:2]):
        text = new["user_text"]
        left, right = _clause_bounds(text, start)
        clause = text[left:right]
        if (right < len(text) and text[right] == "?" or _UNCERTAIN.search(clause)
                or re.search(r"\b(?:add|extra|additional|more|another|or)\b", clause, re.I)):
            continue
        explicit = re.search(r"\b(?:actually|correction|make\s+that|i\s+meant|should\s+be|instead\s+of|rather\s+than)\b", clause, re.I)
        if not explicit:
            # "7 cords, not 5" is also an explicit correction. Do not borrow
            # a negation from a separate component joined by a conjunction.
            tail = text[start:right]
            explicit = any(_LITERAL_COUNT.match(tail[match.end():].rstrip(".!?; \t"))
                           for match in re.finditer(r",\s*not\s+", tail, re.I))
        if not explicit:
            continue
        older = [(old, old_tokens) for old_turn, old_start, old, old_tokens in eligible
                 if (old_turn, old_start) < (turn, start) and old.get("scope", {}).get("role") != "superseded_input"
                 and old.get("unit") == new.get("unit") and old_tokens[-1] == tokens[-1]
                 and (subsequence(tokens, old_tokens) or subsequence(old_tokens, tokens))]
        if len({old_tokens for _, old_tokens in older}) != 1:
            continue
        for old, _ in older:
            scope = old.setdefault("scope", {})
            scope["role"] = "superseded_input"
            scope["supersession_provenance"] = {"turn": turn, "quote": clause.strip(), "replaced_by_ref": new["id"]}
    return result
