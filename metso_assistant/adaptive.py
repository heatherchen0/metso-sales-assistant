"""Typed request interpretation and source-bound numerical execution.

The runner interprets once, then executes reusable checked operations in code;
it has no repeated action-planning loop.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re
from pathlib import Path


WORDS = {word: number for number, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMBER = re.compile(r"(?<![\w.\-])(?:[+\-\u2212]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)|" + "|".join(WORDS) + r")(?!\w)", re.I)
REQUEST_UNITS = {"pairs": "pair", "pair": "pair", "tubes": "tube", "tube": "tube",
                 "sets": "set", "set": "set", "kits": "kit", "kit": "kit",
                 "bottles": "bottle", "bottle": "bottle", "rolls": "roll", "roll": "roll",
                 "boxes": "box", "box": "box", "lots": "lot", "lot": "lot",
                 "ml": "ml", "millilitres": "ml", "milliliters": "ml", "litres": "litre", "liters": "litre"}


def request_facts(messages: list[str], route: str, known: dict | None = None) -> list[dict]:
    """Numeric operands must originate in actual user messages, not the model.

    The spans remain visible so the model can choose the current relevant
    operand. Explicit travel interpretation uses the existing provenance parser.
    """
    from .conversation import _travel_value
    from .service_request import duration_literal_spans
    output = []
    for turn, message in enumerate(messages):
        date_spans = [match.span() for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", message)]
        duration_spans = duration_literal_spans(message) if route == "service_cost" else ()
        travel, _ = _travel_value(message, [], "")
        for match in NUMBER.finditer(message):
            if any(a <= match.start() < b for a, b in date_spans):
                continue
            if route == "flotation" and re.search(r"\bcolumn\s+$", message[:match.start()], re.I):
                continue
            token = match.group().casefold()
            canonical = token.replace("\u2212", "-")
            canonical = canonical.replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", canonical) else canonical.replace(",", ".")
            value = Decimal(WORDS[token]) if token in WORDS else Decimal(canonical)
            if token in WORDS and re.search(r"\b(?:minus|negative)\s+$", message[:match.start()], re.I):
                value = -value
            around = message[max(0, match.start()-35):min(len(message), match.end()+65)]
            suffix = message[match.end():match.end()+18]
            unit = "percent" if re.match(r"\s*(?:%|percent\b)", suffix, re.I) else "number"
            role = "reading" if route == "flotation" else "requested_quantity"
            if route == "flotation" and re.search(r"\b(?:middle|percentile|quantile)\s+$", message[:match.start()], re.I):
                role = "analytical_parameter"
            duration = re.match(r"\s*(?:(on[- ]?site|labou?r|installation|repair)\s+)?(?:hours?|hrs?|h)\b", suffix, re.I)
            if duration or match.span() in duration_spans:
                # A raw journey duration is not necessarily billed round-trip.
                unit, role = "number", "ambiguous_duration"
                before = message[max(0, match.start()-35):match.start()]
                context = list(re.finditer(r"\b(?:on[- ]?site|labou?r|installation|repair|travel|trip|journey)\b", before, re.I))
                if (duration and duration[1]) or (context and context[-1].group().casefold() not in {"travel", "trip", "journey"}):
                    unit, role = "hour", "labour_hours"
            elif route == "service_cost" and unit == "number":
                unit = "each"
                immediate = re.match(r"\s*([A-Za-z]+)\b", suffix)
                if immediate and immediate[1].casefold() in REQUEST_UNITS:
                    unit = REQUEST_UNITS[immediate[1].casefold()]
                if re.match(r"\s*(?:(?:USD|EUR|CAD|AUD|SEK|ZAR)\s+)?(?:totals?|subtotals?|sums?|results?|visits?|trips?|reports?|records?|scenarios?|currencies)\b", suffix, re.I):
                    unit, role = "number", "analytical_parameter"
            if re.search(r"\bnot\s+$", message[:match.start()], re.I):
                unit, role = "number", "superseded_input"
            output.append({"id": f"U{len(output)+1:03d}", "label": around,
                           "value": value, "unit": unit, "currency": None,
                           "kind": "user_input", "user_text": message,
                           "source_ids": [], "parents": [], "assumptions": [],
                           "scope": {"role": role, "turn": turn, "span": [match.start(), match.end()],
                                     "entity_text": re.split(r"\b(?:and|or|for|per|each)\b|[,.;!?]", message[match.end():match.end()+100], maxsplit=1, flags=re.I)[0].strip()}})
        if route == "service_cost" and travel is not None:
            output.append({"id": f"U{len(output)+1:03d}", "label": "Explicit round-trip travel hours: " + message,
                           "value": Decimal(str(travel)), "unit": "hour", "currency": None,
                           "kind": "user_input", "user_text": message,
                           "source_ids": [], "parents": [], "assumptions": [],
                           "scope": {"role": "travel_hours", "turn": turn}})
    # A bare numerical reply to a travel-only question has an explicit basis in
    # the validated conversational state, even though its text omits 'hours'.
    verified_travel = (known or {}).get("round_trip_travel_hours")
    if route == "service_cost" and "round_trip_travel_hours" in (known or {}):
        output = [fact for fact in output if fact.get("scope", {}).get("role") != "travel_hours" or
                  verified_travel is not None and fact["value"] == Decimal(str(verified_travel))]
    if route == "service_cost" and verified_travel is not None and not any(
            fact["unit"] == "hour" and fact["value"] == Decimal(str(verified_travel)) for fact in output):
        output.append({"id": f"U{len(output)+1:03d}", "label": "Round-trip travel confirmed in this conversation",
                       "value": Decimal(str(verified_travel)), "unit": "hour", "currency": None,
                       "kind": "user_input", "user_text": messages[-1] if messages else "",
                       "source_ids": [], "parents": [], "assumptions": [], "scope": {"role": "travel_hours"}})
    for index, fact in enumerate(output, 1):
        fact["id"] = f"U{index:03d}"
    if route == "service_cost":
        from .request_scope import annotate_material_supply, annotate_quantity_supersession
        output = annotate_quantity_supersession(annotate_material_supply(output, messages), messages)
    return output


def format_fact(fact):
    value = Decimal(str(fact["value"]))
    unit, currency = fact.get("unit"), fact.get("currency")
    if currency:
        number = f"{value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP):,.2f}"
        denominator = unit.split("/", 1)[1] if isinstance(unit, str) and unit.startswith("money/") else None
        return f"{currency} {number}" + ("/h" if denominator == "hour" else " each" if denominator == "each" else " per " + denominator if denominator else "")
    if value == value.to_integral_value():
        number = f"{int(value):,}"
    else:
        number = format(value.quantize(Decimal(".0001"), rounding=ROUND_HALF_UP), "f").rstrip("0").rstrip(".")
    return number + ("%" if unit in {"percent", "%"} else " percentage points" if unit == "percentage_point" else " h" if unit in {"hour", "hours"} else "")


def active_duration_inputs(facts, categories=None):
    """Keep the latest supplied hours for each requested billing role."""
    requested = set(categories or ["labour", "travel"])
    roles = {category + "_hours" for category in requested if category in {"labour", "travel"}}
    result = []
    for role in sorted(roles):
        values = [fact for fact in facts.values() if fact.get("kind") == "user_input" and fact.get("unit") == "hour"
                  and fact.get("scope", {}).get("role") == role]
        if values:
            latest = max(fact.get("scope", {}).get("turn", -1) for fact in values)
            result.extend(fact for fact in values if fact.get("scope", {}).get("turn", -1) == latest)
    return result


class AdaptiveRunner:
    def __init__(self, model, *, store_path, csv_path, index_path=None, service_sources=None,
                 retrieval_mode="keyword", service_semantic_index=None, embedding_endpoint="http://127.0.0.1:11435"):
        self.model, self.store_path, self.csv_path = model, Path(store_path), Path(csv_path)
        self.index_path = index_path
        self.service_sources = service_sources
        self.retrieval_mode, self.service_semantic_index, self.embedding_endpoint = retrieval_mode, service_semantic_index, embedding_endpoint

    def run(self, route, messages, known=None, *, pending_fields=(), pending_question="", duration_clarifications=()):
        """Interpret once, then execute validated reusable operations in code."""
        inputs = request_facts(messages, route, known)
        if route == "flotation":
            from .process_ops import ProcessWorkspace
            from .process_request import interpret_process_request, execute_process_request
            workspace = ProcessWorkspace(self.csv_path, inputs=inputs)
            metadata = workspace.describe()
            columns = [column for column in metadata.get("data", {}).get("columns", []) if column.get("type") != "datetime"]
            plan, traces = interpret_process_request(self.model, messages, inputs, columns)
            result, execution = execute_process_request(workspace, plan, metadata=metadata, messages=messages)
            return result, [*traces, *execution]
        if route != "service_cost":
            raise ValueError("Unsupported evidence domain")
        from .service_request import bind_duration_facts, resolve_duration_inputs
        from .service_evidence import ServiceWorkspace
        from .service_execution import execute_service_request
        from .analysis_plan import select_analysis
        durations = resolve_duration_inputs(messages, pending_fields=pending_fields, pending_question=pending_question,
                                           clarifications=duration_clarifications)
        inputs = bind_duration_facts(inputs, messages, durations, pending_fields=pending_fields,
                                     pending_question=pending_question, clarifications=duration_clarifications)
        workspace = ServiceWorkspace(self.store_path, index_path=self.index_path, inputs=inputs,
                                     source_dir=self.service_sources)
        plan, traces = select_analysis(self.model, route, messages, inputs=inputs)
        traces.append({"operation": "service_duration_interpretation",
                       "records": [duration.as_dict() for duration in durations]})
        result, execution = execute_service_request(workspace, plan, messages, known,
            retrieval_mode=self.retrieval_mode, semantic_index=self.service_semantic_index,
            endpoint=self.embedding_endpoint, model=self.model, duration_inputs=durations)
        return result, [*traces, *execution]
