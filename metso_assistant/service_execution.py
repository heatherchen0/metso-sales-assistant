"""Execute one service scope with reusable retrieval and checked arithmetic.

The model declares the requested scope once. Source metadata, literal component
bindings and dimensional checks drive execution; no repeated arithmetic planner.
"""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from decimal import Decimal
import re

from .service_evidence import (material_quantity_matches, quantity_relations,
    duration_scaling_matches, _literal_material_phrase, _material_tokens,
    _work_modes, _tokens, _literal_count_spans)


def price_items(workspace, items):
    if not isinstance(items, list) or not 1 <= len(items) <= 6:
        raise ValueError("Price between one and six explicitly referenced items")
    pairs = []
    for item in items:
        if (not isinstance(item, dict) or set(item) not in ({"rate_ref", "quantity_ref"}, {"rate_ref", "quantity_ref", "repeat_ref"})
                or not all(isinstance(item.get(name), str) and item[name].strip() for name in ("rate_ref", "quantity_ref"))
                or item.get("repeat_ref") is not None and (not isinstance(item["repeat_ref"], str) or not item["repeat_ref"].strip())):
            raise ValueError("Each item requires rate_ref and quantity_ref identifiers and an optional repeat_ref identifier or null")
        pair = (item["rate_ref"], item["quantity_ref"], item.get("repeat_ref"))
        if pair in pairs:
            raise ValueError("Duplicate priced items cannot form a subtotal")
        pairs.append(pair)
    facts, sources, warnings, outcomes, grouped = {}, {}, [], [], defaultdict(list)
    for rate, quantity, repeat in pairs:
        base = {"rate_ref": rate, "quantity_ref": quantity, "repeat_ref": repeat}
        effective_quantity = quantity
        if repeat is not None:
            composed = workspace.calculate("quantity_product", [quantity, repeat])
            warnings.extend(composed.get("warnings", []))
            if composed.get("error") or not composed.get("facts"):
                outcomes.append({**base, "error": composed.get("error", "No supported repeated quantity")})
                continue
            quantity_fact = composed["facts"][-1]
            facts[quantity_fact["id"]] = quantity_fact
            effective_quantity = quantity_fact["id"]
            base["effective_quantity_ref"] = effective_quantity
        result = workspace.calculate("multiply", [rate, effective_quantity])
        warnings.extend(result.get("warnings", []))
        sources.update({source["id"]: source for source in result.get("sources", [])})
        produced = result.get("facts", [])
        if result.get("error") or not produced:
            outcomes.append({**base, "error": result.get("error", "No supported price")})
            continue
        fact = produced[-1]
        facts[fact["id"]] = fact
        grouped[fact["currency"]].append(fact["id"])
        outcomes.append({**base, "amount_ref": fact["id"]})
    subtotals, errors = {}, []
    for currency, references in grouped.items():
        if len(references) == 1:
            subtotals[currency] = references[0]
            continue
        result = workspace.calculate("sum", references)
        warnings.extend(result.get("warnings", []))
        if result.get("error") or not result.get("facts"):
            errors.append(result.get("error", "Components could not be safely summed"))
            continue
        fact = result["facts"][-1]
        facts[fact["id"]] = fact
        subtotals[currency] = fact["id"]
        sources.update({source["id"]: source for source in result.get("sources", [])})
    complete = not errors and all("amount_ref" in outcome for outcome in outcomes)
    return {"facts": list(facts.values()), "sources": list(sources.values()), "warnings": list(dict.fromkeys(warnings)),
            "status": "ok" if complete else "partial", "data": {"items": outcomes,
                "subtotal_refs_by_currency": subtotals, "complete_for_selected_items": complete,
                "exclusions": "Only the selected line items; other requested or unpriced work is excluded.", "sum_errors": errors}}


_MODE_WORDS = re.compile(r"\breplac(?:e|es|ed|ing|ement|ements)\b|\binspect(?:s|ed|ing|ion|ions)?\b", re.I)
_WORK_BOUNDARY = re.compile(r"[;.!?\n]|\b(?:and|also|then|plus|but)\b", re.I)


def _requested_work_modes(messages):
    """Explicit generic work requests, respecting later explicit exclusions."""
    modes = set()
    for text in messages:
        for clause in _WORK_BOUNDARY.split(text):
            for match in _MODE_WORDS.finditer(clause):
                named = _work_modes(match.group())
                prefix = clause[max(0, match.start()-40):match.start()]
                suffix = clause[match.end():match.end()+25]
                excluded = (re.search(r"\b(?:not|don['’]t|do\s+not|without|skip|exclude)\b", prefix, re.I)
                            or re.match(r"\s+(?:is\s+)?not\s+(?:required|needed|requested)\b", suffix, re.I))
                if excluded: modes.difference_update(named)
                else: modes.update(named)
    return modes


def _quantity_work_modes(quantity, facts, _visited=None):
    """Bind a literal count to its own work clause before considering extras."""
    literal = facts.get(quantity.get("scope", {}).get("per_item_ref"), quantity)
    visited = set() if _visited is None else set(_visited)
    if literal.get("id") in visited:
        return set()
    visited.add(literal.get("id"))
    text = literal.get("user_text") or ""
    spans = _literal_count_spans(literal)
    modes = set()
    for start, _ in spans:
        boundaries = list(_WORK_BOUNDARY.finditer(text))
        left = max((match.end() for match in boundaries if match.end() <= start), default=0)
        right = min((match.start() for match in boundaries if match.start() > start), default=len(text))
        local = _requested_work_modes([text[left:right]])
        if not local:
            # A verb can govern a conjunctive component list: replace 2 seals
            # and 3 rotors. It is not an invitation to resize a different task.
            prior = list(_MODE_WORDS.finditer(text[:start]))
            if prior:
                local = _requested_work_modes([text[prior[-1].start():start]])
        modes.update(local)
    if not modes:
        # A literal correction changes the count, not its requested work:
        # "make that six modules" still belongs to the prior replacement.
        # Follow the source-free parser's proven correction lineage rather
        # than borrowing a verb from an unrelated message or source report.
        for earlier in facts.values():
            provenance = earlier.get("scope", {}).get("supersession_provenance", {})
            if (earlier.get("kind") == "user_input" and earlier.get("unit") == literal.get("unit")
                    and provenance.get("replaced_by_ref") == literal.get("id")
                    and provenance.get("quote") and provenance["quote"] in text):
                modes.update(_quantity_work_modes(earlier, facts, visited))
    return modes


def _installation_matches(task, quantity, facts):
    """An inspection duration cannot stand in for explicitly requested installation."""
    literal = facts.get(quantity.get("scope", {}).get("per_item_ref"), quantity)
    requested = str(literal.get("user_text") or "")
    installation = r"\binstall(?:ation|ed|ing|s)?\b"
    return not re.search(installation, requested, re.I) or bool(re.search(installation, task.get("label", ""), re.I))


def evidence_rank(fact, known, quantity=None, facts=None):
    """Rank historical comparables, never certify present compatibility."""
    scope = fact.get("scope", {})
    equipment = set(_tokens((known or {}).get("equipment") or ""))
    observed_equipment = set(_tokens(scope.get("equipment") or ""))
    # A model identifier may be printed in the component description while the
    # report's equipment heading names only its product family. Preserve that
    # identity signal before ranking a generic noun match from another machine.
    source_identity = observed_equipment | set(scope.get("entity_tokens") or []) | set(_tokens(fact.get("label") or ""))
    model_overlap = sum(any(char.isdigit() for char in token) for token in equipment & source_identity)
    country = str((known or {}).get("country") or "").casefold()
    exact_count = False
    component_fit = Decimal(0)
    if quantity and facts:
        literal = facts.get(quantity.get("scope", {}).get("per_item_ref"), quantity)
        # Routing can include the component noun inside its equipment string.
        # Only source catalogue equipment words may be removed from the literal
        # requested part, so that such extraction cannot erase its identity.
        catalogue_equipment = _material_tokens(str(scope.get("equipment") or "") + " "
                                               + str(scope.get("equipment_family") or ""))
        wanted = _material_tokens(_literal_material_phrase(literal) or "") - catalogue_equipment
        described = set(scope.get("entity_tokens") or []) - catalogue_equipment
        if wanted and described:
            component_fit = Decimal(len(wanted & described)) / Decimal(len(wanted | described))
        observed = facts.get(scope.get("quantity_ref"), {})
        exact_count = observed.get("value") == quantity.get("value") and observed.get("unit") == quantity.get("unit")
    return (-model_overlap, -len(equipment & observed_equipment), -component_fit,
            -int(bool(country) and str(scope.get("country", "")).casefold() == country),
            -int(exact_count), fact.get("report_id") or "", fact.get("id") or "")


def execute_service_request(workspace, plan, messages, known=None, *, retrieval_mode="keyword",
                            semantic_index=None, endpoint="http://127.0.0.1:11435", model=None,
                            duration_inputs=()):
    """Return the existing service tool envelope and full private trace."""
    from .service_answer import service_answer
    from .service_semantic import search
    from .service_reports import find_reports

    history, traces, inspected = [], [], set()
    travel, work_scope = None, None
    known = known or {}
    def record(action, result):
        history.append({"action": action, "result": result})
        traces.append({"operation": "adaptive_tool", "action": action, "result": deepcopy(result)})
        return result

    def retrieve(query):
        result, runtime = search(workspace, query[:6000], country=known.get("country"), equipment=known.get("equipment"),
            mode=retrieval_mode, semantic_index=semantic_index, endpoint=endpoint, model=model)
        traces.extend(runtime)
        return record({"action": "search", "query": query[:6000]}, result)

    def inspect(identifier):
        if identifier and identifier not in inspected:
            inspected.add(identifier)
            return record({"action": "read_report", "report_id": identifier}, workspace.read_report(identifier))

    query = "\n".join(messages[-3:])
    if plan["purpose"] == "original_visit_totals":
        discovery = record({"action": "find_reports", "query": query}, find_reports(workspace, query))
        candidates = discovery.get("data", {}).get("report_candidates", [])
        countries = discovery.get("data", {}).get("requested_countries", [])
        chosen = []
        if countries:
            for country in countries:
                candidate = next((item for item in candidates if item.get("country") == country), None)
                if candidate:
                    chosen.append(candidate)
        else:
            chosen = candidates[:2]
        for candidate in chosen:
            inspect(candidate["report_id"])
    else:
        retrieve(query)
        categories = set(plan["requested_categories"])
        requested_modes = _requested_work_modes(messages)
        work_scope = {"requested_modes": sorted(requested_modes), "priced_modes": [],
                      "unpriced_modes": [], "original_task_refs": []}
        literal = [fact for fact in workspace.facts.values() if fact.get("kind") == "user_input"
                   and fact.get("scope", {}).get("role") == "requested_quantity"]
        # Compose proven work quantities even when the customer owns the parts.
        effective, anchors = {}, set()
        if plan.get("quantity_structure") == "per_equipment":
            for relation in quantity_relations(literal):
                if relation["equipment_count_ref"] != plan.get("equipment_count_ref"):
                    continue
                refs = [relation["per_item_ref"], relation["equipment_count_ref"]]
                result = record({"action": "calculate", "operation": "quantity_product", "refs": refs},
                                workspace.calculate("quantity_product", refs))
                if result.get("facts"):
                    effective[refs[0]] = result["facts"][-1]
                    anchors.add(refs[1])
        quantities = [effective.get(fact["id"], fact) for fact in literal if fact["id"] not in anchors
                      and fact["id"] != plan.get("equipment_count_ref")]
        chosen_reports = []
        for quantity in quantities[:6]:
            parent = workspace.facts.get(quantity.get("scope", {}).get("per_item_ref"), quantity)
            phrase = _literal_material_phrase(parent)
            if not phrase:
                continue
            retrieve(phrase + " " + str(known.get("equipment") or ""))
            rates = [fact for fact in workspace.facts.values() if fact.get("kind") == "unit_price"
                     and fact.get("scope", {}).get("validated") is True
                     and material_quantity_matches(fact, quantity, workspace.facts)]
            # Ownership removes purchase operands, not source lookup for work.
            if not rates and parent.get("scope", {}).get("customer_supplied") is True:
                probe = deepcopy(parent); probe["scope"].pop("customer_supplied", None)
                rates = [fact for fact in workspace.facts.values() if fact.get("kind") == "unit_price"
                         and fact.get("scope", {}).get("validated") is True
                         and material_quantity_matches(fact, probe, workspace.facts)]
            if rates:
                rate = min(rates, key=lambda fact: evidence_rank(fact, known, quantity, workspace.facts))
                chosen_reports.append(rate["report_id"])
                inspect(rate["report_id"])
                if "material" in categories and parent.get("scope", {}).get("customer_supplied") is not True:
                    record({"action": "price_items", "items": [{"rate_ref": rate["id"], "quantity_ref": quantity["id"], "repeat_ref": None}]},
                        price_items(workspace, [{"rate_ref": rate["id"], "quantity_ref": quantity["id"], "repeat_ref": None}]))
        if not chosen_reports:
            source_facts = [fact for fact in workspace.facts.values() if fact.get("report_id")]
            for fact in sorted(source_facts, key=lambda item: evidence_rank(item, known)):
                if len(inspected) >= 2:
                    break
                inspect(fact["report_id"])
        preferred = list(dict.fromkeys(chosen_reports))

        def hourly(role, report=None):
            rates = [fact for fact in workspace.facts.values() if fact.get("kind") == "hourly_rate"
                     and fact.get("scope", {}).get("validated") is True
                     and fact.get("scope", {}).get("category") == role.removesuffix("_hours")]
            if not rates:
                retrieve(role.replace("_", " ") + " " + str(known.get("equipment") or ""))
                rates = [fact for fact in workspace.facts.values() if fact.get("kind") == "hourly_rate"
                         and fact.get("scope", {}).get("validated") is True
                         and fact.get("scope", {}).get("category") == role.removesuffix("_hours")]
            return min(rates, key=lambda fact: (0 if report and fact.get("report_id") == report else 1,
                0 if fact.get("report_id") in preferred else 1, *evidence_rank(fact, known))) if rates else None

        supplied_roles = set()
        for duration in duration_inputs:
            if duration.status != "known" or duration.role.removesuffix("_hours") not in categories:
                continue
            if duration.role == "travel_hours":
                continue  # Country cohort bounds are priced separately below.
            fact = next((item for item in workspace.facts.values() if item.get("kind") == "user_input"
                         and item.get("scope", {}).get("role") == duration.role and item.get("unit") == "hour"), None)
            if fact:
                supplied_roles.add(duration.role)
                rate = hourly(duration.role)
                if rate:
                    items = [{"rate_ref": rate["id"], "quantity_ref": fact["id"], "repeat_ref": None}]
                    record({"action": "price_items", "items": items}, price_items(workspace, items))

        if "labour" in categories and "labour_hours" not in supplied_roles:
            tasks = [fact for fact in workspace.facts.values() if fact.get("kind") == "task_duration"]
            observed = [fact for fact in workspace.facts.values() if fact.get("kind") == "quantity"]
            billed_tasks = set()
            covered_modes, quantity_modes = set(), set()
            for quantity in quantities:
                modes = _quantity_work_modes(quantity, workspace.facts)
                quantity_modes.update(modes)
                candidates = [(task, count) for task in tasks for count in observed
                    if duration_scaling_matches(task, quantity, count, workspace.facts)
                    and _installation_matches(task, quantity, workspace.facts)
                    and (not modes or not _work_modes(task["label"]) or modes & _work_modes(task["label"]))]
                candidates.sort(key=lambda pair: (0 if pair[0].get("report_id") in preferred else 1,
                    *evidence_rank(pair[0], known), pair[1]["id"]))
                selected = next((pair for pair in candidates if pair[0]["id"] not in billed_tasks), None)
                if selected:
                    task, count = selected
                    refs = [task["id"], quantity["id"], count["id"]]
                    result = record({"action": "calculate", "operation": "scale_duration", "refs": refs},
                                    workspace.calculate("scale_duration", refs))
                    if result.get("facts"):
                        rate = hourly("labour_hours", task["report_id"])
                        if rate:
                            items = [{"rate_ref": rate["id"], "quantity_ref": result["facts"][-1]["id"], "repeat_ref": None}]
                            priced = record({"action": "price_items", "items": items}, price_items(workspace, items))
                            if priced.get("facts"):
                                billed_tasks.add(task["id"])
                                source_modes = _work_modes(task["label"])
                                # Coverage follows the checked task/count
                                # operation, not an exact verb in its label.
                                # A noun such as "component change" can have
                                # no recognized source mode. Its conditional
                                # estimate still prices the one literal work
                                # request it was validated against; it does
                                # not establish several distinct work modes.
                                covered_modes.update(source_modes or (modes if len(modes) == 1 else set()))
            # Additional explicitly named work can be illustrated at its original
            # task scope; it is never silently resized to an unrelated part count.
            # First-turn and follow-up requests use the same decomposition.
            # Counts with unsupported scaling do not become original-scope
            # examples merely because a similarly named task was retrieved.
            additional = requested_modes - covered_modes - quantity_modes
            for mode in sorted(additional):
                matches = [task for task in tasks if mode in _work_modes(task["label"]) and task["id"] not in billed_tasks]
                if matches:
                    task = min(matches, key=lambda fact: (0 if fact.get("report_id") in preferred else 1, *evidence_rank(fact, known)))
                    rate = hourly("labour_hours", task["report_id"])
                    if rate:
                        items = [{"rate_ref": rate["id"], "quantity_ref": task["id"], "repeat_ref": None}]
                        priced = record({"action": "price_items", "items": items}, price_items(workspace, items))
                        if priced.get("facts"):
                            billed_tasks.add(task["id"])
                            covered_modes.update(_work_modes(task["label"]))
                            work_scope["original_task_refs"].append(task["id"])
            work_scope["priced_modes"] = sorted(covered_modes)
            work_scope["unpriced_modes"] = sorted(requested_modes - covered_modes)
        if "travel" in categories:
            from .service_travel import travel_context
            duration = next((item for item in duration_inputs if item.role == "travel_hours"), None)
            travel, travel_traces = travel_context(workspace, known.get("country"), duration)
            traces.extend(travel_traces)
            history.extend({"action": item["action"], "result": item["result"]} for item in travel_traces
                           if item.get("operation") == "adaptive_tool")
    result, composed = service_answer(workspace, history, messages, known, plan, duration_inputs=duration_inputs,
                                      travel_context=travel, work_scope=work_scope)
    traces.extend({"operation": "adaptive_tool", **entry, "result_composition": True} for entry in composed)
    result["data"].update(engine="request-contract-v2", analysis_plan=plan, facts=list(workspace.facts.values()),
                          operations_executed=len(history), execution="deterministic-request-plan", travel_context=travel)
    traces.append({"operation": "deterministic_evidence_rendering", "numerical_generation": False,
                   "selected_fact_ids": result["data"].get("selected_fact_ids", []), "scope": result["data"]["response"]["scope"]})
    return result, traces
