"""Run actual autonomous turns; save private answers, tool evidence and checks.

Checks validate routing, clarifications, tool execution and evidence boundaries.
Semantic product correctness still requires inspection of cited original pages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re

from .agent import Assistant, BASELINE_RECOVERY
from . import WORK_DIR
from .model import LocalModel
from .services import json_numbers


def _engine_argument(value):
    if value == "baseline":
        raise argparse.ArgumentTypeError(BASELINE_RECOVERY)
    return value


def _final_generations(turn):
    """Each product subtask has its own final attempt, not one turn-wide maximum."""
    selected = {}
    for trace in turn.get("trace", []):
        if trace.get("operation") == "grounded_product_generation":
            key = trace.get("request_task_index")
            if key not in selected or trace.get("attempt", 0) >= selected[key].get("attempt", 0):
                selected[key] = trace
    return selected


def _identity(value):
    return re.sub(r"\W", "", str(value).casefold())


def _input_signature(fact):
    scope = fact.get("scope", {})
    return (fact.get("id"), Decimal(str(fact.get("value"))), fact.get("unit"),
            scope.get("role"), scope.get("turn"), tuple(scope.get("span", [])), fact.get("user_text"))


def _product_records(turn):
    from .product_identity import products_from_tool
    tool = turn.get("tool_result", {})
    tools = [tool, *(result.get("tool_result", {}) for result in tool.get("data", {}).get("task_results", []))]
    return {product["doc_id"]: product for selected in tools for product in products_from_tool(selected)}


def _cited_product_ids(tool, source_ids):
    """Resolve cited document identities through metadata, never filename names."""
    from .product_identity import products_from_tool
    data = tool.get("data", {})
    found = {passage["doc_id"] for passage in data.get("passages", [])
             if passage.get("doc_id") and passage.get("source_id") in source_ids}
    sources = [source for source in tool.get("sources", []) if source.get("id") in source_ids]
    found.update(source["doc_id"] for source in sources if source.get("doc_id"))
    for product in products_from_tool(tool):
        if product.get("source_file") and any(str(source.get("file")) == str(product["source_file"])
                or Path(str(source.get("file"))).name == Path(str(product["source_file"])).name for source in sources):
            found.add(product["doc_id"])
    return found


def _description_visible(data, registry, answer):
    """Check requested diagnostics against executed evidence, without reusing prose rendering."""
    from .adaptive import format_fact
    sections, profile = data.get("requested_sections", ["columns"]), data.get("profile", {})
    def number(key):
        fact = registry[profile[key]]
        if fact.get("operation") != "describe" or fact.get("metric") != key or not fact.get("source_ids"):
            raise ValueError("A diagnostic lacks its source-bound fact")
        return fact, format_fact({**fact, "unit": None})
    visible = True
    if "columns" in sections:
        visible &= bool(data.get("columns")) and all("- " + column["name"] + ": " + (column.get("unit") or "not stated") in answer
                                                     for column in data["columns"])
    if "coverage" in sections:
        coverage = data["coverage"]
        n, display = number("unique_valid_hours")
        if coverage.get("start") is None or coverage.get("end") is None:
            visible &= n["value"] == 0 and "Recorded period: unavailable" in answer
        else:
            visible &= (str(coverage["start"]) + " to " + str(coverage["end"]) in answer and
                        display + " distinct valid hourly observations" in answer)
    if "gaps" in sections:
        missing, missing_display = number("missing_hours")
        gaps, gap_display = number("timestamp_gap_count")
        entries = data.get("gaps", [])
        visible &= gaps["value"] == len(entries)
        visible &= Decimal(str(missing["value"])) == sum((Decimal(str(registry[entry["missing_hours_ref"]]["value"])) for entry in entries), Decimal(0))
        if gaps["value"]:
            visible &= missing_display + " missing hourly timestamps across " + gap_display + " recorded gap" in answer
            for entry in entries[:5]:
                fact = registry[entry["missing_hours_ref"]]
                visible &= (fact.get("operation") == "describe" and fact.get("metric") == "missing_hours_in_gap" and
                            str(entry["after"]) in answer and str(entry["before"]) in answer and
                            format_fact({**fact, "unit": None}) + " missing hours" in answer)
        elif number("unique_valid_hours")[0]["value"]:
            visible &= "no missing hourly timestamps" in answer
        else:
            visible &= "Timestamp gaps cannot be assessed" in answer
    return bool(visible)


def _process_checks(turn, messages):
    """Audit current bindings, actual result rows and displayed values separately.

    This verifies the execution contract. It is not an independent recomputation
    of historical statistics from the original CSV; source auditing does that.
    """
    from .adaptive import format_fact, request_facts
    from .process_request import binding_candidates
    data = turn.get("tool_result", {}).get("data", {})
    plan = data.get("analysis_plan")
    if data.get("engine") != "single-plan-computed-process" and not (isinstance(plan, dict) and "compare" in plan):
        return {}
    checks = {"process_actual_input_bindings": False, "process_exact_selected_results": False,
              "process_current_values_displayed": False}
    try:
        context = plan["_request_context"]
        actual_messages = messages[-6:]
        inputs = request_facts(actual_messages, "flotation")
        columns = context["columns"]
        if context["messages"] != actual_messages or sorted(map(_input_signature, context["inputs"])) != sorted(map(_input_signature, inputs)):
            return checks
        pairs, audit, ambiguities = binding_candidates(actual_messages, inputs, columns)
        allowed = {(entry["column"], entry["input_ref"]) for entry in pairs}
        recorded = {(entry["column"], entry["input_ref"]) for entry in plan.get("binding_evidence", [])}
        if recorded != allowed:
            return checks
        actions = [entry for entry in turn.get("trace", []) if entry.get("operation") == "adaptive_tool" and not entry.get("bootstrap")]
        if plan["status"] != "ready":
            valid = (turn["status"] != "answered" and not actions and
                     not any(plan[key] for key in ("compare", "distribution", "count", "describe")))
            return {key: valid for key in checks}
        requested, used = [], set()
        if plan["compare"]:
            requested.append({"action": "compare", "readings": plan["compare"]})
            used.update(entry["input_ref"] for entry in plan["compare"])
        if plan["distribution"]:
            requested.append({"action": "distribution", "columns": plan["distribution"]})
        for count in plan["count"]:
            requested.append({"action": "count", **count})
            used.update(entry["input_ref"] for entry in count["conditions"])
        if plan["describe"]:
            requested.append({"action": "describe"})
        selected_pairs = [*plan["compare"], *(condition for count in plan["count"] for condition in count["conditions"])]
        current = {entry["input_ref"] for entry in audit if entry["turn"] == len(actual_messages)-1}
        checks["process_actual_input_bindings"] = (not ambiguities and current <= used and
            all((entry["column"], entry["input_ref"]) in allowed for entry in selected_pairs))
        exact = ([entry.get("action") for entry in actions] == requested and
                 data.get("operations_executed") == len(requested))
        registry = {fact["id"]: fact for fact in data["facts"]}
        actual_inputs = {fact["id"]: fact for fact in inputs}
        visible = True
        for action, trace in zip(requested, actions):
            result = trace["result"]
            if result.get("status") == "invalid_request" or result.get("error"):
                exact = False
            for fact in result.get("facts", []):
                if registry.get(fact["id"]) != fact:
                    exact = False
            selected = result.get("data", {})
            if action["action"] == "compare":
                rows = selected.get("comparisons", [])
                exact &= [{key: row[key] for key in ("column", "input_ref")} for row in rows] == action["readings"]
                for row in rows:
                    source_input = actual_inputs[row["input_ref"]]
                    exact &= _input_signature(registry[row["input_ref"]]) == _input_signature(source_input)
                    if not row.get("available"):
                        continue
                    value = Decimal(str(source_input["value"]))
                    facts = {metric: registry[identifier] for metric, identifier in row["facts"].items()}
                    exact &= all(fact.get("column") == row["column"] for fact in facts.values())
                    exact &= row["input_ref"] in facts["empirical_percentile"].get("parents", [])
                    low, high = Decimal(str(facts["min"]["value"])), Decimal(str(facts["max"]["value"]))
                    q25, q75 = Decimal(str(facts["q25"]["value"])), Decimal(str(facts["q75"]["value"]))
                    exact &= row.get("range_status") == ("below_min" if value < low else "above_max" if value > high else "within_observed_range")
                    exact &= row.get("middle_50_status") == ("below_q25" if value < q25 else "above_q75" if value > q75 else "within_middle_50")
                    rendered = format_fact(source_input)
                    visible &= bool(re.search(r"^- " + re.escape(row["column"]) + r"(?: \([^\n]*\))?: supplied " + re.escape(rendered) + r";", turn["answer"], re.M))
            elif action["action"] == "distribution":
                rows = selected.get("distributions", [])
                exact &= [row["column"] for row in rows] == action["columns"]
                for row in rows:
                    facts = {metric: registry[identifier] for metric, identifier in row["facts"].items()}
                    count = format_fact({**facts["n"], "unit": None})
                    values = [format_fact(facts[metric]) if row.get("available") else "unavailable"
                              for metric in ("min", "q25", "median", "q75", "max", "mean")]
                    unit = "percent" if row["column"].startswith("%") else "not stated"
                    cells = [row["column"], unit, count, *values]
                    expected_row = "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cells) + " |"
                    visible &= expected_row in turn["answer"]
            elif action["action"] == "count":
                conditions = [{key: row[key] for key in ("column", "input_ref", "operator")} for row in selected.get("conditions", [])]
                exact &= conditions == action["conditions"] and selected.get("combine") == action["combine"]
                labels = []
                for condition in conditions:
                    value = format_fact(actual_inputs[condition["input_ref"]])
                    symbol = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "="}[condition["operator"]]
                    labels.append(re.escape(condition["column"]) + r"(?: \(units not stated\))? " + re.escape(symbol + " " + value))
                visible &= bool(re.search("For " + (" " + action["combine"].upper() + " ").join(labels) + ":", turn["answer"]))
                refs = selected["facts"]
                population, count = registry[refs["population"]], registry[refs["count"]]
                if population["value"] > 0:
                    summary_count = (format_fact({**count, "unit": None}) + " of " + format_fact({**population, "unit": None}) +
                                     " complete hourly observations (" + format_fact(registry[refs["percentage"]]) + ")")
                    visible &= summary_count in turn["answer"]
            elif action["action"] == "describe":
                visible &= _description_visible(selected, registry, turn["answer"])
        checks["process_exact_selected_results"] = bool(exact)
        checks["process_current_values_displayed"] = bool(visible)
    except (KeyError, TypeError, ValueError, InvalidOperation):
        pass
    return checks


def _product_checks(turn, current, previous=None, actual_previous_messages=()):
    """Verify requested tasks produced real sub-answers, including abstentions."""
    plan = turn.get("product_request_plan")
    if plan is None:
        return {}
    checks = {"product_declared_tasks_executed": False, "product_subanswers_evidenced": False,
              "product_explicit_requests_covered": False}
    try:
        data = turn["tool_result"]["data"]
        tasks, results = plan["tasks"], data["task_results"]
        coverage_traces = [trace for trace in turn["trace"] if trace.get("operation") == "product_request_coverage"]
        if not coverage_traces:
            return checks
        coverage = coverage_traces[-1]
        entries = coverage["coverage"]
        exact = (bool(tasks) and data["request_plan"] == plan and len(tasks) == len(results) == len(entries)
                 and coverage["requested_tasks"] == coverage["completed_tasks"] == len(tasks)
                 and bool(coverage.get("all_requested_tasks_addressed")))
        evidenced = True
        from .presentation import readable_citations
        final_sources = {source["id"] for source in turn.get("sources", [])}
        generations = _final_generations(turn)
        from .product_identity import name_occurs
        products = _product_records(turn)
        prior = (previous or {}).get("conversation_context", {}).get("product", {})
        prior_recommendations = {_identity(name) for name in [*prior.get("recommendations", []), prior.get("recommendation")] if name}
        kinds = set()
        for index, (task, result, entry) in enumerate(zip(tasks, results, entries)):
            kinds.add(task["kind"])
            exact &= (task["question_quote"] in current and bool(task["question_quote"].strip()) and
                      result["task"] == task and result.get("executed") is True and
                      entry["task_index"] == index and entry["kind"] == task["kind"] and
                      entry["subject_product"] == task["subject_product"] and
                      entry["status"] == result["status"] and entry["source_ids"] == result["source_ids"])
            text = result.get("answer")
            rendered = readable_citations(text, turn.get("sources", [])) if isinstance(text, str) else None
            evidenced &= isinstance(text, str) and bool(text.strip()) and rendered in turn["answer"]
            source_ids = set(result["source_ids"])
            evidenced &= source_ids <= final_sources and source_ids <= {source["id"] for source in result["tool_result"].get("sources", [])}
            if result["status"] == "answered":
                generation = generations.get(index)
                reviewed = any(trace.get("operation") == "reviewed_fact_rendering" and
                               trace.get("request_task_index") == index and trace.get("numerical_generation") is False
                               for trace in turn["trace"])
                evidenced &= bool(source_ids) and bool(reviewed or generation and not generation.get("validation_errors") and generation.get("raw", {}).get("status") == "answered")
                if generation:
                    recommendation = generation.get("raw", {}).get("recommendation")
                    if task["kind"] in {"describe", "explain", "specification", "visual", "compare"}:
                        evidenced &= recommendation is None
                    elif task["kind"] in {"recommend", "alternatives"}:
                        evidenced &= bool(recommendation)
                        cited_docs = _cited_product_ids(result["tool_result"], source_ids)
                        evidenced &= any(name_occurs(product, recommendation or "") for doc_id, product in products.items() if doc_id in cited_docs)
                    if task["kind"] == "alternatives" and recommendation:
                        evidenced &= _identity(recommendation) not in prior_recommendations
                if task["kind"] == "visual":
                    evidenced &= bool(generation and generation.get("visual_attachments"))
            elif result["status"] == "insufficient_evidence":
                evidenced &= bool(re.search(r"\b(?:not|no|cannot|could not|unable|unavailable|unsupported|unclear|insufficient|ambiguous|unverified)\b", text or "", re.I))
            else:
                evidenced = False
        complete = all(result.get("status") == "answered" for result in results)
        expected_status = "answered" if complete else "partial_answer" if any(result.get("status") == "answered" for result in results) else "insufficient_evidence"
        exact &= (turn.get("status") == expected_status and coverage.get("all_requested_tasks_completed", complete) == complete)
        witnesses = True
        alternative_mentions = [match for match in re.finditer(r"\b(?:other|different|additional|alternative|another)\s+(?:\w+\s+)?(?:recommendations?|options?|products?|solutions?|suggestions?)\b", current, re.I)
                                if not re.search(r"\b(?:no|not|without|don't want|do not want)\s*$", current[:match.start()], re.I)]
        if alternative_mentions:
            witnesses &= "alternatives" in kinds
            from .product_request import _terms
            request_only = all(not (_terms(task["question_quote"]) - {"any", "else", "there", "available", "please", "do", "have"})
                               for task in tasks if task["kind"] == "alternatives")
            if prior and request_only:
                need = prior.get("subject_query")
                current_context = turn.get("conversation_context", {}).get("product", {})
                checks["product_alternative_need_retained"] = bool(
                    isinstance(need, str) and need.strip() and
                    any(need in message for message in actual_previous_messages) and
                    current_context.get("subject_query") == need and plan.get("context_status") != "new_need")
        detail_request = bool(re.search(r"\b(?:tell\s+me\s+more|more\s+about|describe|details?)\b", current, re.I))
        if detail_request:
            witnesses &= bool(kinds & {"describe", "explain", "specification"})
        if re.search(r"\bwhy\b", current, re.I):
            witnesses &= "explain" in kinds
        visual_request = any(not re.search(r"\b(?:no|without|do not need|don't need)\s*$", current[:match.start()], re.I)
                             for match in re.finditer(r"\b(?:diagrams?|flowcharts?|flowsheets?|schematics?|layouts?|figures?|images?|pictures?)\b", current, re.I))
        comparison_request = bool(re.search(r"\b(?:compare|comparison|versus|vs|differences?)\b", current, re.I))
        if visual_request:
            witnesses &= "visual" in kinds
        if comparison_request:
            witnesses &= "compare" in kinds
        if detail_request or comparison_request or visual_request:
            mentioned = {doc_id for doc_id, product in products.items() if name_occurs(product, current)}
            addressed = set()
            for task, result in zip(tasks, results):
                if task["kind"] not in {"describe", "explain", "specification", "visual", "compare"}:
                    continue
                if task["kind"] == "compare":
                    if result["status"] == "answered":
                        addressed.update(doc_id for doc_id in _cited_product_ids(result["tool_result"], set(result["source_ids"]))
                                         if doc_id in products and name_occurs(products[doc_id], result.get("answer", "")))
                    else:
                        addressed.update(mentioned)
                elif task.get("subject_product"):
                    addressed.update(doc_id for doc_id, product in products.items() if name_occurs(product, task["subject_product"]))
            witnesses &= mentioned <= addressed
        checks.update(product_declared_tasks_executed=bool(exact), product_subanswers_evidenced=bool(evidenced),
                      product_explicit_requests_covered=bool(witnesses))
    except (KeyError, TypeError, ValueError):
        pass
    return checks


def _service_scope_checks(turn, messages):
    """An additive request retains active literal quantities, even if unpriced.

    This checks requested scope rather than requiring a historical rate for every
    item. Quantity corrections use the existing source-free supersession ledger.
    """
    if len(messages) < 2 or not re.search(r"\b(?:also|add|include|additional|extra)\b", messages[-1], re.I):
        return {}
    if re.search(r"\b(?:new|another|different|separate)\s+(?:customer|job|visit|site|plant|request|topic)\b", messages[-1], re.I):
        return {}
    from .adaptive import request_facts
    checks = {"service_additive_requested_quantities_retained": False}
    try:
        def eligible(fact):
            return (fact.get("kind") == "user_input" and fact.get("scope", {}).get("role") == "requested_quantity"
                    and fact.get("unit") not in {"number", "hour", "percent"})
        def signature(fact):
            return (fact.get("user_text"), tuple(fact.get("scope", {}).get("span", [])),
                    Decimal(str(fact["value"])), fact.get("unit"))
        expected = {signature(fact) for fact in request_facts(messages, "service_cost") if eligible(fact)
                    and fact.get("scope", {}).get("turn", len(messages)) < len(messages)-1}
        data = turn.get("tool_result", {}).get("data", {})
        actual = {signature(fact) for fact in data.get("facts", []) if eligible(fact)}
        checks["service_additive_requested_quantities_retained"] = expected <= actual
    except (TypeError, KeyError, ValueError, InvalidOperation):
        pass
    return checks


def _travel_checks(turn):
    """Validate country cohorts and alternative interval endpoints, without FX."""
    data = turn.get("tool_result", {}).get("data", {})
    context = data.get("travel_context")
    if not isinstance(context, dict):
        return {}
    checks = {"travel_exact_country_cohort": False, "travel_validated_rate_sources": False,
              "travel_endpoint_arithmetic": False, "travel_alternative_subtotals": False,
              "travel_context_displayed": False}
    try:
        registry = {fact["id"]: fact for fact in data.get("facts", [])}
        country = turn.get("tool_arguments", {}).get("country")
        wanted = country.strip().casefold() if isinstance(country, str) and country.strip() else None
        actual = context.get("country")
        actual = actual.strip().casefold() if isinstance(actual, str) and actual.strip() else None
        cohorts = context["cohorts"]
        rate_actions = [trace for trace in turn.get("trace", []) if trace.get("operation") == "adaptive_tool"
                        and trace.get("action", {}).get("action") == "travel_rates"]
        retrieved = {fact["id"]: fact for trace in rate_actions for fact in trace.get("result", {}).get("facts", [])}
        source_ids = {source["id"] for trace in rate_actions for source in trace.get("result", {}).get("sources", [])}
        cohort_ids = [reference for cohort in cohorts for reference in cohort["fact_ids"]]
        country_ok = (wanted == actual and set(cohort_ids) == set(retrieved)
                      and len(cohort_ids) == len(set(cohort_ids))
                      and all(str(trace["action"]["country"]).strip().casefold() == wanted for trace in rate_actions))
        source_ok, arithmetic, visible = True, True, True
        endpoint_ids, alternatives = set(), []
        known = [record for record in data.get("duration_inputs", [])
                 if record.get("role") == "travel_hours" and record.get("status") == "known"]
        for cohort in cohorts:
            rates = [registry[reference] for reference in cohort["fact_ids"]]
            low, high = Decimal(str(cohort["minimum"])), Decimal(str(cohort["maximum"]))
            country_ok &= bool(wanted) and cohort["country"].strip().casefold() == wanted
            source_ok &= bool(rates) and low <= high and cohort.get("range_kind") == "observed_min_max"
            for fact in rates:
                source_ok &= (fact == retrieved[fact["id"]] and fact.get("kind") == "hourly_rate" and fact.get("unit") == "money/hour"
                              and fact.get("currency") == cohort["currency"] and fact.get("scope", {}).get("validated") is True
                              and fact.get("scope", {}).get("category") == "travel"
                              and str(fact.get("scope", {}).get("country", "")).strip().casefold() == wanted
                              and bool(fact.get("source_ids")) and set(fact["source_ids"]) <= source_ids
                              and Decimal(str(fact["value"])).is_finite() and Decimal(str(fact["value"])) >= 0)
            source_ok &= low == min(Decimal(str(fact["value"])) for fact in rates)
            source_ok &= high == max(Decimal(str(fact["value"])) for fact in rates)
            refs = cohort["endpoint_refs"]
            source_ok &= bool(refs) and set(refs) <= set(cohort["fact_ids"])
            source_ok &= {Decimal(str(registry[reference]["value"])) for reference in refs} == {low, high}
            quantity_ref = cohort.get("quantity_ref")
            if quantity_ref:
                quantity = registry[quantity_ref]
                hours = Decimal(str(quantity["value"]))
                arithmetic &= (quantity.get("kind") == "user_input" and quantity.get("unit") == "hour" and
                               quantity.get("scope", {}).get("role") == "travel_hours" and hours.is_finite() and hours >= 0 and
                               any(Decimal(str(record["total_hours"])) == hours for record in known))
                cost_refs = [cohort["amount_min_ref"], cohort["amount_max_ref"]]
                for amount_ref, rate in zip(cost_refs, (low, high)):
                    amount = registry[amount_ref]
                    arithmetic &= (amount.get("kind") == "calculation" and amount.get("unit") == "money" and
                                   amount.get("currency") == cohort["currency"] and quantity_ref in amount.get("parents", []) and
                                   any(reference in amount.get("parents", []) and Decimal(str(registry[reference]["value"])) == rate for reference in refs) and
                                   Decimal(str(amount["value"])) == (rate * hours).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                endpoint_ids.update(cost_refs)
                if low == high:
                    arithmetic &= cost_refs[0] == cost_refs[1]
                else:
                    alternatives.append(set(cost_refs))
            else:
                arithmetic &= not known and cohort.get("amount_min_ref") is None and cohort.get("amount_max_ref") is None
            answer = turn.get("answer", "")
            visible &= (str(cohort["country"]).casefold() in answer.casefold() and "travel" in answer.casefold()
                        and cohort["currency"] in answer and f"{low:,.2f}" in answer and f"{high:,.2f}" in answer)
        arithmetic &= (endpoint_ids == set(context["endpoint_fact_ids"]) and
                       len(context["endpoint_fact_ids"]) == len(set(context["endpoint_fact_ids"])))
        travel_amounts = {fact["id"] for fact in registry.values() if fact.get("kind") == "calculation"
                          and fact.get("unit") == "money" and fact.get("scope", {}).get("operation") == "multiply"
                          and any(str(token).startswith("billed:travel_hours:") for token in fact.get("scope", {}).get("coverage", []))}
        arithmetic &= travel_amounts == endpoint_ids
        if context["status"] != "available":
            country_ok &= not cohorts and not retrieved and not endpoint_ids
            country_ok &= context["status"] == ("missing_country" if not wanted else "unavailable")
            answer = turn.get("answer", "").casefold()
            visible &= "travel" in answer and ("country" in answer if not wanted else wanted in answer)
        if not known or context["status"] != "available":
            arithmetic &= not endpoint_ids and "travel" in data.get("unpriced_categories", [])

        def ancestors(reference, visiting=None):
            visiting = set() if visiting is None else visiting
            if reference in visiting:
                raise ValueError("Cyclic subtotal ancestry")
            fact = registry[reference]
            found = {reference} & endpoint_ids
            if fact.get("kind") == "calculation" and fact.get("scope", {}).get("operation") == "sum":
                parents = [registry[parent] for parent in fact.get("parents", [])]
                if fact.get("unit") == "money" and (
                        any(parent.get("unit") != "money" or parent.get("currency") != fact.get("currency") for parent in parents) or
                        Decimal(str(fact["value"])) != sum((Decimal(str(parent["value"])) for parent in parents), Decimal(0))):
                    raise ValueError("A money subtotal changes currency or does not equal its selected charges")
                for parent in fact.get("parents", []):
                    found |= ancestors(parent, visiting | {reference})
            return found

        ranges_ok = True
        groups = [data["subtotal_ranges_by_currency"], *data["category_subtotal_ranges_by_currency"].values()]
        for group in groups:
            for currency, bounds in group.items():
                lower, upper = Decimal(str(bounds["minimum"])), Decimal(str(bounds["maximum"]))
                ranges_ok &= lower <= upper
                for reference, value in ((bounds["minimum_ref"], lower), (bounds["maximum_ref"], upper)):
                    fact = registry[reference]
                    ranges_ok &= fact.get("unit") == "money" and fact.get("currency") == currency and Decimal(str(fact["value"])) == value
                    included = ancestors(reference)
                    ranges_ok &= all(not alternative <= included for alternative in alternatives)
        # No intermediate sum may turn min/max alternatives into two charges.
        for fact in registry.values():
            if fact.get("kind") == "calculation" and fact.get("scope", {}).get("operation") == "sum":
                included = ancestors(fact["id"])
                ranges_ok &= all(not alternative <= included for alternative in alternatives)
        checks.update(travel_exact_country_cohort=bool(country_ok), travel_validated_rate_sources=bool(source_ok),
                      travel_endpoint_arithmetic=bool(arithmetic), travel_alternative_subtotals=bool(ranges_ok),
                      travel_context_displayed=bool(visible))
    except (KeyError, TypeError, ValueError, InvalidOperation):
        pass
    return checks


def _fixture_checks(case, turns):
    """Apply authored customer-input/task contracts without an answer catalogue."""
    from .adaptive import format_fact
    from .product_identity import name_occurs
    checks = {}

    def check(label, index, predicate):
        try:
            checks[f"{label}_turn_{index}"] = bool(predicate(turns[int(index)]))
        except (IndexError, KeyError, TypeError, ValueError, InvalidOperation):
            checks[f"{label}_turn_{index}"] = False

    def results(turn):
        return turn.get("tool_result", {}).get("data", {}).get("task_results", [])

    def no_process_calculation(turn):
        data = turn.get("tool_result", {}).get("data", {})
        plan = data.get("analysis_plan", {})
        return (turn.get("route") == "flotation" and turn.get("status") == "insufficient_evidence" and
                data.get("operations_executed") == 0 and plan.get("status") == "insufficient_evidence" and
                not any(plan.get(key) for key in ("compare", "distribution", "count", "describe")) and
                not any(trace.get("operation") == "adaptive_tool" and not trace.get("bootstrap")
                        for trace in turn.get("trace", [])))

    def money_uses(turn, *, role=None, category=None):
        data = turn.get("tool_result", {}).get("data", {})
        registry = {fact["id"]: fact for fact in data.get("facts", [])}
        if category and (data.get("category_subtotal_refs_by_currency", {}).get(category) or
                         data.get("category_subtotal_ranges_by_currency", {}).get(category)):
            return True
        def matches(reference, visiting=()):
            if reference in visiting:
                raise ValueError("Cyclic money ancestry")
            fact = registry[reference]
            coverage = fact.get("scope", {}).get("coverage", [])
            if role and (fact.get("kind") == "user_input" and fact.get("scope", {}).get("role") == role or
                         any(str(token).startswith("billed:" + role + ":") for token in coverage)):
                return True
            if category and (any(str(token).startswith(category + ":") or
                                 str(token).startswith("billed:" + category + "_hours:") for token in coverage) or
                             fact.get("scope", {}).get("category") == category or
                             category == "material" and fact.get("kind") == "unit_price"):
                return True
            return any(matches(parent, (*visiting, reference)) for parent in fact.get("parents", []))
        return any(matches(fact["id"]) for fact in registry.values()
                   if fact.get("kind") == "calculation" and fact.get("unit") == "money")

    for index in case.get("requires_recommendation_turns", []):
        check("required_recommendation", index, lambda turn: any(
            trace.get("raw", {}).get("status") == "answered" and trace["raw"].get("recommendation") and not trace.get("validation_errors")
            for trace in _final_generations(turn).values()))
    for index in case.get("requires_clarification_turns", []):
        def clarification(turn):
            question = turn.get("tool_result", {}).get("question")
            return isinstance(question, str) and "?" in question and question.strip() in turn.get("answer", "")
        check("required_clarification", index, clarification)
    for index in case.get("requires_new_request_boundary_turns", []):
        check("new_request_boundary", index, lambda turn: any(
            trace.get("operation") == "current_request_boundary" and trace.get("new_request") is True for trace in turn.get("trace", [])))
    for index, roles in case.get("forbidden_priced_duration_roles_by_turn", {}).items():
        check("rejected_duration_not_priced", index, lambda turn, roles=roles: bool(roles) and all(not money_uses(turn, role=role) for role in roles))
    for index, categories in case.get("forbidden_charged_categories_by_turn", {}).items():
        check("excluded_category_not_charged", index, lambda turn, categories=categories: bool(categories) and all(not money_uses(turn, category=category) for category in categories))
    for index, required in case.get("required_product_tasks_by_turn", {}).items():
        check("required_product_tasks", index, lambda turn, required=required: bool(required) and all(any(
            result.get("task", {}).get("kind") == kind and result.get("executed") is True and
            result.get("status") in {"answered", "insufficient_evidence"} and result.get("answer") and
            result["answer"] in turn.get("answer", "") for result in results(turn)) for kind in required))
    for index, subjects in case.get("required_product_subjects_by_turn", {}).items():
        def subjects_evidenced(turn, subjects=subjects):
            products = _product_records(turn)
            return bool(subjects) and all(any(
                name_occurs(product, subject) and name_occurs(product, result.get("answer", "")) and
                doc_id in _cited_product_ids(result.get("tool_result", {}), set(result.get("source_ids", [])))
                for doc_id, product in products.items() for result in results(turn) if result.get("status") == "answered") for subject in subjects)
        check("required_product_subjects", index, subjects_evidenced)
    for index in case.get("requires_visual_evidence_or_abstention_turns", []):
        def visual_result(turn):
            selected = [(index, result) for index, result in enumerate(results(turn)) if result.get("task", {}).get("kind") == "visual"]
            generations = _final_generations(turn)
            return bool(selected) and all(
                result.get("status") == "insufficient_evidence" and bool(re.search(
                    r"\b(?:not|cannot|unavailable|unsupported|unclear|insufficient|ambiguous)\b", result.get("answer", ""), re.I)) or
                result.get("status") == "answered" and bool(generations.get(task_index, {}).get("visual_attachments")) and
                bool(result.get("source_ids")) and set(result["source_ids"]) <= set(generations.get(task_index, {}).get("visual_source_ids", []))
                for task_index, result in selected)
        check("visual_evidence_or_abstention", index, visual_result)
    for index in case.get("requires_no_process_calculation_turns", []):
        check("unsupported_process_not_calculated", index, no_process_calculation)
    for index, kinds in case.get("required_process_unsupported_kinds_by_turn", {}).items():
        def unsupported(turn, kinds=kinds):
            issues = turn.get("tool_result", {}).get("data", {}).get("analysis_plan", {}).get("unsupported_requests", [])
            return (bool(kinds) and no_process_calculation(turn) and set(kinds) <= {issue.get("kind") for issue in issues} and
                    all(issue.get("explanation") and issue["explanation"] in turn.get("answer", "") for issue in issues))
        check("required_process_capability_decline", index, unsupported)
    for index, sections in case.get("required_process_description_sections_by_turn", {}).items():
        def descriptions(turn, sections=sections):
            data = turn.get("tool_result", {}).get("data", {})
            registry = {fact["id"]: fact for fact in data.get("facts", [])}
            return bool(sections) and any(
                set(sections) <= set(trace.get("result", {}).get("data", {}).get("requested_sections", [])) and
                _description_visible(trace["result"]["data"], registry, turn.get("answer", ""))
                for trace in turn.get("trace", []) if trace.get("operation") == "adaptive_tool" and trace.get("action", {}).get("action") == "describe")
        check("required_process_diagnostics", index, descriptions)
    for index, expected in case.get("expected_process_readings_by_turn", {}).items():
        def readings(turn, expected=expected):
            registry = {fact["id"]: fact for fact in turn.get("tool_result", {}).get("data", {}).get("facts", [])}
            comparisons = [row for trace in turn.get("trace", []) if trace.get("operation") == "adaptive_tool" and
                           trace.get("action", {}).get("action") == "compare" for row in trace["result"]["data"]["comparisons"]]
            actual = {row["column"]: Decimal(str(registry[row["input_ref"]]["value"])) for row in comparisons}
            return bool(expected) and actual == {column: Decimal(str(value)) for column, value in expected.items()}
        check("expected_current_process_readings", index, readings)
    for index in case.get("requires_shared_process_population_header_turns", []):
        def shared_header(turn):
            registry = {fact["id"]: fact for fact in turn.get("tool_result", {}).get("data", {}).get("facts", [])}
            populations = [registry[row["facts"]["n"]] for trace in turn.get("trace", [])
                           if trace.get("operation") == "adaptive_tool" and trace.get("action", {}).get("action") == "compare"
                           for row in trace["result"]["data"]["comparisons"] if row.get("available")]
            if not populations or len({fact["value"] for fact in populations}) != 1:
                return False
            display = format_fact({**populations[0], "unit": None})
            return (turn.get("answer", "").count("From " + display + " hourly historical observations:") == 1 and
                    not re.search(r"; [\d,]+ hourly observations\.", turn.get("answer", "")))
        check("shared_process_population_header", index, shared_header)
    return checks


def evaluate_case(case: dict, assistant: Assistant) -> dict:
    assistant.reset()
    turns = [assistant.turn(message) for message in case["messages"]]
    checks = {"routes": [r["route"] for r in turns] == case["expected_routes"],
              "statuses": (len(turns) == len(case["allowed_statuses"]) and all(turn["status"] in allowed for turn, allowed in zip(turns, case["allowed_statuses"]))
                           if "allowed_statuses" in case else [r["status"] for r in turns] == case["expected_statuses"]),
              "no_errors": all(r["status"] != "error" for r in turns),
              "tools_executed": all("tool_result" in r for r in turns if r["route"] in {"product", "service_cost", "flotation"}),
              "answers_with_evidence": all(bool(r["sources"]) for r in turns if r["status"] in {"answered", "partial_answer"})}
    final = turns[-1]
    for index in case.get("requires_source_turns", []):
        checks[f"source_turn_{index}"] = bool(turns[index].get("sources"))
    for index in case.get("requires_calculation_turns", []):
        checks[f"calculation_turn_{index}"] = any(t["operation"] == "adaptive_tool" and t["action"]["action"] in {"calculate", "price_items", "compare", "distribution", "count"}
                                                    and t["result"].get("facts") for t in turns[index]["trace"])
    for index in case.get("must_not_claim_full_estimate_turns", []):
        response = turns[index].get("tool_result", {}).get("data", {}).get("response", {})
        checks[f"partial_scope_turn_{index}"] = response.get("scope") in {"partial_components", "historical_comparison", "insufficient", "clarification"}
    if "expected_travel_countries" in case:
        expected = case["expected_travel_countries"]
        checks["expected_travel_countries"] = len(expected) == len(turns) and all(
            (turn.get("tool_result", {}).get("data", {}).get("travel_context") or {}).get("country") == country
            for turn, country in zip(turns, expected))
    for index in case.get("requires_country_travel_turns", []):
        context = turns[index].get("tool_result", {}).get("data", {}).get("travel_context") or {}
        checks[f"country_travel_available_turn_{index}"] = context.get("status") == "available" and bool(context.get("cohorts"))
    for index in case.get("requires_unpriced_travel_turns", []):
        data = turns[index].get("tool_result", {}).get("data", {})
        checks[f"travel_unpriced_turn_{index}"] = "travel" in data.get("unpriced_categories", []) and not (data.get("travel_context") or {}).get("endpoint_fact_ids")
    for key, require_range in (("requires_travel_range_turns", True), ("requires_travel_point_turns", False)):
        for index in case.get(key, []):
            cohorts = (turns[index].get("tool_result", {}).get("data", {}).get("travel_context") or {}).get("cohorts", [])
            checks[f"travel_{'range' if require_range else 'point'}_turn_{index}"] = bool(cohorts) and all(
                cohort.get("amount_min_ref") and cohort.get("amount_max_ref") for cohort in cohorts) and (
                any(Decimal(str(cohort["minimum"])) < Decimal(str(cohort["maximum"])) for cohort in cohorts) if require_range else
                all(Decimal(str(cohort["minimum"])) == Decimal(str(cohort["maximum"])) for cohort in cohorts))
    if any(turn.get("engine") == "adaptive" for turn in turns):
        checks["adaptive_result_rendering"] = all(any(trace["operation"] == "deterministic_evidence_rendering" for trace in turn["trace"])
            for turn in turns if turn["route"] in {"service_cost", "flotation"} and turn["status"] == "answered")
        checks["declared_analysis_completed"] = all(not turn.get("tool_result", {}).get("data", {}).get("incomplete_analysis")
            for turn in turns if turn["status"] == "answered")
    if case.get("requires_recommendation"):
        generated = [trace["raw"] for trace in _final_generations(final).values() if not trace.get("validation_errors")]
        checks["specific_source_bound_recommendation"] = any(bool(result.get("recommendation")) for result in generated)
    if case.get("requires_visual"):
        checks["source_image_attached"] = any(t.get("visual_attachments") for r in turns for t in r["trace"]
                                              if t["operation"] == "grounded_product_generation")
    if case.get("expected_intents"):
        checks["response_intents"] = [r.get("response_intent") for r in turns] == case["expected_intents"]
    for index in case.get("no_recommendation_turns", []):
        checks[f"no_new_recommendation_turn_{index}"] = all(t["raw"].get("recommendation") is None for t in turns[index]["trace"]
                                                            if t["operation"] == "grounded_product_generation" and not t["validation_errors"])
    if case.get("requires_travel_scenarios"):
        checks["assumed_travel_labelled"] = bool(final.get("tool_result", {}).get("data", {}).get("travel_is_hypothetical"))
    if case.get("requires_material_examples"):
        data = final.get("tool_result", {}).get("data", {})
        checks["separate_material_examples"] = bool(data.get("material_illustrations") or any(b.get("with_cloths_subtotal") for b in data.get("budget_ranges", [])))
    checks["readable_citations"] = all(not re.search(r"\[(?:B[0-9a-f]{12}|S\d+|P\d+)(?:[,\] ])", r["answer"]) for r in turns)
    for key, value in case.get("expected_final_fields", {}).items():
        checks["final_field_" + key] = final.get("tool_arguments", {}).get(key) == value
    active_process_messages, active_service_messages = [], []
    for index, (message, turn) in enumerate(zip(case["messages"], turns)):
        if any(trace.get("operation") == "current_request_boundary" and trace.get("new_request") for trace in turn.get("trace", [])):
            active_process_messages = []
        if turn["route"] == "flotation":
            active_process_messages.append(message)
            for key, value in _process_checks(turn, active_process_messages).items():
                checks[f"{key}_turn_{index}"] = value
        else:
            active_process_messages = []
        if turn["route"] == "product":
            for key, value in _product_checks(turn, message, turns[index-1] if index else None, case["messages"][:index]).items():
                checks[f"{key}_turn_{index}"] = value
        if turn["route"] == "service_cost":
            additive = bool(re.search(r"\b(?:also|add|include|additional|extra)\b", message, re.I))
            if not additive and any(trace.get("operation") == "current_request_boundary" and trace.get("new_request") for trace in turn.get("trace", [])):
                active_service_messages = []
            active_service_messages.append(message)
            for key, value in _service_scope_checks(turn, active_service_messages).items():
                checks[f"{key}_turn_{index}"] = value
            for key, value in _travel_checks(turn).items():
                checks[f"{key}_turn_{index}"] = value
        else:
            active_service_messages = []
    checks["product_citations_and_numeric_gate"] = all(not trace.get("validation_errors") for turn in turns
                                                        for trace in _final_generations(turn).values())
    checks.update(_fixture_checks(case, turns))
    return {"case_id": case["id"], "checks": checks, "contract_checks_passed": all(checks.values()), "turns": turns}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/regression-cases.json"))
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path, default=WORK_DIR / "evaluation.json")
    parser.add_argument("--store", type=Path, default=WORK_DIR / "prepared/knowledge.sqlite")
    parser.add_argument("--raw-dir", type=Path, default=WORK_DIR / "raw")
    parser.add_argument("--service-index", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct-q4_K_M")
    parser.add_argument("--retrieval", choices=("keyword", "semantic", "hybrid"), default="keyword")
    parser.add_argument("--semantic-index", type=Path, default=WORK_DIR / "brochure-semantic.json")
    parser.add_argument("--engine", type=_engine_argument, choices=("adaptive",), default="adaptive")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output file; live evidence is not overwritten")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.case:
        selected = set(args.case)
        if selected - {c["id"] for c in cases}:
            parser.error("Unknown case identifiers")
        cases = [c for c in cases if c["id"] in selected]
    assistant = Assistant(store_path=args.store, data_dir=args.data_dir, raw_dir=args.raw_dir,
                          service_index=args.service_index, model=LocalModel(args.endpoint, args.model),
                          retrieval_mode=args.retrieval, semantic_index=args.semantic_index, embedding_endpoint=args.endpoint, engine=args.engine)
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "actual_model_calls": True, "production_tools_executed": True,
              "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model": args.model, "endpoint": args.endpoint,
              "retrieval_mode": args.retrieval,
              "engine": args.engine,
              "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
              "store_sha256": hashlib.sha256(args.store.read_bytes()).hexdigest(),
              "limitations": ["Contract checks are not a corpus-wide answer accuracy estimate.", "Product paraphrases and visual relationships require source inspection."], "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        print("Running " + case["id"], flush=True)
        result = evaluate_case(case, assistant)
        report["results"].append(result)
        print("  " + ("PASS" if result["contract_checks_passed"] else "FAILED") + "; " + str(round(sum(t["wall_seconds"] for t in result["turns"]), 2)) + "s", flush=True)
        args.output.write_text(json_numbers(report) + "\n", encoding="utf-8")
    report["summary"] = {"cases_run": len(cases), "turns_run": sum(len(r["turns"]) for r in report["results"]),
                         "contract_cases_passed": sum(r["contract_checks_passed"] for r in report["results"])}
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json_numbers(report) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    return 0 if report["summary"]["contract_cases_passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
