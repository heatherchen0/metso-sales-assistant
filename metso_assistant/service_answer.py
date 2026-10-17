"""Concise presentation of checked service arithmetic and historical rates."""
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal

from .presentation import render_sources
from .service_request import duration_question, resolve_duration_inputs


def _source_listing(sources):
    grouped = {}
    for source in sources:
        key = (source.get("report_id") or source.get("source_file"), source.get("source_sha256"))
        if key not in grouped:
            grouped[key] = deepcopy(source); grouped[key]["lines"] = []
        grouped[key]["lines"].extend(source.get("lines", []))
    return render_sources(list(grouped.values()))


def _hourly_cohorts(workspace, categories, known, preferred_currencies):
    grouped = defaultdict(list)
    for fact in workspace.facts.values():
        scope = fact.get("scope", {})
        if (fact.get("kind") == "hourly_rate" and fact.get("unit") == "money/hour"
                and scope.get("validated") is True and scope.get("category") in categories
                and isinstance(fact.get("value"), Decimal) and fact["value"].is_finite()):
            grouped[(scope.get("category"), scope.get("country"), fact.get("currency"))].append(fact)
    result = []
    for (category,country,currency),facts in grouped.items():
        facts.sort(key=lambda fact:(fact["value"],fact["id"]))
        result.append({"category":category,"country":country,"currency":currency,
                       "minimum":facts[0]["value"],"maximum":facts[-1]["value"],
                       "fact_ids":[fact["id"] for fact in facts],
                       "endpoint_refs":list(dict.fromkeys([facts[0]["id"],facts[-1]["id"]])),
                       "equipment_families":sorted({fact["scope"].get("equipment_family") or "unclassified" for fact in facts}),
                       "reports":sorted({fact.get("report_id") for fact in facts if fact.get("report_id")}),
                       "range_kind":"observed_min_max"})
    return sorted(result,key=lambda group:(group["country"] != (known or {}).get("country"),
                                           group["currency"] not in preferred_currencies,
                                           group["category"],group["country"] or "",group["currency"] or ""))


def _quantity_text(fact, format_fact):
    unit = fact.get("unit")
    return ("about " if fact.get("scope",{}).get("duration_estimated") is True else "") + format_fact(fact) + (" " + unit if unit not in {None,"each","hour"} else "")


def _labour_label(quantity, facts):
    if quantity.get("kind") == "task_duration":
        return "Labour — " + quantity["label"] + " (original task example)"
    if quantity.get("scope",{}).get("operation") == "scale_duration":
        return "Labour (estimated hours)"
    if quantity.get("scope",{}).get("duration_estimated") is True:
        return "Labour (user-estimated hours)"
    return "Labour"


def _equipment_label(value):
    text = str(value).splitlines()[0] if value else ""
    return text if len(text) <= 120 else text[:117].rstrip()+"…"


def service_answer(workspace, history, messages, known=None, plan=None, duration_inputs=(), travel_context=None,
                   work_scope=None):
    from .adaptive import format_fact, active_duration_inputs
    from .service_evidence import quantity_relations

    selected, derived, paragraphs = {}, [], []
    details, scope_notes, applied_countries = [], [], set()
    assumptions = sorted({assumption for fact in workspace.facts.values()
                          if fact.get("kind") == "calculation" for assumption in fact.get("assumptions",[])})
    categories = set((plan or {}).get("requested_categories",["material"]))
    records = tuple(duration_inputs) if duration_inputs else resolve_duration_inputs(messages)
    question = None
    visit_comparisons = []
    original = (plan or {}).get("purpose") == "original_visit_totals" or (
        plan is None and any(entry.get("action",{}).get("action") == "find_reports" for entry in history))
    travel_requested = "travel" in categories and not original
    travel_groups = []
    if travel_requested and travel_context is not None and travel_context.get("status") == "available" and travel_context.get("country"):
        travel_groups = deepcopy(travel_context.get("cohorts", []))
    endpoint_ids = set((travel_context or {}).get("endpoint_fact_ids", []))
    endpoint_ids.update(reference for group in (travel_context or {}).get("cohorts", [])
                        for reference in (group.get("amount_min_ref"),group.get("amount_max_ref")) if reference)
    def billed_category(fact):
        rate = next((workspace.facts[reference] for reference in fact.get("parents", [])
                     if reference in workspace.facts and str(workspace.facts[reference].get("unit", "")).startswith("money/")), {})
        category = rate.get("scope", {}).get("category")
        return category if category in {"labour","travel"} else "material"
    amounts = [] if original else [fact for fact in workspace.facts.values()
        if fact.get("kind") == "calculation" and fact.get("unit") == "money"
        and fact.get("scope",{}).get("operation") == "multiply" and fact["id"] not in endpoint_ids
        and (travel_context is None or billed_category(fact) in categories)]
    travel_amounts = {}
    if travel_requested and travel_context is not None:
        for group in travel_groups:
            if group.get("category") != "travel" or str(group.get("country")).casefold() != str(travel_context.get("country")).casefold():
                raise ValueError("Travel range must use the requested country and travel category")
            rates = [workspace.facts[reference] for reference in group["fact_ids"]]
            if not rates or any(rate.get("kind") != "hourly_rate" or rate.get("unit") != "money/hour"
                                or rate.get("scope", {}).get("category") != "travel"
                                or rate.get("scope", {}).get("country") != group["country"]
                                or rate.get("currency") != group["currency"]
                                or rate.get("scope", {}).get("validated") is not True for rate in rates):
                raise ValueError("Travel range requires validated same-country same-currency hourly references")
            endpoints = group.get("endpoint_refs", [])
            if not endpoints or any(reference not in group["fact_ids"] for reference in endpoints):
                raise ValueError("Travel rate endpoints must belong to the complete country cohort")
            low_rate,high_rate = workspace.facts[endpoints[0]],workspace.facts[endpoints[-1]]
            if low_rate["value"] != min(rate["value"] for rate in rates) or high_rate["value"] != max(rate["value"] for rate in rates):
                raise ValueError("Travel rate endpoints must be the observed country minimum and maximum")
            group.update(minimum=low_rate["value"], maximum=high_rate["value"],
                         endpoint_refs=list(dict.fromkeys([low_rate["id"],high_rate["id"]])))
            low_ref, high_ref = group.get("amount_min_ref"), group.get("amount_max_ref")
            if not low_ref and not high_ref:
                continue
            if not low_ref or not high_ref or not group.get("quantity_ref"):
                raise ValueError("A priced travel range requires both checked endpoints and supplied hours")
            low, high = workspace.facts[low_ref], workspace.facts[high_ref]
            for amount, rate in ((low,low_rate),(high,high_rate)):
                if (amount.get("kind") != "calculation" or amount.get("unit") != "money"
                        or amount.get("currency") != group["currency"]
                        or not {rate["id"],group["quantity_ref"]} <= set(amount.get("parents",[]))):
                    raise ValueError("Travel endpoint must be a checked amount using its hourly rate and supplied hours")
            if group["currency"] in travel_amounts:
                raise ValueError("Only one observed country travel range is allowed per currency")
            travel_amounts[group["currency"]] = (low,high,group)
    by_currency = defaultdict(list)
    for fact in amounts: by_currency[fact["currency"]].append(fact)
    subtotal_cache = {}
    def checked_subtotal(items):
        if len(items) == 1: return items[0]
        key = tuple(sorted(fact["id"] for fact in items))
        if key in subtotal_cache: return subtotal_cache[key]
        refs = [fact["id"] for fact in items]
        result = workspace.calculate("sum",refs)
        derived.append({"action":{"action":"calculate","operation":"sum","refs":refs},"result":result})
        if result.get("error"):
            details.append("These amounts overlap; they are alternatives, not an overall subtotal.")
        value = result["facts"][-1] if result.get("facts") and not result.get("error") else None
        subtotal_cache[key] = value
        return value

    subtotals = {currency:subtotal for currency,items in by_currency.items() if (subtotal:=checked_subtotal(items)) is not None}
    amount_categories = {}
    priced_service_categories = set()
    grouped_amounts = {"service":defaultdict(list),"material":defaultdict(list)}
    for fact in amounts:
        rate = next((workspace.facts[reference] for reference in fact.get("parents",[])
                     if str(workspace.facts[reference].get("unit","")).startswith("money/")),{})
        category = "service" if rate.get("scope",{}).get("category") in {"labour","travel"} else "material"
        if category == "service": priced_service_categories.add(rate["scope"]["category"])
        amount_categories[fact["id"]] = category
        grouped_amounts[category][fact["currency"]].append(fact)
    category_subtotals = {category:{currency:subtotal for currency,items in groups.items()
                                   if (subtotal:=checked_subtotal(items)) is not None}
                          for category,groups in grouped_amounts.items()}
    subtotal_ranges, category_ranges = {}, {"service":{}, "material":{}}
    def range_record(low,high):
        return {"minimum_ref":low["id"],"maximum_ref":high["id"],
                "minimum":low["value"],"maximum":high["value"],"range_kind":"observed_min_max"}
    for currency,(low,high,group) in travel_amounts.items():
        lower = checked_subtotal([*by_currency.get(currency,[]),low])
        upper = checked_subtotal([*by_currency.get(currency,[]),high])
        service_lower = checked_subtotal([*grouped_amounts["service"].get(currency,[]),low])
        service_upper = checked_subtotal([*grouped_amounts["service"].get(currency,[]),high])
        if lower is not None and upper is not None:
            subtotal_ranges[currency] = range_record(lower,upper)
            if lower["value"] == upper["value"]: subtotals[currency] = lower
            else: subtotals.pop(currency,None)
        if service_lower is not None and service_upper is not None:
            category_ranges["service"][currency] = range_record(service_lower,service_upper)
            if service_lower["value"] == service_upper["value"]: category_subtotals["service"][currency] = service_lower
            else: category_subtotals["service"].pop(currency,None)
        priced_service_categories.add("travel")
    for currency,fact in subtotals.items():
        subtotal_ranges.setdefault(currency,range_record(fact,fact))
    for category,groups in category_subtotals.items():
        for currency,fact in groups.items():
            category_ranges[category].setdefault(currency,range_record(fact,fact))

    def cite_many(facts):
        facts = list(facts)
        selected.update((fact["id"],fact) for fact in facts)
        reports = sorted({workspace.sources[source].get("report_id") for fact in facts for source in fact.get("source_ids",[])
                          if source in workspace.sources and workspace.sources[source].get("report_id")})
        return " [" + "; ".join(reports) + "]" if reports else ""

    def cite(fact):
        return cite_many([fact])

    def format_range(record):
        low,high = workspace.facts[record["minimum_ref"]],workspace.facts[record["maximum_ref"]]
        low_text,high_text = format_fact(low),format_fact(high)
        if low["value"] == high["value"]: return low_text+cite(low)
        prefix = str(high.get("currency") or "")+" "
        if prefix.strip() and high_text.startswith(prefix): high_text=high_text[len(prefix):]
        return low_text+"–"+high_text+cite_many([low,high])

    billed_amounts = [*amounts,*(fact for low,high,group in travel_amounts.values() for fact in (low,high))]
    coverage = {token for fact in billed_amounts for token in fact.get("scope",{}).get("coverage",[])}
    unpriced_roles = {category for category in categories & {"labour","travel"}
                     if not any(token.startswith("billed:"+category+"_hours:") for token in coverage)}
    supplied = active_duration_inputs(workspace.facts,categories)
    used = {reference for amount in billed_amounts for reference in amount.get("parents",[])}
    for fact in supplied:
        if fact["id"] not in used: unpriced_roles.add(fact["scope"]["role"].removesuffix("_hours"))
    if "labour" in categories and (work_scope or {}).get("unpriced_modes"):
        unpriced_roles.add("labour")
    if not original: question = duration_question(records,unpriced_roles)
    rate_cohorts, labour_basis = [], []
    status, scope = "insufficient_evidence", "insufficient"
    # Comparison facts have their own operation/context and never enter the
    # requested amounts or currency subtotals calculated above.
    comparison_reports = set()
    for amount in amounts:
        operands = [workspace.facts[reference] for reference in amount.get("parents", [])]
        rate = next((item for item in operands if item.get("kind") == "hourly_rate"
                     and item.get("scope", {}).get("category") == "labour"), None)
        quantity = next((item for item in operands if item is not rate), {})
        task = quantity if quantity.get("kind") == "task_duration" else next((workspace.facts[reference]
            for reference in quantity.get("parents", []) if workspace.facts[reference].get("kind") == "task_duration"), {})
        if rate and task and task.get("report_id") not in comparison_reports and task.get("report_id") == rate.get("report_id"):
            comparison_reports.add(task["report_id"])
            comparison = workspace.visit_labour_comparison(task["id"], rate["id"])
            derived.append({"action": {"action": "visit_labour_comparison", "task_ref": task["id"], "rate_ref": rate["id"]},
                            "result": comparison})
            if comparison.get("status") == "available":
                visit_comparisons.append(comparison["data"])

    if original:
        reads = {entry["action"]["report_id"] for entry in history if entry.get("action",{}).get("action") == "read_report" and not entry.get("result",{}).get("error")}
        countries = {country for entry in history for country in entry.get("result",{}).get("data",{}).get("requested_countries",[])}
        totals = [fact for fact in workspace.facts.values() if fact.get("scope",{}).get("type") == "reported_total"
                  and fact.get("report_id") in reads and (not countries or fact["scope"].get("country") in countries)]
        paragraphs.append("Recorded whole-visit costs for their original scopes:" if totals else "No matching original whole-visit total has been inspected.")
        if totals:
            paragraphs.append("\n".join("- "+", ".join(str(value) for value in (fact.get("report_id"),fact["scope"].get("country"),_equipment_label(fact["scope"].get("equipment"))) if value)
                                          +": **"+format_fact(fact)+"**."+cite(fact) for fact in totals))
        missing = countries-{fact["scope"].get("country") for fact in totals}
        if missing: details.append("No inspected total for: "+", ".join(sorted(missing))+".")
        scope_notes.append("Historical comparisons, not a new quotation. Currencies stay separate; no exchange rate is provided.")
        scope, status = "historical_comparison", "ok" if totals and not missing else "insufficient_evidence"
    else:
        if billed_amounts:
            service_subtotals, material_subtotals = category_subtotals["service"],category_subtotals["material"]
            service_ranges = category_ranges["service"]
            service_currencies = list(dict.fromkeys([*service_subtotals,*service_ranges]))
            task_labour = any(billed_category(fact) == "labour" and any(
                workspace.facts[reference].get("kind") == "task_duration" or
                workspace.facts[reference].get("scope", {}).get("operation") == "scale_duration"
                for reference in fact.get("parents", [])) for fact in amounts)
            if service_currencies:
                service_title = "labour and billed travel subtotal" if priced_service_categories == {"labour","travel"} else "labour subtotal" if "labour" in priced_service_categories else "billed travel subtotal"
                if task_labour:
                    service_title = service_title.replace("labour and", "labour tasks and").replace("labour subtotal", "labour task subtotal")
                omitted = (categories & {"labour","travel"}) - priced_service_categories
                if omitted: service_title += "; "+" and ".join(sorted(omitted))+" not yet priced"
                paragraphs.append("; ".join("**"+format_range(service_ranges[currency])+"**" if currency in service_ranges else "**"+format_fact(service_subtotals[currency])+"**"+cite(service_subtotals[currency]) for currency in service_currencies)+" — "+service_title+".")
                if task_labour:
                    focus = "requested replacement-task labour" if (work_scope or {}).get("requested_modes") == ["replacement"] else "the stated task labour"
                    paragraphs.append("Assumption: the task estimate covers "+focus+" only, with other visit work excluded.")
            if material_subtotals:
                material_text = "; ".join("**"+format_fact(fact)+"**"+cite(fact) for fact in material_subtotals.values())
                paragraphs.append(("Optional material addition: " if service_currencies else "Conditional historical material subtotal: ")+material_text+", if these materials must be purchased.")
            if service_currencies and material_subtotals:
                combined = [fact for currency,fact in subtotals.items() if currency not in subtotal_ranges]
                combined_ranges = list(subtotal_ranges.values())
                if combined or combined_ranges:
                    paragraphs.append("With these materials: "+"; ".join([*("**"+format_fact(fact)+"**"+cite(fact) for fact in combined),*("**"+format_range(record)+"**" for record in combined_ranges)])+" — conditional priced subtotal.")
            if not subtotals and not subtotal_ranges:
                paragraphs.append("Separate historical alternatives are available; they cannot form one subtotal.")
            rows = []
            for fact in sorted(amounts,key=lambda item:amount_categories[item["id"]] != "service"):
                operands = [workspace.facts[reference] for reference in fact.get("parents",[])]
                rate = next(item for item in operands if str(item.get("unit","")).startswith("money/"))
                quantity = next(item for item in operands if item is not rate)
                label = rate.get("scope",{}).get("description") or rate["label"]
                if rate.get("scope",{}).get("category") == "labour":
                    label = _labour_label(quantity,workspace.facts)
                    if quantity.get("scope",{}).get("operation") == "scale_duration":
                        task = next((workspace.facts[reference] for reference in quantity.get("parents",[]) if workspace.facts[reference].get("kind") == "task_duration"),{})
                        labour_basis.append({"hours_ref":quantity["id"],"amount_ref":fact["id"],"rate_ref":rate["id"],
                                             "task_ref":task.get("id"),"task_label":task.get("label"),"operand_refs":list(quantity.get("parents",[])),
                                             "assumptions":list(quantity.get("assumptions",[]))})
                country = rate.get("scope",{}).get("country")
                if country: applied_countries.add(country)
                label += " ("+country+")" if country else ""
                rows.append("- "+label+": "+_quantity_text(quantity,format_fact)+" × "+format_fact(rate)+" = "+format_fact(fact)+cite(fact))
            if rows: paragraphs.append("\n".join(rows))
            for comparison in visit_comparisons:
                fact = lambda key: workspace.facts[comparison[key]]
                order = {label: index for index, label in enumerate(("Preparation and safety", "Inspections", "Testing", "Cleanup and documentation", "Other work"))}
                groups = sorted((group for group in comparison["groups"] if group["label"] != "Matched task"),
                                key=lambda group: order.get(group["label"], len(order)))
                group_text = "; ".join(group["label"].lower()+" "+format_fact(workspace.facts[group["hours_ref"]]) for group in groups)
                source = comparison.get("country") or "country unreported"
                matched = workspace.facts[comparison["matched_task_ref"]]
                bases = comparison["reported_bases"]
                agrees = bool(bases) and all(basis["matches_itemized"] for basis in bases)
                label = "Listed tasks (matches "+", ".join(basis["basis"] for basis in bases)+")" if agrees else "Listed tasks"
                rows = ["- "+label+": "+format_fact(fact("itemized_hours_ref"))+" × "+format_fact(fact("rate_ref"))+" = **"+format_fact(fact("itemized_amount_ref"))+"**."]
                selected_comparison = [fact("itemized_amount_ref"), fact("other_hours_ref"), matched, fact("rate_ref"),
                                       *(workspace.facts[group["hours_ref"]] for group in groups)]
                for basis in bases:
                    hours, cost = workspace.facts[basis["hours_ref"]], workspace.facts[basis["amount_ref"]]
                    if not agrees:
                        qualifier = "; differs from listed tasks" if not basis["matches_itemized"] else "; matches listed tasks"
                        rows.append("- "+basis["basis"].capitalize()+": "+format_fact(hours)+" = **"+format_fact(cost)+"**"+qualifier+".")
                    selected_comparison.extend([hours, cost])
                cite_many(selected_comparison)
                activities = "Matched task: "+format_fact(matched)+". Other listed work: "+format_fact(fact("other_hours_ref"))
                if group_text: activities += " — "+group_text
                paragraphs.append("Historical whole-visit labour — "+comparison["report_id"]+" ("+source+"):\n\n"+"\n".join(rows)
                                  +"\n\n"+activities+".\n\nOriginal, unscaled visit comparison; not added to this task budget.")
            scope, status = "partial_components", "ok"
        else: paragraphs.append("A priced subtotal is not supported yet.")

        legacy_roles = unpriced_roles - {"travel"} if travel_context is not None else unpriced_roles
        rate_cohorts = _hourly_cohorts(workspace,legacy_roles,known,set(by_currency))
        if rate_cohorts:
            rows = []
            for group in rate_cohorts[:3]:
                low,high = group["minimum"],group["maximum"]
                amount = f"{group['currency']} {low:,.2f}"+(f"–{high:,.2f}" if low != high else "")+"/h"
                citations = "".join(cite(workspace.facts[reference]) for reference in group["endpoint_refs"])
                mixed = "; mixed equipment" if len(group["equipment_families"])>1 else ""
                rows.append("- Historical "+group["category"]+" rate ("+(group["country"] or "location unreported")+mixed+"): **"+amount+"**"+citations)
            paragraphs.append("\n".join(rows))
            if not billed_amounts: scope,status = "rates_only","ok"
        if travel_requested and travel_context is not None:
            rows = []
            for group in travel_groups:
                low,high = group["minimum"],group["maximum"]
                rate_text = format_fact({"value":low,"unit":"money","currency":group["currency"]})
                if low != high:
                    high_text = format_fact({"value":high,"unit":"money","currency":group["currency"]})
                    rate_text += "–"+high_text.removeprefix(str(group["currency"])+" ")
                rate_text += "/h"
                citations = cite_many(workspace.facts[reference] for reference in group["fact_ids"])
                prefix = "- Historical travel rate ("+group["country"]+"): **"+rate_text+"**"+citations
                if group.get("amount_min_ref") and group.get("amount_max_ref"):
                    quantity = workspace.facts[group["quantity_ref"]]
                    record = range_record(workspace.facts[group["amount_min_ref"]],workspace.facts[group["amount_max_ref"]])
                    prefix += "; billed travel for "+_quantity_text(quantity,format_fact)+": **"+format_range(record)+"**"
                    applied_countries.add(group["country"])
                rows.append(prefix)
            if rows:
                paragraphs.append("\n".join(rows))
                details.append("Travel uses the observed range across these reports.")
                if not billed_amounts: scope,status = "rates_only","ok"
            elif travel_context.get("status") == "missing_country":
                details.append("Travel rates require the service country; no foreign-country travel rate was substituted.")
            else:
                details.append("No validated historical travel rate is available for "+str(travel_context.get("country") or "the requested country")+"; billed travel remains unpriced.")
            rate_cohorts.extend(travel_groups)
        if not rate_cohorts and not billed_amounts:
            observations = [fact for fact in workspace.facts.values() if fact.get("kind") == "unit_price" and "material" in categories]
            for fact in observations[:3]:
                qualifier = "pricing basis unconfirmed; do not extend this value" if not fact.get("scope",{}).get("validated") else "reported rate; requested quantity not priced"
                paragraphs.append("- "+(fact.get("scope",{}).get("description") or fact["label"])+": "+format_fact(fact)+" — "+qualifier+cite(fact))

        if billed_amounts:
            scope_notes.append("Partial visit budget; unlisted work, parts and charges are excluded. Historical rates and equipment compatibility need confirmation.")
        elif rate_cohorts:
            scope_notes.append("Historical hourly references, not a current quote or service total.")
        else: scope_notes.append("A matching unit-price or task-duration basis is still needed; no missing quantities or hours were invented.")
        if len({fact.get("currency") for fact in selected.values() if fact.get("currency")})>1:
            scope_notes.append("Currencies stay separate; no exchange rate is provided.")
        known_country = (known or {}).get("country")
        if known_country and (applied_countries-{known_country} or any(fact.get("scope",{}).get("country") not in {None,known_country} for fact in selected.values())):
            scope_notes.append("Foreign-country prices are examples, not confirmed local rates.")
        equipment_refs = {(plan or {}).get("equipment_count_ref")}
        equipment_refs.update(pair["equipment_count_ref"] for pair in quantity_relations(list(workspace.facts.values())))
        equipment_refs.update(fact.get("scope",{}).get("equipment_count_ref") for fact in workspace.facts.values() if fact.get("kind") == "request_quantity")
        if any(fact.get("scope",{}).get("customer_supplied") is True
               and fact.get("scope",{}).get("role") == "requested_quantity" and fact["id"] not in equipment_refs
               for fact in workspace.facts.values() if fact.get("kind") == "user_input"):
            details.append("Customer-supplied materials are excluded from purchases; their work can still require labour.")
        if any("single reported task" in assumption for assumption in assumptions):
            details.append("Labour uses a hypothetical single-task example; other visit work is excluded.")
        if labour_basis:
            details.append("Estimated labour hours scale a recorded task to the requested quantity; other visit work is excluded.")
        if any(workspace.facts[reference].get("scope",{}).get("duration_estimated") is True
               for amount in billed_amounts for reference in amount.get("parents",[])):
            details.append("Uses your nominal hour estimate; actual billable time may differ.")
        unpriced_modes = (work_scope or {}).get("unpriced_modes")
        if unpriced_modes:
            details.append("No supported duration for requested "+", ".join(unpriced_modes)+"; that work is excluded.")
        if any("does not reconcile" in assumption for assumption in assumptions):
            details.append("The source task totals do not reconcile; the isolated task estimate is qualified.")
        for role in sorted(unpriced_roles):
            record = next((record for record in records if record.role == role+"_hours"),None)
            known_hours = next((fact for fact in supplied if fact["scope"].get("role") == role+"_hours" and fact["id"] not in used),None)
            if known_hours: details.append(role.capitalize()+" ("+format_fact(known_hours)+" supplied) is not yet priced.")
            elif record and record.status == "unknown": details.append(role.capitalize()+" hours remain unknown.")
            elif not question: details.append(role.capitalize()+" is excluded until its hours are established.")
        outside_durations = []
        for record in records:
            role = record.role.removesuffix("_hours")
            if role in categories or record.status == "missing": continue
            if record.status == "known":
                outside_durations.append(role.capitalize()+": "+("about " if record.estimated else "")+format_fact({"value":record.total_hours,"unit":"hour"})+" total supplied")
            elif record.status == "unknown": outside_durations.append(role.capitalize()+" hours remain unknown")
            else:
                given = format_fact({"value":record.value,"unit":"hour"})+" supplied; " if record.value is not None else ""
                outside_durations.append(role.capitalize()+": "+given+"billing duration unresolved")
        if outside_durations: details.append("Outside this subtotal: "+"; ".join(outside_durations)+".")
        # Purchase coverage is separate from labour coverage. A count used to
        # scale work is neither an unpaid material bill nor proof of purchase.
        purchased, pending = set(), [fact["id"] for fact in amounts if billed_category(fact) == "material"]
        while pending:
            reference = pending.pop()
            if reference in purchased or reference not in workspace.facts: continue
            purchased.add(reference)
            pending.extend(workspace.facts[reference].get("parents", []))
        unpriced = [fact for fact in workspace.facts.values() if fact.get("kind") == "user_input"
                    and "material" in categories
                    and fact.get("scope",{}).get("role") == "requested_quantity" and fact.get("scope",{}).get("turn") == len(messages)-1
                    and fact.get("scope",{}).get("customer_supplied") is not True and fact["id"] not in purchased
                    and fact["id"] not in equipment_refs]
        if unpriced:
            details.append("Still unpriced: "+"; ".join(_quantity_text(fact,format_fact)+" "+(fact.get("scope",{}).get("entity_text") or "unspecified component")[:80] for fact in unpriced)+".")

    if original:
        if scope_notes: paragraphs.append(" ".join(dict.fromkeys(scope_notes)))
        if details: paragraphs.append(" ".join(dict.fromkeys(details)))
    else:
        foreign = any(note.startswith("Foreign-country") for note in scope_notes)
        if billed_amounts:
            note = ("Foreign-country historical prices are examples; local rates, equipment compatibility and unlisted visit charges remain unconfirmed."
                    if foreign else "Partial historical budget; unlisted work, parts and charges excluded, rates and equipment compatibility unconfirmed.")
        else:
            note = scope_notes[0] if scope_notes else ""
            if foreign: note += " Foreign-country rates require local confirmation."
        if any("exchange rate" in item for item in scope_notes): note += " Currencies stay separate; no exchange rate is provided."
        if note: paragraphs.append(note)
        compact_details = []
        if any("Customer-supplied" in item for item in details): compact_details.append("Customer-supplied materials excluded from purchases")
        if labour_basis:
            compared = {comparison["report_id"] for comparison in visit_comparisons}
            fully_compared = all(workspace.facts.get(basis.get("task_ref"), {}).get("report_id") in compared for basis in labour_basis)
            if not fully_compared:
                if any("do not reconcile" in item for item in details):
                    compact_details.append("Estimated hours cover one historical task; source task totals do not reconcile" if len(labour_basis)==1 else "Estimated hours use historical task examples; source task totals do not reconcile")
                else:
                    compact_details.append("Estimated hours scale one historical task; other visit work excluded" if len(labour_basis)==1 else "Estimated hours scale historical task examples; other visit work excluded")
        elif any("do not reconcile" in item for item in details): compact_details.append("isolated labour scenario: source task totals do not reconcile")
        elif any("hypothetical single-task" in item for item in details): compact_details.append("hypothetical single-task labour excludes other visit work")
        compact_details.extend(item.rstrip(".") for item in dict.fromkeys(details)
                               if not any(token in item for token in ("Customer-supplied","do not reconcile","hypothetical single-task","Estimated labour hours scale","Travel uses the observed range")))
        if compact_details: paragraphs.append("; ".join(compact_details)+".")
    if question: paragraphs.append(question)
    source_ids = list(dict.fromkeys(source for fact in selected.values() for source in fact.get("source_ids",[])))
    sources = [workspace.sources[source] for source in source_ids if source in workspace.sources]
    if sources: paragraphs.append("Sources:\n"+_source_listing(sources))
    if question and not sources and not billed_amounts: status = "clarify"
    question_record = next((record for record in records if duration_question([record],categories) == question),None) if question else None
    missing_fields = ["round_trip_travel_hours" if question_record.role == "travel_hours" else "labour_hours"] if question_record else []
    return {"tool":"service_cost","status":status,"data":{
                "summary":"\n\n".join(paragraphs),"response":{"status":"answered" if status=="ok" else status,"scope":scope},
                "selected_fact_ids":list(selected),"subtotal_refs_by_currency":{currency:fact["id"] for currency,fact in subtotals.items()},
                "category_subtotal_refs_by_currency":{category:{currency:fact["id"] for currency,fact in groups.items()} for category,groups in category_subtotals.items()},
                "subtotal_ranges_by_currency":subtotal_ranges,"category_subtotal_ranges_by_currency":category_ranges,
                "labour_estimation_basis":labour_basis,"travel_context":deepcopy(travel_context),
                "historical_visit_labour_comparisons":deepcopy(visit_comparisons),
                "execution_work_scope":deepcopy(work_scope),
                "assumptions":assumptions,"disclosures":list(dict.fromkeys(scope_notes+details)),
                "duration_inputs":[record.as_dict() for record in records],"historical_hourly_rate_cohorts":rate_cohorts,
                "unpriced_categories":sorted(unpriced_roles),"rendering":"deterministic-operation-results-v1"},
            "sources":sources,"warnings":[],"missing_fields":missing_fields,
            "question":question},derived
