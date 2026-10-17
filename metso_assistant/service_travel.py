"""Country-based historical travel references and checked alternative bounds.

Rates describe an observed cohort, not a national tariff or a prediction
interval. The minimum and maximum are alternatives for the same billed hours;
they must never be added as two charges. Currency conversion is unavailable.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from decimal import Decimal


def travel_context(workspace, country, duration=None):
    """Retrieve all eligible local rates, and price bounds for confirmed hours."""
    context = {"status": "missing_country", "country": country,
               "cohorts": [], "endpoint_fact_ids": [], "errors": []}
    traces = []
    if not isinstance(country, str) or not country.strip():
        context["country"] = None
        return context, traces
    country = country.strip()
    result = workspace.travel_rates(country)
    traces.append({"operation": "adaptive_tool", "action": {"action": "travel_rates", "country": country},
                   "result": deepcopy(result)})
    context.update(status="unavailable", country=country)
    if result.get("error"):
        context["errors"].append(result["error"])
        return context, traces
    grouped = defaultdict(list)
    for raw in result.get("facts", []):
        fact = workspace._ref(raw["id"])
        scope = fact.get("scope", {})
        if (fact.get("kind") != "hourly_rate" or fact.get("unit") != "money/hour"
                or scope.get("category") != "travel" or scope.get("validated") is not True
                or str(scope.get("country") or "").casefold() != country.casefold()
                or not isinstance(fact.get("value"), Decimal) or not fact["value"].is_finite() or fact["value"] < 0):
            raise ValueError("Country travel references must be validated, source-bound local hourly rates")
        grouped[fact["currency"]].append(fact)
    if not grouped:
        return context, traces
    context["status"] = "available"
    quantity = None
    if duration is not None and duration.status == "known":
        candidates = [workspace._ref(fact["id"]) for fact in workspace.facts.values()
                      if fact.get("kind") == "user_input" and fact.get("unit") == "hour"
                      and fact.get("scope", {}).get("role") == "travel_hours"
                      and fact.get("value") == duration.total_hours]
        if len(candidates) == 1:
            quantity = candidates[0]
        else:
            context["errors"].append("Confirmed travel hours lack a unique matching user-input reference")
    for currency, rates in sorted(grouped.items()):
        rates.sort(key=lambda fact: (fact["value"], fact["report_id"], fact["id"]))
        minimum, maximum = rates[0], rates[-1]
        # Equal rates yield one amount, even if several reports corroborate it.
        if minimum["value"] == maximum["value"]:
            maximum = minimum
        endpoint_refs = list(dict.fromkeys([minimum["id"], maximum["id"]]))
        cohort = {"category": "travel", "country": rates[0]["scope"]["country"], "currency": currency,
                  "minimum": minimum["value"], "maximum": maximum["value"],
                  "fact_ids": [fact["id"] for fact in rates], "endpoint_refs": endpoint_refs,
                  "reports": sorted({fact["report_id"] for fact in rates}),
                  "equipment_families": sorted({fact["scope"].get("equipment_family") or "unclassified" for fact in rates}),
                  "range_kind": "observed_min_max", "quantity_ref": quantity["id"] if quantity else None,
                  "amount_min_ref": None, "amount_max_ref": None}
        if quantity:
            amounts = {}
            for reference in endpoint_refs:
                refs = [reference, quantity["id"]]
                calculated = workspace.calculate("multiply", refs)
                traces.append({"operation": "adaptive_tool", "action": {"action": "calculate", "operation": "multiply", "refs": refs},
                               "travel_range_endpoint": True, "result": deepcopy(calculated)})
                if calculated.get("error") or not calculated.get("facts"):
                    context["errors"].append(calculated.get("error", "Travel endpoint could not be priced"))
                else:
                    amount = calculated["facts"][-1]
                    amounts[reference] = amount["id"]
                    context["endpoint_fact_ids"].append(amount["id"])
            # A partially computed interval cannot be presented as a point cost.
            if len(amounts) == len(endpoint_refs):
                cohort["amount_min_ref"] = amounts[minimum["id"]]
                cohort["amount_max_ref"] = amounts[maximum["id"]]
        context["cohorts"].append(cohort)
    traces.append({"operation": "country_travel_context", "status": context["status"], "country": country,
                   "rate_ids": [reference for cohort in context["cohorts"] for reference in cohort["fact_ids"]],
                   "endpoint_fact_ids": context["endpoint_fact_ids"], "range_kind": "observed_min_max"})
    return context, traces
