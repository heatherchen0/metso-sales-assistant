"""Render model-selected CSV operations without generated statistical prose."""
from copy import deepcopy
import json

from .presentation import render_sources


_MIDDLE = {"below_q25": "below the lower quartile", "within_middle_50": "within the historical middle half",
           "above_q75": "above the upper quartile"}
_RELATION = {"below": "below", "equal": "equal to", "above": "above"}
_OPERATORS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "="}
_LIMIT = ("Historical position does not establish customer targets, an overall improvement, causes or operating setpoints. "
          "Confirm units and plant comparability.")


def _formatted(fact, *, count=False):
    # Lazy import keeps this standalone renderer usable from AdaptiveRunner.
    from .adaptive import format_fact
    return format_fact({**fact, "unit": None} if count else fact)


def _fact(workspace, identifier, column=None):
    fact = workspace.facts.get(identifier)
    if not fact or (column is not None and fact.get("column") != column):
        raise ValueError("A selected process result lacks its source-bound numerical evidence")
    return fact


def _input(workspace, identifier, column):
    return workspace._input(identifier, column)


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _selected(history):
    comparisons, distributions, counts, descriptions = {}, {}, {}, []
    for entry in history:
        result = entry.get("result", {})
        if result.get("error") or result.get("status") == "invalid_request":
            continue
        data = result.get("data", {})
        operation = entry.get("action", {}).get("action") or result.get("operation")
        if operation == "compare":
            for row in data.get("comparisons", []):
                comparisons[row["column"]] = row
        elif operation == "distribution":
            for row in data.get("distributions", []):
                distributions[row["column"]] = row
        elif operation == "count":
            key = json.dumps({"conditions": data.get("conditions"), "combine": data.get("combine")}, sort_keys=True)
            counts[key] = data
        elif operation == "describe":
            descriptions.append(data)
    return list(comparisons.values()), list(distributions.values()), list(counts.values()), descriptions


def _comparison_conclusion(workspace, rows):
    available = [row for row in rows if row.get("available")]
    outside = [row for row in available if row.get("range_status") in {"below_min", "above_max"}]
    if outside:
        labels = []
        for row in outside:
            reading = _formatted(_input(workspace, row["input_ref"], row["column"]))
            direction = "below its recorded minimum" if row["range_status"] == "below_min" else "above its recorded maximum"
            labels.append(f"{row['column']} ({reading}) is {direction}")
        return "Outside this plant's recorded experience: " + "; ".join(labels) + "."
    if available and all(row.get("middle_50_status") == "within_middle_50" for row in available):
        subject = "Each selected reading" if len(available) == len(rows) else "Each available selected reading"
        return ("The individual readings are broadly typical of this dataset: " + subject.lower() + " lies within its historical middle half. "
                "This does not establish that the combination is typical.")
    if available:
        subject = "Both readings" if len(available) == len(rows) == 2 else "The available readings"
        positions = []
        for row in available:
            status = row.get("middle_50_status")
            if status == "above_q75":
                positions.append(row["column"].lstrip("% ") + " is relatively high (above its upper quartile)")
            elif status == "below_q25":
                positions.append(row["column"].lstrip("% ") + " is relatively low (below its lower quartile)")
        return subject + " fall within this plant's historical ranges. " + "; ".join(positions) + "."
    return "The selected reading comparisons have no usable hourly observations."


def _comparison_rows(workspace, rows):
    text = []
    sample_counts = [_fact(workspace, row["facts"]["n"], row["column"]) for row in rows if row.get("available")]
    shared_count = bool(sample_counts) and len({fact["value"] for fact in sample_counts}) == 1
    if shared_count:
        qualifier = " for the available columns" if len(sample_counts) != len(rows) else ""
        text.append(f"From {_formatted(sample_counts[0], count=True)} hourly historical observations{qualifier}:\n")
    elif sample_counts:
        text.append("From the available hourly historical observations (sample sizes differ by column):\n")
    for row in rows:
        column = row["column"]
        if not row.get("available"):
            text.append(f"- {column}: no usable hourly observations for the selected reading comparison.")
            continue
        refs = row["facts"]
        selected = {name: _fact(workspace, refs[name], column) for name in ("n", "min", "q25", "median", "q75", "max")}
        reading = _formatted(_input(workspace, row["input_ref"], column))
        middle = _MIDDLE.get(row.get("middle_50_status"))
        median = _RELATION.get(row.get("median_relation"))
        if middle is None or median is None:
            raise ValueError("A selected process comparison lacks its computed ordering relationships")
        unit_note = " (units not stated in the CSV header)" if selected["median"].get("unit") is None else ""
        if row.get("range_status") in {"below_min", "above_max"}:
            evidence = f"recorded range {_formatted(selected['min'])}–{_formatted(selected['max'])}"
        else:
            evidence = f"{middle} (quartiles {_formatted(selected['q25'])}–{_formatted(selected['q75'])})"
        population = "" if shared_count else f"; {_formatted(selected['n'], count=True)} hourly observations"
        text.append(f"- {column}{unit_note}: supplied {reading}; {evidence}; "
                    f"{median} the median {_formatted(selected['median'])}{population}.")
    return "\n".join(text)


def _description_text(workspace, data):
    """Display requested diagnostics; their numerical values remain fact-bound."""
    parts = []
    sections = data.get("requested_sections", ["columns"])
    if "columns" in sections and data.get("columns"):
        parts.append("Available CSV columns and units:\n\n" + "\n".join(
            "- " + column["name"] + ": " + (column.get("unit") or "not stated") for column in data["columns"]))
    profile = data.get("profile", {})
    if "coverage" in sections:
        coverage = data.get("coverage", {})
        if coverage != workspace.coverage:
            raise ValueError("Displayed process coverage differs from the source snapshot")
        n = _fact(workspace, profile["unique_valid_hours"])
        if coverage.get("start") is None or coverage.get("end") is None:
            parts.append("Recorded period: unavailable; the CSV has no valid hourly timestamps.")
        else:
            timezone = "UTC" if coverage.get("timezone") == "aware" else "timezone not stated"
            parts.append(f"Recorded period: {coverage['start']} to {coverage['end']} ({timezone}); "
                         f"{_formatted(n, count=True)} distinct valid hourly observations.")
    if "gaps" in sections:
        missing = _fact(workspace, profile["missing_hours"])
        gap_count = _fact(workspace, profile["timestamp_gap_count"])
        if gap_count["value"]:
            gap_label = "recorded gap" if gap_count["value"] == 1 else "recorded gaps"
            parts.append(f"Timestamp gaps: {_formatted(missing, count=True)} missing hourly timestamps across "
                         f"{_formatted(gap_count, count=True)} {gap_label}; gaps are not filled.")
            details = []
            for gap in data.get("gaps", [])[:5]:
                hours = _fact(workspace, gap["missing_hours_ref"])
                details.append(f"- After {gap['after']} and before {gap['before']}: {_formatted(hours, count=True)} missing hours.")
            if details:
                label = "First recorded gaps:\n\n" if gap_count["value"] > len(details) else "Recorded gaps:\n\n"
                parts.append(label + "\n".join(details))
        elif _fact(workspace, profile["unique_valid_hours"])["value"]:
            parts.append("There are no missing hourly timestamps between the recorded first and last observations.")
        else:
            parts.append("Timestamp gaps cannot be assessed without recorded hourly timestamps.")
    if "quality" in sections:
        labels = {"malformed_rows": "malformed rows", "invalid_dates": "invalid-date rows",
                  "non_hourly_rows": "non-hourly rows", "duplicate_timestamp_rows": "duplicate timestamp rows"}
        diagnostics = [f"{_formatted(_fact(workspace, profile[key]), count=True)} {label}" for key, label in labels.items()]
        parts.append("Source diagnostics: " + "; ".join(diagnostics) + ". Duplicate hours count once; conflicting cells are excluded.")
        cells = []
        for column, issues in data.get("cell_issues", {}).items():
            for issue, ref in issues.items():
                fact = _fact(workspace, ref, column)
                if fact["value"]:
                    cells.append(f"- {column}: {_formatted(fact, count=True)} {issue.replace('_', ' ')} cells.")
        if cells:
            parts.append("Nonmissing populations vary where cells are unavailable:\n\n" + "\n".join(cells))
    return parts


def _distribution_table(workspace, rows):
    lines = ["| Column | Units | N | Minimum | Lower quartile | Median | Upper quartile | Maximum | Mean |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        column, refs = row["column"], row["facts"]
        n = _formatted(_fact(workspace, refs["n"], column), count=True)
        unit = workspace._unit(column) or "not stated"
        values = [_formatted(_fact(workspace, refs[name], column)) if row.get("available") else "unavailable"
                  for name in ("min", "q25", "median", "q75", "max", "mean")]
        lines.append("| " + " | ".join(map(_cell, [column, unit, n, *values])) + " |")
    return "\n".join(lines)


def _count_text(workspace, data):
    predicates = []
    for condition in data.get("conditions", []):
        column, op = condition["column"], condition["operator"]
        if op not in _OPERATORS:
            raise ValueError("A selected count lacks a supported predicate direction")
        value = _formatted(_input(workspace, condition["input_ref"], column))
        unit_note = " (units not stated)" if workspace._unit(column) is None else ""
        predicates.append(column + unit_note + " " + _OPERATORS[op] + " " + value)
    combine = data.get("combine")
    if combine not in {"and", "or"} or not predicates:
        raise ValueError("A selected count lacks its actual predicate combination")
    refs = data["facts"]
    count, population = _fact(workspace, refs["count"]), _fact(workspace, refs["population"])
    label = (" " + combine.upper() + " ").join(predicates)
    if population["value"] == 0:
        return "For " + label + ": no complete hourly observations are available; a percentage is unavailable."
    percentage = _fact(workspace, refs["percentage"])
    return ("For " + label + f": {_formatted(count, count=True)} of {_formatted(population, count=True)} complete hourly observations "
            f"({_formatted(percentage)}). Hours with a missing value in any selected predicate column are excluded.")


def process_answer(workspace, history, metadata, messages):
    """Render only the operations the model chose; infer no preferred direction.

    Question wording does not choose a workflow here. Available comparison,
    distribution and predicate result types determine their generic presentation.
    User values remain attached to their selected column and input reference.
    """
    warnings = list(workspace.warnings)
    for entry in history:
        warnings.extend(entry.get("result", {}).get("warnings", []))
    sources = deepcopy(workspace.sources)
    status, parts = "ok", []
    try:
        workspace._current()
        comparisons, distributions, counts, descriptions = _selected(history)
        if comparisons:
            parts.append(_comparison_conclusion(workspace, comparisons))
            parts.append(_comparison_rows(workspace, comparisons))
        elif distributions:
            parts.append("These distributions summarize the selected columns in this plant's historical hourly data; they do not define customer operating targets.")
        if distributions:
            parts.append(_distribution_table(workspace, distributions))
        for count in counts:
            parts.append(_count_text(workspace, count))
        described = _description_text(workspace, descriptions[-1]) if descriptions else []
        parts.extend(described)
        usable = (any(row.get("available") for row in [*comparisons, *distributions])
                  or any(_fact(workspace, count["facts"]["population"])["value"] > 0 for count in counts)
                  or bool(described))
        if not usable:
            status = "insufficient_evidence"
            if not parts:
                parts.append("The selected operations did not produce a usable comparison, distribution or count from the historical CSV.")
        if comparisons or distributions or counts:
            parts.append(_LIMIT)
        workspace._current()
    except (KeyError, TypeError, ValueError) as exc:
        status = "insufficient_evidence"
        parts = ["I could not render a source-validated process result from the selected operations."]
        warnings.append(str(exc))
        sources = []
    if sources:
        parts.append("Sources:\n" + render_sources(sources))
    return {"tool": "flotation", "status": status,
            "data": {"summary": "\n\n".join(parts), "response": {
                "status": "answered" if status == "ok" else "insufficient_evidence", "scope": "historical_assessment"},
                "engine": "model-selected-computed-process", "facts": list(workspace.facts.values())},
            "sources": sources, "warnings": list(dict.fromkeys(warnings)), "missing_fields": [], "question": None}
