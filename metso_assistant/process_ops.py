"""Small source-bound CSV operations selected by a model, calculated by Python.

There is no question classifier, preferred quality direction, or prose answer
template here. Thresholds come exclusively from verified user-input references.
The existing loader owns timestamp cleaning, missing cells and source hashing.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from decimal import Decimal
import hashlib
import json
import math
import operator
from pathlib import Path
import re
from statistics import fmean

from .process_data import _distribution, _load


METHOD_VERSION = "source-bound-csv-operations-1"
OPERATORS = {"gt": operator.gt, "ge": operator.ge, "lt": operator.lt,
             "le": operator.le, "eq": operator.eq}
_SYMBOLS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "="}


class _RequestError(ValueError):
    pass


def _canonical(value) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value).__name__)


def _finite(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, OverflowError):
        return False


class ProcessWorkspace:
    """One immutable source snapshot and an auditable numerical fact registry.

    A new workspace is needed after a source-file change. Only constructor
    inputs with ``kind=user_input`` can serve as comparison/count thresholds;
    models cannot manufacture a constant or promote an earlier computed fact.
    ``facts`` is exposed for answer rendering; calculations use private copies
    of input facts so public-registry mutation cannot change a user threshold.
    """

    def __init__(self, csv_path, inputs=None):
        self.path = Path(csv_path)
        self.facts: dict[str, dict] = {}
        self._inputs: dict[str, dict] = {}
        if inputs is not None and not isinstance(inputs, list):
            raise ValueError("Verified process inputs must be a list of facts")
        for supplied in inputs or []:
            if (not isinstance(supplied, dict) or supplied.get("kind") != "user_input"
                    or not isinstance(supplied.get("id"), str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", supplied["id"])
                    or not isinstance(supplied.get("label"), str) or not supplied["label"].strip()
                    or not _finite(supplied.get("value"))
                    or (supplied.get("unit") is not None and not isinstance(supplied["unit"], str))):
                raise ValueError("Process thresholds require identified, finite, verified user-input facts")
            if supplied["id"] in self._inputs:
                raise ValueError("Duplicate verified process-input identifier")
            fact = deepcopy(supplied)
            fact.setdefault("currency", None)
            fact.setdefault("parents", [])
            fact.setdefault("source_ids", [])
            fact.setdefault("assumptions", [])
            self._inputs[fact["id"]] = fact
            self.facts[fact["id"]] = deepcopy(fact)
        self._error = None
        self.records, self.profile, self.columns, self.sources = [], {}, [], []
        try:
            records, source, profile, _ = _load(self.path)
            self.records, self.profile = records, profile
            self.columns = list(profile["cell_issues"])
            self.source = {key: source[key] for key in ("id", "file", "sha256", "kind")}
            self.sources = [self.source]
            self.coverage = {"start": source.get("date_start"), "end": source.get("date_end"),
                             "timezone": profile.get("timestamp_timezone")}
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            self._error = str(exc)
        self.warnings = [
            "Hourly observations are serially dependent; historical frequencies are descriptive rather than independent-trial probabilities.",
            "Plant, measurement and operating-condition comparability with the customer is unconfirmed.",
            "Historical comparisons do not establish causes, operating targets, recovery, throughput or recommended setpoints.",
            "Non-percent numeric columns have no inferred physical units; inspect their original CSV headers.",
        ]
        if self.profile.get("missing_hours"):
            self.warnings.append("Missing hourly timestamps are not filled; inspect diagnostic facts.")
        if self.profile.get("adjacent_repeated_quality_pairs"):
            self.warnings.append("Adjacent hours share concentrate readings; the source does not establish why.")
        if self.profile.get("duplicate_timestamp_rows"):
            self.warnings.append("Duplicate hours count once; conflicting cells become unavailable rather than being averaged.")
        if any(self.profile.get(key) for key in ("malformed_rows", "invalid_dates", "non_hourly_rows")):
            self.warnings.append("Malformed, invalid-date or non-hourly rows are excluded; diagnostic facts retain their counts.")
        if any(any(values.values()) for values in self.profile.get("cell_issues", {}).values()):
            self.warnings.append("Missing, invalid or conflicting cells are excluded per operation; numeric values are not imputed.")

    def _current(self):
        if self._error:
            raise _RequestError("Historical process CSV is unavailable: " + self._error)
        try:
            current = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise _RequestError("Historical process CSV is unavailable: " + str(exc)) from exc
        if current != self.source["sha256"]:
            raise _RequestError("Historical process CSV changed; create a new workspace before calculating from its current bytes")

    def _fact(self, operation: str, arguments, metric: str, label: str, value,
              unit=None, parents=(), assumptions=(), **metadata):
        if not _finite(value):
            raise _RequestError("A requested statistic is unavailable or non-finite")
        identity = {"source_sha256": self.source["sha256"], "method": METHOD_VERSION,
                    "operation": operation, "arguments": arguments, "metric": metric}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, default=_canonical).encode("utf-8")).hexdigest()
        identifier = "PF" + digest[:20]
        fact = {"id": identifier, "label": label, "value": value, "unit": unit,
                "currency": None, "kind": "computed", "source_ids": ["P01"],
                "parents": list(parents), "assumptions": list(assumptions),
                "operation": operation, "metric": metric, **metadata}
        if identifier in self.facts and self.facts[identifier] != fact:
            raise _RequestError("Computed fact identity conflicts with an existing registry entry")
        self.facts[identifier] = deepcopy(fact)
        return fact

    def _envelope(self, operation, facts=(), data=None, status="ok", warnings=()):
        return {"status": status, "operation": operation, "facts": deepcopy(list(facts)),
                "sources": deepcopy(self.sources), "warnings": [*self.warnings, *warnings],
                "data": deepcopy(data or {})}

    def _failure(self, operation, exc, previous_ids=()):
        # A failed operation must not leave partially computed authority in the
        # registry, especially if its source changes before the final check.
        previous_ids = set(previous_ids)
        for identifier in set(self.facts) - previous_ids:
            del self.facts[identifier]
        return self._envelope(operation, status="insufficient_evidence" if self._error or str(exc).startswith("Historical process CSV")
                              else "invalid_request", warnings=[str(exc)],
                              data={"available_columns": list(self.columns)})

    def _column(self, value):
        if not isinstance(value, str) or value not in self.columns:
            raise _RequestError("Unavailable numeric CSV column: " + str(value))
        return value

    @staticmethod
    def _unit(column):
        return "percent" if column.startswith("%") else None

    def _input(self, identifier, column):
        if not isinstance(identifier, str) or identifier not in self._inputs:
            raise _RequestError("Threshold must reference a verified user-input fact")
        fact = self._inputs[identifier]
        unit = (fact.get("unit") or "").strip().casefold()
        if self._unit(column) == "percent":
            if unit not in {"%", "percent", "percentage", "pct"}:
                raise _RequestError("A percent CSV column requires an explicit percent input unit")
        elif unit not in {"", "number", "unknown", "unspecified"}:
            raise _RequestError("The CSV does not establish a compatible unit for this non-percent column")
        if fact.get("currency") is not None:
            raise _RequestError("Currency-valued inputs cannot be process thresholds")
        return fact

    def describe(self):
        operation = "describe"
        previous_ids = set(self.facts)
        try:
            self._current()
            facts, profile = [], {}
            for metric, value, unit in (("column_count", len(self.columns) + 1, "column"),
                                        ("numeric_column_count", len(self.columns), "column"),
                                        ("timestamp_gap_count", len(self.profile.get("gaps", [])), "count")):
                fact = self._fact(operation, {}, metric, "CSV diagnostic: " + metric.replace("_", " "), value, unit)
                facts.append(fact)
                profile[metric] = fact["id"]
            for key, value in self.profile.items():
                if isinstance(value, int):
                    fact = self._fact(operation, {}, key, "CSV diagnostic: " + key.replace("_", " "), value,
                                      "hour" if key.endswith("hours") else "count")
                    facts.append(fact)
                    profile[key] = fact["id"]
            cells = {}
            for column, issues in self.profile.get("cell_issues", {}).items():
                cells[column] = {}
                for issue, value in issues.items():
                    fact = self._fact(operation, {"column": column}, issue,
                                      column + ": " + issue.replace("_", " ") + " cells", value, "count", column=column)
                    facts.append(fact)
                    cells[column][issue] = fact["id"]
            gaps = []
            for gap in self.profile.get("gaps", []):
                args = {"after": gap["after"], "before": gap["before"]}
                fact = self._fact(operation, args, "missing_hours_in_gap", "Missing hours in recorded timestamp gap",
                                  gap["missing_hours"], "hour")
                facts.append(fact)
                gaps.append({**args, "missing_hours_ref": fact["id"]})
            self._current()
            return self._envelope(operation, facts, {"columns": [{"name": "date", "type": "datetime", "unit": None},
                *[{"name": column, "type": "number", "unit": self._unit(column)} for column in self.columns]],
                "coverage": self.coverage, "profile": profile, "cell_issues": cells, "gaps": gaps,
                "population": "One observation per valid, distinct hourly timestamp; each operation excludes its required missing or conflicting cells."})
        except _RequestError as exc:
            return self._failure(operation, exc, previous_ids)

    def _distribution(self, column):
        values = [record[column] for record in self.records if record[column] is not None]
        stats = _distribution(values)
        if values:
            stats["mean"] = fmean(values)
        facts, refs = [], {}
        for metric, value in stats.items():
            if value is None:
                continue
            unit = "hour" if metric == "n" else self._unit(column)
            label = column + ": " + ("observed nonmissing hourly population" if metric == "n" else metric)
            fact = self._fact("distribution", {"column": column}, metric, label, value, unit, column=column,
                              population="Nonmissing distinct hourly observations for this column")
            facts.append(fact)
            refs[metric] = fact["id"]
        return values, facts, refs

    def distribution(self, columns):
        operation = "distribution"
        previous_ids = set(self.facts)
        try:
            self._current()
            if not isinstance(columns, list) or not columns or len(columns) > 64:
                raise _RequestError("Distribution requires a bounded nonempty list of actual numeric CSV columns")
            selected = list(dict.fromkeys(self._column(column) for column in columns))
            facts, result = [], []
            for column in selected:
                values, computed, refs = self._distribution(column)
                facts.extend(computed)
                result.append({"column": column, "available": bool(values), "facts": refs})
            self._current()
            return self._envelope(operation, facts, {"distributions": result},
                status="ok" if any(row["available"] for row in result) else "insufficient_evidence",
                warnings=[] if all(row["available"] for row in result) else ["A selected column has no usable hourly observations."])
        except (TypeError, _RequestError, OverflowError) as exc:
            return self._failure(operation, exc, previous_ids)

    def compare(self, readings):
        operation = "compare"
        previous_ids = set(self.facts)
        try:
            self._current()
            if not isinstance(readings, list) or not readings or len(readings) > 64:
                raise _RequestError("Compare requires a bounded nonempty list of column/input references")
            validated = []
            for reading in readings:
                if not isinstance(reading, dict) or set(reading) != {"column", "input_ref"}:
                    raise _RequestError("Each reading requires only column and input_ref; model-supplied constants are forbidden")
                column = self._column(reading["column"])
                validated.append((column, self._input(reading["input_ref"], column)))
            facts, comparisons = {}, []
            for column, input_fact in validated:
                values, computed, refs = self._distribution(column)
                facts.update((fact["id"], fact) for fact in computed)
                result = {"column": column, "input_ref": input_fact["id"], "available": bool(values), "facts": refs}
                if not values:
                    result["range_status"] = "unavailable"
                    result["middle_50_status"] = "unavailable"
                    result["median_relation"] = "unavailable"
                    result["mean_relation"] = "unavailable"
                    comparisons.append(result)
                    continue
                value = float(input_fact["value"])
                below, equal, above = (sum(test(item, value) for item in values)
                                       for test in (operator.lt, operator.eq, operator.gt))
                lower, upper = min(values), max(values)
                result["range_status"] = "below_min" if value < lower else "above_max" if value > upper else "within_observed_range"
                statistics = {fact["metric"]: fact["value"] for fact in computed}
                result["middle_50_status"] = ("below_q25" if value < statistics["q25"]
                                              else "above_q75" if value > statistics["q75"]
                                              else "within_middle_50")
                for name in ("median", "mean"):
                    boundary = statistics[name]
                    result[name + "_relation"] = "below" if value < boundary else "above" if value > boundary else "equal"
                metrics = {"below_count": below, "equal_count": equal, "above_count": above,
                           "empirical_percentile": 100 * (below + .5 * equal) / len(values),
                           "distance_below_min": max(0.0, lower - value),
                           "distance_above_max": max(0.0, value - upper)}
                args = {"column": column, "input": input_fact}
                for metric, amount in metrics.items():
                    unit = ("hour" if metric.endswith("count") else "percentile" if metric == "empirical_percentile"
                            else "percentage_point" if self._unit(column) == "percent" else None)
                    fact = self._fact(operation, args, metric, column + ": " + metric.replace("_", " "),
                                      amount, unit, parents=[input_fact["id"], refs["n"], refs["min"], refs["max"]], column=column)
                    facts[fact["id"]] = fact
                    result["facts"][metric] = fact["id"]
                comparisons.append(result)
            self._current()
            return self._envelope(operation, facts.values(), {"comparisons": comparisons,
                "percentile_definition": "Midrank empirical percentile: below count plus half the equal count, divided by the nonmissing hourly population."},
                status="ok" if any(row["available"] for row in comparisons) else "insufficient_evidence")
        except (TypeError, _RequestError, OverflowError) as exc:
            return self._failure(operation, exc, previous_ids)

    def count(self, conditions, combine="and"):
        operation = "count"
        previous_ids = set(self.facts)
        try:
            self._current()
            if combine not in {"and", "or"}:
                raise _RequestError("Combine must be and or or")
            if not isinstance(conditions, list) or not conditions or len(conditions) > 64:
                raise _RequestError("Count requires a bounded nonempty list of predicates")
            selected, predicate_info = [], []
            for condition in conditions:
                if not isinstance(condition, dict) or set(condition) != {"column", "operator", "input_ref"}:
                    raise _RequestError("Each predicate requires only column, operator and input_ref; model-supplied constants are forbidden")
                column = self._column(condition["column"])
                operation_name = condition["operator"]
                if operation_name not in OPERATORS:
                    raise _RequestError("Unavailable comparison operator")
                input_fact = self._input(condition["input_ref"], column)
                selected.append((column, operation_name, input_fact))
                predicate_info.append({"column": column, "operator": operation_name,
                                       "symbol": _SYMBOLS[operation_name], "input_ref": input_fact["id"]})
            columns = {column for column, _, _ in selected}
            population = [row for row in self.records if all(row[column] is not None for column in columns)]
            aggregator = all if combine == "and" else any
            matching = sum(aggregator(OPERATORS[op](row[column], float(fact["value"]))
                                     for column, op, fact in selected) for row in population)
            args = {"conditions": [{"column": column, "operator": op, "input": fact}
                                    for column, op, fact in selected], "combine": combine}
            parents = list(dict.fromkeys(fact["id"] for _, _, fact in selected))
            n_fact = self._fact(operation, args, "population", "Complete hourly population for the selected predicates",
                                len(population), "hour", parents=parents)
            count_fact = self._fact(operation, args, "count", "Hours satisfying the selected " + combine.upper() + " predicates",
                                    matching, "hour", parents=[*parents, n_fact["id"]])
            facts = [n_fact, count_fact]
            refs = {"population": n_fact["id"], "count": count_fact["id"]}
            if population:
                percentage = self._fact(operation, args, "percentage", "Percentage of complete hours satisfying the selected " + combine.upper() + " predicates",
                                        100 * matching / len(population), "percent", parents=[count_fact["id"], n_fact["id"]])
                facts.append(percentage)
                refs["percentage"] = percentage["id"]
            self._current()
            return self._envelope(operation, facts, {"conditions": predicate_info, "combine": combine, "facts": refs,
                "population": "Distinct hourly timestamps with nonmissing values for every selected predicate column, including for OR combinations.",
                "interpretation": "Predicate directions are supplied by the plan; frequencies do not establish whether a reading is desirable or a target is appropriate."},
                status="ok" if population else "insufficient_evidence",
                warnings=[] if population else ["No complete hourly observations support this predicate population."])
        except (TypeError, _RequestError, OverflowError) as exc:
            return self._failure(operation, exc, previous_ids)
