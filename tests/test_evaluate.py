"""Evaluator contracts distinguish actual task/result coverage from registry hits."""
import csv
from copy import deepcopy
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from metso_assistant.adaptive import request_facts
from metso_assistant.evaluate import evaluate_case
from metso_assistant.process_ops import ProcessWorkspace
from metso_assistant.process_request import execute_process_request, interpret_process_request


IRON, SILICA, FLOW = "% Iron Concentrate", "% Silica Concentrate", "Invented Signal Flow"


class Assistant:
    def __init__(self, turns):
        self.turns = turns

    def reset(self):
        self.index = 0

    def turn(self, message):
        result = deepcopy(self.turns[self.index])
        self.index += 1
        return result


class Model:
    def __init__(self, output=None):
        self.output = output

    def invoke(self, system, messages, schema, **kwargs):
        import json
        payload = json.loads(messages[0]["content"])
        return (self.output(payload) if self.output else {
            "status": "ready", "question": None, "compare": payload["binding_candidates"],
            "distribution": [], "count": [], "describe": False}), {}


def case(messages, route):
    return {"id": "synthetic", "messages": messages, "expected_routes": [route] * len(messages),
            "allowed_statuses": [["answered"] for _ in messages]}


def product_turn(message, kinds=("describe", "alternatives")):
    sources = [{"id": "Balpha", "file": "invented-alpha.pdf", "page": 1, "doc_id": "alpha-controller"},
               {"id": "Bbeta", "file": "invented-beta.pdf", "page": 2, "doc_id": "beta-cell"}]
    products = [{"doc_id": "alpha-controller", "name": "alpha controller", "aliases": ["AC-7"], "source_file": "invented-alpha.pdf"},
                {"doc_id": "beta-cell", "name": "beta cell", "aliases": ["BC-2"], "source_file": "invented-beta.pdf"}]
    tasks, results, coverage, trace, text = [], [], [], [], []
    for index, kind in enumerate(kinds):
        subject = "alpha controller" if kind in {"describe", "explain", "specification", "visual"} else None
        task = {"kind": kind, "subject_product": subject, "question_quote": message}
        selected_sources = sources if kind == "compare" else [sources[0 if subject else 1]]
        answer = ("The alpha controller controls levels, while the beta cell handles fine particles." if kind == "compare" else
                  "The alpha controller adjusts its control outputs." if subject else "A separate candidate is the beta cell.")
        tasks.append(task)
        text.append(answer)
        results.append({"task": task, "executed": True, "status": "answered", "answer": answer,
                        "source_ids": [source["id"] for source in selected_sources], "tool_result": {"sources": selected_sources, "data": {
                            "products": deepcopy(products), "passages": [{"doc_id": source["doc_id"], "source_id": source["id"]} for source in selected_sources]}}})
        coverage.append({"task_index": index, "kind": kind, "subject_product": subject,
                         "status": "answered", "source_ids": [source["id"] for source in selected_sources]})
        trace.append({"operation": "grounded_product_generation", "request_task_index": index,
                      "attempt": 1, "validation_errors": [], "raw": {"status": "answered",
                      "recommendation": "beta cell" if kind in {"recommend", "alternatives"} else None}})
    plan = {"context_status": "continue_need", "need_quote": None, "equipment_quote": None, "tasks": tasks}
    trace.append({"operation": "product_request_coverage", "requested_tasks": len(tasks),
                  "completed_tasks": len(tasks), "all_requested_tasks_addressed": True,
                  "all_requested_tasks_completed": True, "coverage": coverage})
    return {"route": "product", "status": "answered", "answer": "\n\n".join(text), "sources": sources,
            "trace": trace, "product_request_plan": plan, "tool_result": {"data": {"request_plan": plan,
            "task_results": results}, "sources": sources}}


def travel_turn():
    sources = [{"id": "SE1", "file": "invented-one.txt"}, {"id": "SE2", "file": "invented-two.txt"}]
    rates = [{"id": identifier, "kind": "hourly_rate", "unit": "money/hour", "currency": "USD", "value": Decimal(value),
              "source_ids": [source], "scope": {"category": "travel", "country": "Chile", "validated": True}}
             for identifier, value, source in (("R1", "11", "SE1"), ("R2", "23", "SE2"))]
    quantity = {"id": "D8", "kind": "user_input", "value": Decimal(8), "unit": "hour", "scope": {"role": "travel_hours"}}
    amounts = [{"id": identifier, "kind": "calculation", "unit": "money", "currency": "USD", "value": Decimal(value),
                "parents": [rate, "D8"], "scope": {"operation": "multiply", "coverage": ["billed:travel_hours:D8"]}}
               for identifier, rate, value in (("A1", "R1", "88.00"), ("A2", "R2", "184.00"))]
    base = {"id": "B", "kind": "calculation", "unit": "money", "currency": "USD", "value": Decimal("290.00"),
            "parents": [], "scope": {"operation": "multiply", "coverage": ["material:U1:invented"]}}
    totals = [{"id": identifier, "kind": "calculation", "unit": "money", "currency": "USD", "value": Decimal(value),
               "parents": ["B", amount], "scope": {"operation": "sum"}}
              for identifier, amount, value in (("T1", "A1", "378.00"), ("T2", "A2", "474.00"))]
    cohort = {"country": "Chile", "currency": "USD", "minimum": Decimal(11), "maximum": Decimal(23),
              "fact_ids": ["R1", "R2"], "endpoint_refs": ["R1", "R2"], "quantity_ref": "D8",
              "amount_min_ref": "A1", "amount_max_ref": "A2", "range_kind": "observed_min_max"}
    data = {"travel_context": {"status": "available", "country": "Chile", "cohorts": [cohort], "endpoint_fact_ids": ["A1", "A2"]},
            "duration_inputs": [{"role": "travel_hours", "status": "known", "total_hours": Decimal(8)}],
            "facts": [*rates, quantity, *amounts, base, *totals], "unpriced_categories": [],
            "subtotal_ranges_by_currency": {"USD": {"minimum_ref": "T1", "maximum_ref": "T2", "minimum": Decimal("378.00"), "maximum": Decimal("474.00")}},
            "category_subtotal_ranges_by_currency": {}, "response": {"scope": "partial_components"}}
    return {"route": "service_cost", "status": "answered", "engine": "adaptive", "tool_arguments": {"country": "Chile"},
            "answer": "Historical travel rates (Chile): USD 11.00-23.00/h. The travel endpoints are alternatives, not two charges.",
            "sources": sources, "tool_result": {"data": data}, "trace": [
                {"operation": "adaptive_tool", "action": {"action": "travel_rates", "country": "Chile"}, "result": {"facts": rates, "sources": sources}},
                {"operation": "deterministic_evidence_rendering", "numerical_generation": False}]}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.csv"
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", IRON, SILICA, FLOW])
            for index, values in enumerate([(50, 6, 100), (55, 5, 200), (60, 4, 300), (65, 3, 400)]):
                writer.writerow([(datetime(2022, 1, 1) + timedelta(hours=index)).isoformat(sep=" "), *values])

    def process_turn(self, messages, output=None):
        inputs = request_facts(messages, "flotation")
        workspace = ProcessWorkspace(self.path, inputs)
        metadata = workspace.describe()
        columns = [column for column in metadata["data"]["columns"] if column["type"] != "datetime"]
        plan, trace = interpret_process_request(Model(output), messages, inputs, columns)
        result, execution = execute_process_request(workspace, plan, metadata, messages)
        return {"route": "flotation", "status": "answered" if result["status"] == "ok" else result["status"],
                "answer": result["data"]["summary"], "sources": result["sources"], "tool_result": result,
                "trace": [*trace, *execution], "engine": "adaptive"}

    def evaluate(self, messages, turns, route="flotation", **requirements):
        return evaluate_case({**case(messages, route), **requirements}, Assistant(turns))

    def test_task_only_alternatives_cannot_replace_actual_prior_need_with_the_task(self):
        messages = ["Our alpha controller has unstable levels.", "Do you have other recommendations?"]
        first = product_turn(messages[0], ("describe",))
        first["conversation_context"] = {"product": {"subject_query": messages[0]}}
        second = product_turn(messages[1], ("alternatives",))
        second["conversation_context"] = {"product": {"subject_query": messages[0]}}
        result = self.evaluate(messages, [first, second], "product")
        self.assertTrue(result["checks"]["product_alternative_need_retained_turn_1"])
        second["conversation_context"]["product"]["subject_query"] = messages[1]
        second["product_request_plan"]["context_status"] = "new_need"
        result = self.evaluate(messages, [first, second], "product")
        self.assertFalse(result["checks"]["product_alternative_need_retained_turn_1"])

    def test_alternative_prior_need_must_have_actual_user_provenance(self):
        messages = ["Our alpha controller has unstable levels.", "Any additional options?"]
        first = product_turn(messages[0], ("describe",))
        first["conversation_context"] = {"product": {"subject_query": "Invented customer symptom"}}
        second = product_turn(messages[1], ("alternatives",))
        second["conversation_context"] = deepcopy(first["conversation_context"])
        self.assertFalse(self.evaluate(messages, [first, second], "product")["checks"]["product_alternative_need_retained_turn_1"])

    @staticmethod
    def service_scope_turn(messages, *, keep_history=True):
        return {"route": "service_cost", "status": "answered", "answer": "Historical components remain partly unpriced.",
                "sources": [{"id": "SEsynthetic", "file": "invented.txt"}], "trace": [],
                "tool_result": {"data": {"facts": request_facts(messages if keep_history else messages[-1:], "service_cost")}}}

    def test_additive_request_cannot_drop_prior_quantity_even_when_model_declares_new_boundary(self):
        messages = ["Replace 4 sleeves.", "Also add 2 sealing rings. Give the subtotal now."]
        first = self.service_scope_turn(messages[:1])
        second = self.service_scope_turn(messages, keep_history=False)
        second["trace"] = [{"operation": "current_request_boundary", "new_request": True}]
        result = self.evaluate(messages, [first, second], "service_cost")
        self.assertFalse(result["checks"]["service_additive_requested_quantities_retained_turn_1"])
        second["tool_result"]["data"]["facts"] = request_facts(messages, "service_cost")
        result = self.evaluate(messages, [first, second], "service_cost")
        self.assertTrue(result["checks"]["service_additive_requested_quantities_retained_turn_1"])

    def test_additive_scope_check_requires_retention_without_requiring_price_availability(self):
        messages = ["We need 3 protective covers.", "Also include 6 gaskets."]
        turns = [self.service_scope_turn(messages[:i+1]) for i in range(2)]
        result = self.evaluate(messages, turns, "service_cost")
        self.assertTrue(result["checks"]["service_additive_requested_quantities_retained_turn_1"])
        self.assertFalse(any(fact.get("currency") for fact in turns[1]["tool_result"]["data"]["facts"]))

    def test_additive_scope_respects_explicitly_superseded_counts(self):
        messages = ["We need 4 sleeves.", "Actually make that 7 sleeves, not 4.", "Also add 2 sealing rings."]
        turns = [self.service_scope_turn(messages[:i+1]) for i in range(3)]
        facts = turns[-1]["tool_result"]["data"]["facts"]
        self.assertTrue(any(fact.get("scope", {}).get("role") == "superseded_input" for fact in facts))
        turns[-1]["tool_result"]["data"]["facts"] = [fact for fact in facts if fact.get("scope", {}).get("role") != "superseded_input"]
        result = self.evaluate(messages, turns, "service_cost")
        self.assertTrue(result["checks"]["service_additive_requested_quantities_retained_turn_2"])

    def test_explicit_different_customer_is_not_an_addition_to_previous_job(self):
        messages = ["We need 4 sleeves.", "A different customer also needs 2 gaskets."]
        turns = [self.service_scope_turn(messages[:1]), self.service_scope_turn(messages, keep_history=False)]
        result = self.evaluate(messages, turns, "service_cost")
        self.assertNotIn("service_additive_requested_quantities_retained_turn_1", result["checks"])

    def test_complete_current_measurement_replacement_has_nonvacuous_checks(self):
        messages = ["Concentrate has 3.1% silica and 64.5% iron. Are these typical?",
                    "Concentrate now has 2.2% silica and 65% iron. Are these typical?"]
        turns = [self.process_turn(messages[:index+1]) for index in range(2)]
        result = self.evaluate(messages, turns)
        self.assertTrue(result["contract_checks_passed"], result["checks"])
        self.assertTrue(result["checks"]["process_actual_input_bindings_turn_1"])
        self.assertTrue(result["checks"]["process_exact_selected_results_turn_1"])
        self.assertTrue(result["checks"]["process_current_values_displayed_turn_1"])

    def test_refs_anywhere_do_not_hide_swapped_final_selected_columns(self):
        messages = ["Concentrate has 2.2% silica and 65% iron. Are these typical?"]
        turn = self.process_turn(messages)
        trace = next(entry for entry in turn["trace"] if entry.get("action", {}).get("action") == "compare")
        rows = trace["result"]["data"]["comparisons"]
        rows[0]["input_ref"], rows[1]["input_ref"] = rows[1]["input_ref"], rows[0]["input_ref"]
        result = self.evaluate(messages, [turn])
        self.assertFalse(result["checks"]["process_exact_selected_results_turn_0"])
        self.assertFalse(result["contract_checks_passed"])

    def test_correct_registry_and_plan_do_not_hide_wrong_visible_column_value(self):
        messages = ["Concentrate iron is 60%. Is that typical?"]
        turn = self.process_turn(messages)
        turn["answer"] = turn["answer"].replace("supplied 60%", "supplied 6%")
        result = self.evaluate(messages, [turn])
        self.assertTrue(result["checks"]["process_exact_selected_results_turn_0"])
        self.assertFalse(result["checks"]["process_current_values_displayed_turn_0"])

    def test_stale_context_cannot_validate_new_turn_even_with_all_inputs_in_registry(self):
        messages = ["Concentrate iron is 60%. Is that typical?", "Concentrate iron is now 55%."]
        turns = [self.process_turn(messages[:1]), self.process_turn(messages)]
        turns[-1]["tool_result"]["data"]["analysis_plan"]["_request_context"]["messages"] = messages[:1]
        result = self.evaluate(messages, turns)
        self.assertFalse(result["checks"]["process_actual_input_bindings_turn_1"])

    def test_computed_classification_must_match_selected_evidence_values(self):
        messages = ["Concentrate iron is 60%. Is that typical?"]
        turn = self.process_turn(messages)
        trace = next(entry for entry in turn["trace"] if entry.get("action", {}).get("action") == "compare")
        trace["result"]["data"]["comparisons"][0]["range_status"] = "above_max"
        result = self.evaluate(messages, [turn])
        self.assertFalse(result["checks"]["process_exact_selected_results_turn_0"])

    def test_generic_distribution_checks_selected_column_and_actual_table_row(self):
        messages = ["Show the median of Invented Signal Flow."]
        def output(payload):
            return {"status": "ready", "question": None, "compare": [], "distribution": [FLOW], "count": [], "describe": False}
        turn = self.process_turn(messages, output)
        result = self.evaluate(messages, [turn])
        self.assertTrue(result["contract_checks_passed"], result["checks"])
        turn["answer"] = turn["answer"].replace("| " + FLOW + " |", "| Wrong Signal |")
        self.assertFalse(self.evaluate(messages, [turn])["checks"]["process_current_values_displayed_turn_0"])

    def test_count_display_preserves_predicate_combination_and_computed_count(self):
        messages = ["How many hours had Invented Signal Flow above 150 or concentrate silica below 5%?"]
        def output(payload):
            return {"status": "ready", "question": None, "compare": [], "distribution": [], "describe": False,
                    "count": [{"conditions": payload["available_operations"]["count_predicates"], "combine": "or"}]}
        turn = self.process_turn(messages, output)
        self.assertTrue(self.evaluate(messages, [turn])["contract_checks_passed"])
        wrong_combination = deepcopy(turn)
        wrong_combination["answer"] = wrong_combination["answer"].replace(" OR ", " AND ")
        self.assertFalse(self.evaluate(messages, [wrong_combination])["checks"]["process_current_values_displayed_turn_0"])
        wrong_count = deepcopy(turn)
        wrong_count["answer"] = wrong_count["answer"].replace("3 of 4", "2 of 4")
        self.assertFalse(self.evaluate(messages, [wrong_count])["checks"]["process_current_values_displayed_turn_0"])

    def test_product_compound_request_checks_both_real_subanswers(self):
        message = "Tell me more about the alpha controller, and do you have other recommendations?"
        result = self.evaluate([message], [product_turn(message)], "product")
        self.assertTrue(result["contract_checks_passed"], result["checks"])

    def test_task_text_uses_the_same_readable_citation_mapping_as_final_answer(self):
        from metso_assistant.presentation import readable_citations
        message = "Tell me more about the alpha controller."
        turn = product_turn(message, ("describe",))
        raw = turn["tool_result"]["data"]["task_results"][0]["answer"] + " [Balpha]"
        turn["tool_result"]["data"]["task_results"][0]["answer"] = raw
        turn["answer"] = readable_citations(raw, turn["sources"])
        result = self.evaluate([message], [turn], "product")
        self.assertTrue(result["checks"]["product_subanswers_evidenced_turn_0"])

    def test_omitted_compound_task_fails_even_when_all_declared_tasks_completed(self):
        message = "Tell me more about the alpha controller, and do you have other recommendations?"
        result = self.evaluate([message], [product_turn(message, ("describe",))], "product")
        self.assertTrue(result["checks"]["product_declared_tasks_executed_turn_0"])
        self.assertFalse(result["checks"]["product_explicit_requests_covered_turn_0"])

    def test_claimed_coverage_without_executed_result_or_rendered_text_fails(self):
        message = "Tell me more about the alpha controller."
        for missing in ("result", "text", "generation"):
            turn = product_turn(message, ("describe",))
            if missing == "result":
                turn["tool_result"]["data"]["task_results"] = []
            elif missing == "text":
                turn["tool_result"]["data"]["task_results"][0]["answer"] = "A task that never appears."
            else:
                turn["trace"] = [entry for entry in turn["trace"] if entry["operation"] != "grounded_product_generation"]
            self.assertFalse(self.evaluate([message], [turn], "product")["contract_checks_passed"], missing)

    def test_final_generation_gate_is_per_task_not_global_attempt_number(self):
        message = "Tell me more about the alpha controller, and do you have other recommendations?"
        turn = product_turn(message)
        turn["trace"][0]["validation_errors"] = ["Unapproved claim"]
        turn["trace"][1]["attempt"] = 2
        result = self.evaluate([message], [turn], "product")
        self.assertFalse(result["checks"]["product_citations_and_numeric_gate"])

    def test_named_detail_must_not_become_new_recommendation(self):
        message = "Tell me more about the alpha controller."
        turn = product_turn(message, ("describe",))
        turn["trace"][0]["raw"]["recommendation"] = "beta cell"
        result = self.evaluate([message], [turn], "product")
        self.assertFalse(result["checks"]["product_subanswers_evidenced_turn_0"])

    def test_why_request_requires_an_explanation_task(self):
        message = "Why could that happen?"
        result = self.evaluate([message], [product_turn(message, ("recommend",))], "product")
        self.assertFalse(result["checks"]["product_explicit_requests_covered_turn_0"])

    def test_required_recommendation_and_task_kind_are_nonvacuous_per_turn(self):
        message = "What could help this customer?"
        turn = product_turn(message, ("recommend",))
        requirements = {"requires_recommendation_turns": [0], "required_product_tasks_by_turn": {"0": ["recommend"]}}
        self.assertTrue(self.evaluate([message], [turn], "product", **requirements)["contract_checks_passed"])
        omitted = product_turn(message, ("describe",))
        checks = self.evaluate([message], [omitted], "product", **requirements)["checks"]
        self.assertFalse(checks["required_recommendation_turn_0"])
        self.assertFalse(checks["required_product_tasks_turn_0"])

    def test_source_identity_alias_is_used_instead_of_a_filename_product_guess(self):
        message = "Describe AC-7."
        turn = product_turn(message, ("describe",))
        tool = turn["tool_result"]["data"]["task_results"][0]["tool_result"]
        tool["sources"][0].update(file="unrelated-file-900.pdf", doc_id="opaque-900")
        tool["data"]["passages"] = [{"doc_id": "opaque-900", "source_id": "Balpha"}]
        tool["data"]["products"] = [{"doc_id": "opaque-900", "name": "alpha controller", "aliases": ["AC-7"], "source_file": "unrelated-file-900.pdf"}]
        requirements = {"required_product_tasks_by_turn": {"0": ["describe"]}, "required_product_subjects_by_turn": {"0": ["AC-7"]}}
        result = self.evaluate([message], [turn], "product", **requirements)
        self.assertTrue(result["contract_checks_passed"], result["checks"])
        task = turn["product_request_plan"]["tasks"][0]
        task["subject_product"] = "beta cell"
        turn["tool_result"]["data"]["request_plan"] = deepcopy(turn["product_request_plan"])
        turn["tool_result"]["data"]["task_results"][0]["task"] = deepcopy(task)
        turn["trace"][-1]["coverage"][0]["subject_product"] = "beta cell"
        self.assertFalse(self.evaluate([message], [turn], "product")["checks"]["product_explicit_requests_covered_turn_0"])

    def test_comparison_requires_both_named_products_and_never_a_recommendation(self):
        message = "Compare AC-7 and BC-2."
        turn = product_turn(message, ("compare",))
        requirements = {"required_product_tasks_by_turn": {"0": ["compare"]},
                        "required_product_subjects_by_turn": {"0": ["AC-7", "BC-2"]}}
        self.assertTrue(self.evaluate([message], [turn], "product", **requirements)["contract_checks_passed"])
        missing_source = deepcopy(turn)
        missing_source["tool_result"]["data"]["task_results"][0]["source_ids"] = ["Balpha"]
        missing_source["trace"][-1]["coverage"][0]["source_ids"] = ["Balpha"]
        result = self.evaluate([message], [missing_source], "product", **requirements)
        self.assertFalse(result["checks"]["required_product_subjects_turn_0"])
        self.assertFalse(result["checks"]["product_explicit_requests_covered_turn_0"])
        wrong_kind = product_turn(message, ("describe",))
        self.assertFalse(self.evaluate([message], [wrong_kind], "product")["checks"]["product_explicit_requests_covered_turn_0"])
        turn["trace"][0]["raw"]["recommendation"] = "beta cell"
        self.assertFalse(self.evaluate([message], [turn], "product")["checks"]["product_subanswers_evidenced_turn_0"])

    def test_visual_answer_needs_attached_evidence_or_an_explicit_partial_abstention(self):
        message = "Describe the alpha controller, and explain its control diagram."
        turn = product_turn(message, ("describe", "visual"))
        requirements = {"required_product_tasks_by_turn": {"0": ["describe", "visual"]},
                        "requires_visual_evidence_or_abstention_turns": [0]}
        self.assertFalse(self.evaluate([message], [turn], "product", **requirements)["checks"]["visual_evidence_or_abstention_turn_0"])
        visual_generation = turn["trace"][1]
        visual_generation.update(visual_attachments=[{"source_id": "Balpha", "path": "synthetic-diagram.png"}], visual_source_ids=["Balpha"])
        self.assertTrue(self.evaluate([message], [turn], "product", **requirements)["contract_checks_passed"])
        wrong_source = deepcopy(turn)
        wrong_source["trace"][1]["visual_source_ids"] = ["Bbeta"]
        self.assertFalse(self.evaluate([message], [wrong_source], "product", **requirements)["checks"]["visual_evidence_or_abstention_turn_0"])
        partial = deepcopy(turn)
        result = partial["tool_result"]["data"]["task_results"][1]
        result.update(status="insufficient_evidence", answer="The diagram is unavailable, so I cannot verify its connections.", source_ids=[])
        partial["trace"][1]["raw"].update(status="insufficient_evidence")
        partial["trace"][-1]["coverage"][1].update(status="insufficient_evidence", source_ids=[])
        partial["trace"][-1]["all_requested_tasks_completed"] = False
        partial.update(status="partial_answer", answer=partial["tool_result"]["data"]["task_results"][0]["answer"] + "\n\n" + result["answer"])
        requirements["allowed_statuses"] = [["partial_answer"]]
        self.assertTrue(self.evaluate([message], [partial], "product", **requirements)["contract_checks_passed"])
        partial["status"] = "answered"
        requirements["allowed_statuses"] = [["answered"]]
        self.assertFalse(self.evaluate([message], [partial], "product", **requirements)["checks"]["product_declared_tasks_executed_turn_0"])

    def test_new_request_boundary_resets_process_context_before_input_audit(self):
        messages = ["Concentrate iron is 60%. Is that typical?", "A different plant reports concentrate iron of 55%. Is that typical?"]
        turns = [self.process_turn([message]) for message in messages]
        turns[1]["trace"].insert(0, {"operation": "current_request_boundary", "new_request": True})
        result = self.evaluate(messages, turns, requires_new_request_boundary_turns=[1])
        self.assertTrue(result["contract_checks_passed"], result["checks"])
        turns[1]["trace"].pop(0)
        result = self.evaluate(messages, turns, requires_new_request_boundary_turns=[1])
        self.assertFalse(result["checks"]["new_request_boundary_turn_1"])
        self.assertFalse(result["checks"]["process_actual_input_bindings_turn_1"])

    def test_requested_time_scope_or_percentile_decline_has_no_hidden_computation(self):
        for message, kind in [("Show only June concentrate silica means.", "unsupported_time_scope"),
                              ("Show the 90th percentile of concentrate silica.", "unsupported_statistic")]:
            turn = self.process_turn([message])
            requirements = {"allowed_statuses": [["insufficient_evidence"]], "requires_no_process_calculation_turns": [0],
                            "required_process_unsupported_kinds_by_turn": {"0": [kind]}}
            self.assertTrue(self.evaluate([message], [turn], **requirements)["contract_checks_passed"])
            hidden = deepcopy(turn)
            hidden["trace"].append({"operation": "adaptive_tool", "action": {"action": "distribution"}})
            self.assertFalse(self.evaluate([message], [hidden], **requirements)["checks"]["unsupported_process_not_calculated_turn_0"])
            hidden = deepcopy(turn)
            hidden["answer"] = "These are typical readings."
            self.assertFalse(self.evaluate([message], [hidden], **requirements)["checks"]["required_process_capability_decline_turn_0"])

    def test_requested_process_diagnostics_must_be_computed_and_visible(self):
        message = "What period is recorded, and are there timestamp gaps?"
        def output(payload):
            return {"status": "ready", "question": None, "compare": [], "distribution": [], "count": [], "describe": True}
        turn = self.process_turn([message], output)
        requirements = {"required_process_description_sections_by_turn": {"0": ["coverage", "gaps"]}}
        self.assertTrue(self.evaluate([message], [turn], **requirements)["contract_checks_passed"])
        hidden = deepcopy(turn)
        hidden["answer"] = "Available CSV columns and units: " + IRON
        self.assertFalse(self.evaluate([message], [hidden], **requirements)["checks"]["required_process_diagnostics_turn_0"])
        self.assertFalse(self.evaluate([message], [hidden])["checks"]["process_current_values_displayed_turn_0"])

    def test_expected_process_values_and_shared_header_are_dynamic(self):
        message = "Concentrate has 6% silica and 60% iron. Are these usual?"
        turn = self.process_turn([message])
        requirements = {"expected_process_readings_by_turn": {"0": {SILICA: 6, IRON: 60}},
                        "requires_shared_process_population_header_turns": [0]}
        self.assertTrue(self.evaluate([message], [turn], **requirements)["contract_checks_passed"])
        wrong_values = {**requirements, "expected_process_readings_by_turn": {"0": {SILICA: 60, IRON: 6}}}
        self.assertFalse(self.evaluate([message], [turn], **wrong_values)["checks"]["expected_current_process_readings_turn_0"])
        turn["answer"] = turn["answer"].replace("From 4 hourly historical observations:", "From 400 hourly historical observations:")
        self.assertFalse(self.evaluate([message], [turn], **requirements)["checks"]["shared_process_population_header_turn_0"])

    def test_rejected_duration_must_not_reach_money_even_without_coverage_labels(self):
        turn = travel_turn()
        data = turn["tool_result"]["data"]
        data["travel_context"] = None
        data["facts"] = [fact for fact in data["facts"] if fact["id"] in {"R1", "R2", "D8"}]
        data["duration_inputs"][0].update(status="ambiguous", total_hours=None)
        question = "What is the confirmed total billed travel time?"
        turn["tool_result"]["question"] = question
        turn["answer"] = question
        requirements = {"requires_clarification_turns": [0], "forbidden_priced_duration_roles_by_turn": {"0": ["travel_hours"]}}
        self.assertTrue(self.evaluate(["Travel might be between 4 and 8 hours."], [turn], "service_cost", **requirements)["contract_checks_passed"])
        missing_question = deepcopy(turn)
        missing_question["answer"] = "A partial budget."
        self.assertFalse(self.evaluate(["Uncertain travel."], [missing_question], "service_cost", **requirements)["checks"]["required_clarification_turn_0"])
        data["facts"].append({"id": "WRONG", "kind": "calculation", "unit": "money", "currency": "USD", "value": Decimal(88), "parents": ["R1", "D8"], "scope": {"operation": "multiply"}})
        self.assertFalse(self.evaluate(["Uncertain travel."], [turn], "service_cost", **requirements)["checks"]["rejected_duration_not_priced_turn_0"])

    def test_excluded_material_category_cannot_hide_inside_a_total(self):
        turn = travel_turn()
        requirements = {"forbidden_charged_categories_by_turn": {"0": ["material"]}}
        self.assertFalse(self.evaluate(["We already own the materials."], [turn], "service_cost", **requirements)["checks"]["excluded_category_not_charged_turn_0"])
        data = turn["tool_result"]["data"]
        data["facts"] = [fact for fact in data["facts"] if fact["id"] not in {"B", "T1", "T2"}]
        data["subtotal_ranges_by_currency"] = {}
        self.assertTrue(self.evaluate(["We already own the materials."], [turn], "service_cost", **requirements)["contract_checks_passed"])


    def test_allowed_status_contract_must_have_one_entry_per_turn(self):
        turn = {"route": "unsupported", "status": "unsupported", "answer": "Unsupported.", "sources": [], "trace": []}
        result = evaluate_case({"id": "bad-contract", "messages": ["one", "two"], "expected_routes": ["unsupported"]*2,
                                "allowed_statuses": [["unsupported"]]}, Assistant([turn, turn]))
        self.assertFalse(result["checks"]["statuses"])

    def test_country_travel_context_and_optional_benchmark_contracts_pass(self):
        turn = travel_turn()
        result = self.evaluate(["Use eight total travel hours in Chile."], [turn], "service_cost",
            requires_country_travel_turns=[0], expected_travel_countries=["Chile"], requires_travel_range_turns=[0])
        self.assertTrue(result["contract_checks_passed"], result["checks"])

    def test_country_travel_omitted_rate_or_foreign_country_cannot_pass(self):
        omitted = travel_turn()
        omitted["tool_result"]["data"]["travel_context"]["cohorts"][0]["fact_ids"] = ["R1"]
        self.assertFalse(self.evaluate(["Travel in Chile."], [omitted], "service_cost")["checks"]["travel_exact_country_cohort_turn_0"])
        foreign = travel_turn()
        foreign["tool_result"]["data"]["facts"][1]["scope"]["country"] = "Sweden"
        self.assertFalse(self.evaluate(["Travel in Chile."], [foreign], "service_cost")["checks"]["travel_validated_rate_sources_turn_0"])

    def test_country_travel_endpoints_must_use_confirmed_hours_and_exact_arithmetic(self):
        for wrong in ("hours", "amount"):
            turn = travel_turn()
            if wrong == "hours":
                turn["tool_result"]["data"]["duration_inputs"][0]["total_hours"] = Decimal(16)
            else:
                next(fact for fact in turn["tool_result"]["data"]["facts"] if fact["id"] == "A2")["value"] = Decimal("185.00")
            result = self.evaluate(["Travel in Chile."], [turn], "service_cost")
            self.assertFalse(result["checks"]["travel_endpoint_arithmetic_turn_0"], wrong)

    def test_country_travel_bounds_are_never_added_to_one_another(self):
        turn = travel_turn()
        turn["tool_result"]["data"]["facts"].append({"id": "WRONG", "kind": "calculation", "unit": "money", "currency": "USD",
            "value": Decimal("272.00"), "parents": ["A1", "A2"], "scope": {"operation": "sum"}})
        result = self.evaluate(["Travel in Chile."], [turn], "service_cost")
        self.assertFalse(result["checks"]["travel_alternative_subtotals_turn_0"])

    def test_country_travel_range_refs_preserve_currency_and_displayed_local_context(self):
        wrong_currency = travel_turn()
        next(fact for fact in wrong_currency["tool_result"]["data"]["facts"] if fact["id"] == "T2")["currency"] = "EUR"
        self.assertFalse(self.evaluate(["Travel in Chile."], [wrong_currency], "service_cost")["checks"]["travel_alternative_subtotals_turn_0"])
        hidden = travel_turn()
        hidden["answer"] = "No useful travel information is shown."
        self.assertFalse(self.evaluate(["Travel in Chile."], [hidden], "service_cost")["checks"]["travel_context_displayed_turn_0"])

    def test_null_travel_context_is_a_failed_requested_contract_without_crashing(self):
        turn = travel_turn()
        turn["tool_result"]["data"]["travel_context"] = None
        result = self.evaluate(["Travel in Chile."], [turn], "service_cost", expected_travel_countries=["Chile"],
            requires_country_travel_turns=[0], requires_travel_range_turns=[0], requires_unpriced_travel_turns=[0])
        self.assertFalse(result["contract_checks_passed"])
        self.assertFalse(result["checks"]["country_travel_available_turn_0"])


if __name__ == "__main__":
    unittest.main()
