import base64
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metso_assistant.agent import Assistant, HYPOTHESIS_INTRO, FIT_REVIEW_SYSTEM, TEXT_REVIEW_SYSTEM, EXTRACT_SYSTEM, evidence_quotes, validate_answer, expand_extract_selection
from metso_assistant.product_request import SYSTEM as PRODUCT_REQUEST_SYSTEM
from tests.test_model import routed


def evidence():
    return {"tool": "product", "status": "ok", "data": {"passages": [], "facts": []},
            "sources": [{"id": "B1", "file": "synthetic.pdf", "page": 2, "sha": "a" * 64,
                         "excerpt": "The Example optimizer adjusts setpoints when slurry characteristics change."}],
            "warnings": [], "missing_fields": [], "question": None}


def answer(**changes):
    value = {"status": "answered", "recommendation": "Example optimizer",
             "claims": [{"kind": "fit", "text": "adjusts setpoints when slurry characteristics change",
                         "source_id": "B1", "quote": "adjusts setpoints when slurry characteristics change", "fact_ids": []}],
             "uncertainties": []}
    value.update(changes)
    return value


class EvidenceBoundaryTests(unittest.TestCase):
    def test_distinct_mechanism_and_fit_slots_cannot_duplicate_a_claim_kind(self):
        extracts = [{"excerpt_id": "Q1", "source_id": "B1", "text": "Recipes can be difficult to adjust."},
                    {"excerpt_id": "Q2", "source_id": "B1", "text": "The optimizer adjusts setpoints."}]
        selection = {"recommendation": "Example optimizer", "uncertainties": [],
                     "claims": {"hypothesis": "Q1", "fit": "Q2"}}
        result = expand_extract_selection(selection, extracts)
        self.assertEqual([row["kind"] for row in result["claims"]], ["hypothesis", "fit"])
        self.assertEqual(result["claims"][0]["text"], HYPOTHESIS_INTRO)
        self.assertEqual(result["claims"][1]["quote"], extracts[1]["text"])
        selection["claims"] = {"fit": "Q2"}
        with self.assertRaises(ValueError):
            expand_extract_selection(selection, extracts)

    def test_extract_id_binds_the_exact_text_and_source_without_model_rewriting(self):
        selections = {"status": "answered", "recommendation": "Example optimizer", "uncertainties": [],
                      "claims": [{"kind": "fit", "excerpt_id": "Q2"}]}
        extracts = [{"excerpt_id": "Q1", "source_id": "B404", "text": "An unrelated title."},
                    {"excerpt_id": "Q2", "source_id": "B1", "text": "adjusts setpoints when slurry characteristics change"}]
        expanded = expand_extract_selection(selections, extracts)
        self.assertEqual(expanded["claims"][0]["text"], extracts[1]["text"])
        self.assertEqual(expanded["claims"][0]["quote"], extracts[1]["text"])
        self.assertEqual(expanded["claims"][0]["source_id"], "B1")
        self.assertEqual(validate_answer(expanded, evidence(), "changing slurry"), [])
        for claim in [{"kind": "fit", "excerpt_id": "Q999"},
                      {"kind": "fit", "excerpt_id": "Q2", "text": "Invented capability"}]:
            selections["claims"] = [claim]
            with self.assertRaises(ValueError):
                expand_extract_selection(selections, extracts)

    def test_quote_choices_preserve_source_list_punctuation_and_bound_length(self):
        source = {"id": "B1", "excerpt": "Adjusted variables:\n\nFeed amount\n\nPressure difference\n\nDrying time"}
        options = evidence_quotes([source])
        self.assertTrue(any("Feed amount Pressure difference Drying time" in x for x in options))
        for quote in options:
            self.assertIn(" ".join(quote.split()).casefold(), " ".join(source["excerpt"].split()).casefold())
            self.assertLessEqual(len(quote.split()), 60)
            self.assertNotIn(",", quote)

    def test_unknown_citation_and_fabricated_quote_fail(self):
        for changes in ({"source_id": "B404"}, {"quote": "guarantees perfect recovery"}):
            result = answer()
            result["claims"][0].update(changes)
            self.assertTrue(validate_answer(result, evidence(), "help"))

    def test_exact_extract_passes_but_generated_paraphrase_needs_no_semantic_guess(self):
        self.assertEqual(validate_answer(answer(), evidence(), "help"), [])
        changed = answer()
        changed["claims"][0]["text"] = "The Example optimizer is a pressure filter for changing feed grades."
        self.assertTrue(any("unsupported paraphrase" in error for error in validate_answer(changed, evidence(), "help")))

    def test_customer_specific_possibility_cannot_masquerade_as_documented(self):
        result = answer()
        result["claims"][0].update(kind="documented", text="This may explain the customer's performance drop.")
        self.assertTrue(any("hypothesis" in e for e in validate_answer(result, evidence(), "why")))

    def test_cycle_frequency_requires_the_claims_own_supporting_quote(self):
        tool = evidence()
        tool["sources"][0]["excerpt"] += " Traditional recipes can be difficult to adjust. The Example optimizer adjusts cycle by cycle."
        result = answer()
        for phrase in ["Traditional controls cannot respond cycle-by-cycle.", "Traditional controls do not update each cycle."]:
            result["claims"][0].update(text=phrase, quote="Traditional recipes can be difficult to adjust.")
            self.assertTrue(any("cycle-frequency" in e for e in validate_answer(result, tool, "why")))
        result["claims"][0].update(text="The Example optimizer adjusts cycle-by-cycle.", quote="The Example optimizer adjusts cycle by cycle.")
        self.assertEqual(validate_answer(result, tool, "how"), [])

    def test_hypotheses_use_extracted_challenges_without_a_generated_diagnosis(self):
        tool = evidence()
        tool["sources"][0]["excerpt"] += " Changing slurry can be difficult to handle."
        result = answer(recommendation=None)
        result["claims"][0].update(kind="hypothesis", text="One possibility is that rapid changes broke the control system.", quote="Changing slurry can be difficult to handle.")
        self.assertTrue(any("generated diagnosis" in e for e in validate_answer(result, tool, "why")))
        result["claims"][0]["text"] = HYPOTHESIS_INTRO
        self.assertEqual(validate_answer(result, tool, "why"), [])
        result["claims"][0]["quote"] = "adjusts setpoints when slurry characteristics change"
        self.assertTrue(any("documented challenge" in e for e in validate_answer(result, tool, "why")))

    def test_hypothesis_quotation_cannot_bypass_numerical_authority(self):
        tool = evidence()
        tool["sources"][0]["excerpt"] += " It is difficult to achieve 90 percent recovery."
        result = answer(recommendation=None)
        result["claims"][0].update(kind="hypothesis", text=HYPOTHESIS_INTRO, quote="It is difficult to achieve 90 percent recovery.")
        self.assertTrue(any("qualitative" in e for e in validate_answer(result, tool, "why")))

    def test_narrative_numeric_claim_cannot_be_promoted(self):
        tool = evidence()
        tool["sources"][0]["excerpt"] += " Capacity is 900 tonnes."
        result = answer()
        result["claims"][0].update(text="It handles 900 tonnes.", quote="Capacity is 900 tonnes.")
        self.assertTrue(validate_answer(result, tool, "capacity"))

    def test_numeric_fact_requires_matching_approved_source(self):
        tool = evidence()
        tool["sources"][0]["excerpt"] = "Height: 500 mm."
        fact = {"fact_id": "f1", "source_id": "B1", "review_status": "verified", "status": "reported", "value": 500, "configuration": "synthetic"}
        tool["data"]["facts"] = [fact]
        result = answer(recommendation=None)
        result["claims"][0].update(text="Height: 500 mm.", quote="Height: 500 mm.", fact_ids=["f1"])
        self.assertEqual(validate_answer(result, tool, "height"), [])
        fact["review_status"] = "pending"
        self.assertTrue(validate_answer(result, tool, "height"))

    def test_made_up_product_name_fails(self):
        self.assertTrue(validate_answer(answer(recommendation="Example miracle press"), evidence(), "help"))

    def test_uncertainty_field_cannot_hide_an_invented_percentage(self):
        self.assertTrue(validate_answer(answer(uncertainties=["It could improve recovery by 77%, but this is uncertain."]), evidence(), "help"))

    def test_spelled_out_quantities_and_multipliers_cannot_bypass_numeric_boundary(self):
        for claim in ["It guarantees forty percent improvement.", "The visit takes two and a half hours.",
                      "It has twenty-five chambers.", "The budget is USD five hundred.",
                      "It recovers particles below twenty microns.",
                      "It doubles throughput.", "It reduces energy consumption by half."]:
            with self.subTest(claim=claim):
                result = answer(); result["claims"][0]["text"] = claim
                self.assertTrue(validate_answer(result, evidence(), "help"))

    def test_word_number_uncertainty_cannot_hide_an_unapproved_quantity(self):
        self.assertTrue(validate_answer(answer(uncertainties=["It may improve recovery by forty percent, but I am uncertain."]), evidence(), "help"))

    def test_ordinary_qualitative_adjustment_and_double_sided_label_stay_valid(self):
        for text in ["The double-sided Example filter can reduce manual recipe adjustments.",
                     "The double-sided Example filter supports consistent performance.",
                     "The Example filter has a double-acting cylinder for steady output."]:
            with self.subTest(text=text):
                tool = evidence(); tool["sources"][0]["excerpt"] = text
                result = answer(recommendation="Example filter", uncertainties=["The current integration needs confirmation."])
                result["claims"][0].update(text=text, quote=text)
                self.assertEqual(validate_answer(result, tool, "help"), [])

    def test_recommendation_cannot_borrow_another_machines_capability(self):
        tool = evidence()
        tool["sources"].append({"id": "B2", "excerpt": "The separate Example pump handles thick slurry."})
        result = answer()
        result["claims"].append({"kind": "fit", "text": "The optimizer handles thick slurry.", "source_id": "B2", "quote": "handles thick slurry", "fact_ids": []})
        self.assertTrue(validate_answer(result, tool, "help", recommendation_source_ids={"B1"}))


def request_plan(message, kind="recommend", *, continuation=False):
    return {"context_status": "continue_need" if continuation else "new_need",
            "need_quote": None if continuation else message, "equipment_quote": None,
            "tasks": [{"kind": kind, "question_quote": message, "subject_product": None}]}


class FakeModel:
    def __init__(self, routes, generated=None, plans=None):
        self.routes = iter(routes)
        self.generated = iter(generated or [])
        self.plans = iter(plans) if plans is not None else None
        self.histories = []
        self.invocations = []
        self.plan_invocations = []

    def route(self, history):
        self.histories.append(deepcopy(history))
        return next(self.routes), [{"operation": "synthetic_router", "synthetic": True}]

    def invoke(self, *args, **kwargs):
        if args[0] == FIT_REVIEW_SYSTEM:
            return {"quote_role": "product_capability", "relevant": True}, {"synthetic": True}
        if args[0] == TEXT_REVIEW_SYSTEM:
            return {"substantive": True, "addresses_request": True}, {"synthetic": True}
        if args[0] == PRODUCT_REQUEST_SYSTEM:
            self.plan_invocations.append(deepcopy(args))
            import json
            current = json.loads(args[1][0]["content"])["current_question"]
            return (next(self.plans) if self.plans is not None else request_plan(current)), {"synthetic": True}
        self.invocations.append(deepcopy(args))
        return next(self.generated), {"synthetic": True}


class OrchestrationTests(unittest.TestCase):
    def test_library_and_explicit_current_engine_agree_and_baseline_has_recovery_instructions(self):
        self.assertEqual(Assistant(model=FakeModel([])).engine, "adaptive")
        self.assertEqual(Assistant(model=FakeModel([]), engine="adaptive").engine, "adaptive")
        with self.assertRaisesRegex(ValueError, "baseline engine is retired.*supported adaptive engine"):
            Assistant(model=FakeModel([]), engine="baseline")

    def test_both_entrypoints_reject_baseline_with_recovery_before_any_model_or_source_loading(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from metso_assistant.__main__ import main as chat_main
        from metso_assistant.evaluate import main as evaluation_main
        for entrypoint in (chat_main, evaluation_main):
            with self.subTest(entrypoint=entrypoint.__module__):
                errors = StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as stopped:
                    entrypoint(["--engine", "baseline"])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("baseline engine is retired", errors.getvalue())
                self.assertIn("--engine adaptive", errors.getvalue())

    def test_clarification_recovers_source_named_detail_without_retired_need_or_service_fields(self):
        current = "Describe Example optimizer."
        catalog = [{"name": "Example optimizer", "doc_id": "example-optimizer", "aliases": ["optimizer"],
                    "status": "source_supported", "source_sha256": "a" * 64,
                    "identity_evidence": [{"doc_id": "example-optimizer", "page": 2,
                        "source_sha256": "a" * 64, "quote": "Example optimizer", "authority": "source_identity_only"}]}]
        requested = {"context_status": "subject_only", "need_quote": None, "equipment_quote": None,
                     "tasks": [{"kind": "describe", "question_quote": current, "subject_product": "Example optimizer"}]}
        documented = answer(recommendation=None)
        documented["claims"][0]["kind"] = "documented"
        model = FakeModel([routed("clarify", equipment="Old press", country="Exampleland", cloth_count=7)],
                          [documented], plans=[requested])
        calls = []
        assistant = Assistant(model=model, tools={"product": lambda args: calls.append(deepcopy(args)) or evidence()})
        assistant.history = [{"role": "user", "content": "Our old pump loses pressure."},
                             {"role": "assistant", "content": "An old controller might be relevant."}]
        assistant.context = {"last_route": "product", "product": {
            "subject_query": "Our old pump loses pressure.", "equipment": "old pump",
            "recommendation": "old controller", "discussed_subject": "old controller"}}
        with patch("metso_assistant.agent.product_catalog", return_value=catalog):
            result = assistant.turn(current)
        self.assertEqual(result["raw_routing"]["route"], "clarify")
        self.assertEqual((result["route"], result["status"]), ("product", "answered"))
        self.assertTrue(all(value is None for value in result["routing"]["known_fields"].values()))
        self.assertTrue(all("old pump" not in call["query"] for call in calls))
        payload = json.loads(model.plan_invocations[0][1][0]["content"])
        self.assertEqual(payload["actual_user_messages"], [current])
        self.assertNotIn("old controller", json.dumps(payload))
        self.assertIsNone(result["conversation_context"]["product"].get("recommendation"))
        proof = next(row for row in result["trace"] if row["operation"] == "product_named_subject_recovery")
        self.assertEqual(proof["source_identity_proof"]["matches"][0]["doc_id"], "example-optimizer")

    def test_clarification_does_not_recover_generic_category_reference_from_search_alias(self):
        catalog = [{"name": "Example optimizer", "doc_id": "example-optimizer", "aliases": ["optimizer"],
                    "status": "source_supported", "source_sha256": "a" * 64,
                    "identity_evidence": [{"doc_id": "example-optimizer", "page": 2,
                        "source_sha256": "a" * 64, "quote": "Example optimizer", "authority": "source_identity_only"}]}]
        assistant = Assistant(model=FakeModel([routed("clarify")]), tools={})
        with patch("metso_assistant.agent.product_catalog", return_value=catalog):
            result = assistant.turn("Describe this optimizer.")
        self.assertEqual((result["route"], result["status"]), ("clarify", "clarify"))
        self.assertFalse(any(row["operation"] == "product_named_subject_recovery" for row in result["trace"]))

    def test_native_description_prefers_available_prose_over_cover_fragments(self):
        import json
        seen = []
        class NativeSelectionModel:
            def invoke(self, system, messages, schema, **kwargs):
                if system == TEXT_REVIEW_SYSTEM:
                    return {"substantive": True, "addresses_request": True}, {"synthetic": True}
                assert system == EXTRACT_SYSTEM
                payload = json.loads(messages[0]["content"])
                seen.extend(row["text"] for row in payload["evidence_extracts"])
                return {"recommendation": None, "claims": [{"kind": "documented",
                    "excerpt_id": payload["evidence_extracts"][0]["excerpt_id"]}], "uncertainties": []}, {"synthetic": True}
        tool = evidence()
        tool["sources"].insert(0, {"id": "Bcover", "file": "synthetic.pdf", "page": 1,
            "excerpt": "Unprecedented outcomes and efficiency"})
        text, status, _, _ = Assistant(model=NativeSelectionModel())._product_answer(
            tool, "Describe the product.", intent="describe")
        self.assertEqual(status, "answered")
        self.assertIn("adjusts setpoints", text)
        self.assertTrue(seen)
        self.assertFalse(any("Unprecedented" in quote for quote in seen))
        seen.clear()
        fragments = evidence()
        fragments["sources"][0]["excerpt"] = "Feed amount\n\nPressure difference\n\nDrying time"
        _, status, _, _ = Assistant(model=NativeSelectionModel())._product_answer(
            fragments, "Which variables are listed?", intent="describe")
        self.assertEqual(status, "answered")
        self.assertTrue(all(any(label in quote for quote in seen) for label in (
            "Feed amount", "Pressure difference", "Drying time")))

    def test_nested_description_quotes_do_not_count_as_two_details(self):
        tool = evidence()
        tool["sources"][0]["excerpt"] += " Example optimizer: Better outcomes, more value."
        repeated = answer(recommendation=None)
        repeated["claims"] = [{"kind": "documented", "text": quote, "quote": quote,
            "source_id": "B1", "fact_ids": []} for quote in (
                "Better outcomes, more value.", "Example optimizer: Better outcomes, more value.")]
        feature = answer(recommendation=None)
        feature["claims"][0]["kind"] = "documented"
        text, status, _, traces = Assistant(model=FakeModel([], [repeated, feature]))._product_answer(
            tool, "Describe its purpose.", intent="describe")
        self.assertEqual(status, "answered")
        self.assertNotIn("Better outcomes", text)
        first = next(row for row in traces if row["operation"] == "grounded_product_generation")
        self.assertTrue(any("distinct" in error for error in first["validation_errors"]))

    def test_description_review_replaces_source_exact_slogans_with_a_substantive_feature(self):
        class ReviewingModel(FakeModel):
            reviews = iter([{"substantive": False, "addresses_request": False},
                            {"substantive": True, "addresses_request": True}])
            def invoke(self, system, *args, **kwargs):
                if system == TEXT_REVIEW_SYSTEM:
                    return next(self.reviews), {"synthetic": True}
                return super().invoke(system, *args, **kwargs)
        tool = evidence()
        tool["sources"][0]["excerpt"] += " Better outcomes, more value."
        slogan = answer(recommendation=None)
        slogan["claims"] = [{"kind": "documented", "text": "Better outcomes, more value.",
            "quote": "Better outcomes, more value.", "source_id": "B1", "fact_ids": []}]
        feature = answer(recommendation=None)
        feature["claims"][0]["kind"] = "documented"
        assistant = Assistant(model=ReviewingModel([], [slogan, feature]))
        text, status, _, traces = assistant._product_answer(tool, "Describe its features.", intent="describe")
        self.assertEqual(status, "answered")
        self.assertNotIn("Better outcomes", text)
        self.assertIn("adjusts setpoints", text)
        self.assertEqual(len([row for row in traces if row["operation"] == "product_text_task_review"]), 2)

    def test_incomplete_description_is_withheld_after_bounded_repair(self):
        class RejectingModel(FakeModel):
            def invoke(self, system, *args, **kwargs):
                if system == TEXT_REVIEW_SYSTEM:
                    return {"substantive": False, "addresses_request": False}, {"synthetic": True}
                return super().invoke(system, *args, **kwargs)
        feature = answer(recommendation=None)
        feature["claims"][0]["kind"] = "documented"
        text, status, sources, _ = Assistant(model=RejectingModel([], [feature, feature]))._product_answer(
            evidence(), "Describe its features.", intent="describe")
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(sources, [])
        self.assertNotIn("adjusts setpoints", text)

    def test_visible_fit_review_rejects_and_repairs_an_unsupported_role(self):
        review_inputs = []
        class ReviewingModel(FakeModel):
            reviews = iter([{"quote_role": "problem_statement", "relevant": False},
                            {"quote_role": "product_capability", "relevant": True}])
            def invoke(self, system, *args, **kwargs):
                if system == FIT_REVIEW_SYSTEM:
                    review_inputs.append(str(args[0]))
                    return next(self.reviews), {"synthetic": True}
                return super().invoke(system, *args, **kwargs)
        repaired = answer()
        repaired["claims"][0].update(text="Hidden capabilities elsewhere in this paragraph.", quote="Hidden capabilities elsewhere in this paragraph.")
        model = ReviewingModel([routed("product")], [answer(), repaired])
        tool = evidence()
        tool["sources"][0]["excerpt"] += " Hidden capabilities elsewhere in this paragraph."
        result = Assistant(model=model, tools={"product": lambda args: tool}).turn("What product could help?")
        self.assertEqual(result["status"], "answered")
        generation = [row for row in result["trace"] if row["operation"] == "grounded_product_generation"]
        self.assertEqual(len(generation), 2)
        self.assertTrue(generation[0]["validation_errors"])
        self.assertFalse(generation[1]["validation_errors"])
        self.assertEqual(len(review_inputs), 2)
        self.assertNotIn("Hidden capabilities", review_inputs[0])
        self.assertIn("Hidden capabilities", review_inputs[1])

    def test_native_fit_retry_excludes_duplicate_quotes_only_from_fit_role(self):
        challenge = "Example optimizer recipes can be difficult to adjust."
        capability = "The Example optimizer adjusts setpoints when slurry characteristics change."
        for mechanism in (False, True):
            for ignore_retry_enum in (False, True):
                with self.subTest(mechanism=mechanism, ignores_enum=ignore_retry_enum):
                    captured, reviews = [], []
                    class NativeModel:
                        def invoke(self, system, messages, schema, **kwargs):
                            if system == FIT_REVIEW_SYSTEM:
                                reviews.append(json.loads(messages[0]["content"])["visible_fit_quotation"])
                                return {"quote_role": "problem_statement" if len(reviews) == 1 else "product_capability",
                                        "relevant": len(reviews) != 1}, {"synthetic": True}
                            if system != EXTRACT_SYSTEM:
                                return {"product_name": "Example optimizer"}, {"synthetic": True}
                            payload = json.loads(messages[0]["content"])
                            captured.append((deepcopy(schema), payload))
                            entries = payload["evidence_extracts"]
                            first = next(row["excerpt_id"] for row in entries if row["source_id"] == "B1")
                            duplicate = next(row["excerpt_id"] for row in entries if row["source_id"] == "B2")
                            good = next(row["excerpt_id"] for row in entries if row["text"] == capability)
                            fit = first if len(captured) == 1 else duplicate if ignore_retry_enum else good
                            claims = ({"hypothesis": first, "fit": fit} if mechanism else
                                      [{"kind": "fit", "excerpt_id": fit}])
                            return {"recommendation": "Example optimizer", "claims": claims, "uncertainties": []}, {"synthetic": True}
                    tool = evidence()
                    tool["sources"] = [dict(tool["sources"][0], id=source_id, excerpt=text)
                                       for source_id, text in (("B1", challenge),
                                           ("B2", "EXAMPLE optimizer recipes can be difficult to adjust."), ("B3", capability))]
                    tool["data"].update(products=[{"doc_id": "example-optimizer", "name": "Example optimizer",
                        "identity_evidence": [{"quote": "Example optimizer"}]}],
                        passages=[{"doc_id": "example-optimizer", "source_id": row["id"]} for row in tool["sources"]])
                    query = "Our performance drops when feed changes. What can help?" if mechanism else "What product could help adjust setpoints?"
                    text, status, sources, traces = Assistant(model=NativeModel())._product_answer(tool, query)
                    self.assertEqual(len(captured), 2)
                    first_ids = {row["excerpt_id"] for row in captured[0][1]["evidence_extracts"] if row["source_id"] in {"B1", "B2"}}
                    retry_claim_schema = captured[1][0]["properties"]["claims"]
                    retry_fit_ids = (retry_claim_schema["properties"]["fit"]["enum"] if mechanism else
                                     retry_claim_schema["items"]["properties"]["excerpt_id"]["enum"])
                    self.assertTrue(first_ids.isdisjoint(retry_fit_ids))
                    self.assertEqual(captured[1][1]["admissible_fit_excerpt_ids"], retry_fit_ids)
                    self.assertEqual(captured[0][1]["evidence_extracts"], captured[1][1]["evidence_extracts"])
                    if mechanism:
                        self.assertTrue(first_ids <= set(retry_claim_schema["properties"]["hypothesis"]["enum"]))
                    retry_scope = next(row for row in traces if row["operation"] == "product_fit_retry_scope")
                    self.assertEqual(set(retry_scope["excluded_fit_excerpt_ids"]), first_ids)
                    if ignore_retry_enum:
                        self.assertEqual(status, "insufficient_evidence")
                        self.assertEqual(len(reviews), 1)
                        self.assertEqual(sources, [])
                        self.assertNotIn("worth assessing", text)
                        self.assertTrue(any("already rejected" in error for row in traces
                            if row["operation"] == "grounded_product_generation" for error in row["validation_errors"]))
                    else:
                        self.assertEqual(status, "answered")
                        self.assertEqual(len(reviews), 2)
                        self.assertIn(capability, text)
                        if mechanism:
                            self.assertIn(HYPOTHESIS_INTRO, text)

    def test_fit_retry_exhaustion_stops_without_an_empty_enum_invocation(self):
        calls = []
        class NativeModel:
            def invoke(self, system, messages, schema, **kwargs):
                if system == FIT_REVIEW_SYSTEM:
                    return {"quote_role": "problem_statement", "relevant": False}, {"synthetic": True}
                if system != EXTRACT_SYSTEM:
                    return {"product_name": "Example optimizer"}, {"synthetic": True}
                calls.append(deepcopy(schema))
                extracts = json.loads(messages[0]["content"])["evidence_extracts"]
                return {"recommendation": "Example optimizer", "claims": [{"kind": "fit", "excerpt_id": extracts[0]["excerpt_id"]}],
                        "uncertainties": []}, {"synthetic": True}
        tool = evidence()
        tool["sources"][0]["excerpt"] = "The Example optimizer faces difficult conditions."
        tool["data"]["passages"] = [{"doc_id": "example-optimizer", "source_id": "B1"}]
        tool["data"]["products"] = [{"doc_id": "example-optimizer", "name": "Example optimizer",
                                      "identity_evidence": [{"quote": "Example optimizer"}]}]
        text, status, sources, traces = Assistant(model=NativeModel())._product_answer(tool, "What product could help?")
        self.assertEqual(status, "insufficient_evidence")
        self.assertEqual(len(calls), 1)
        self.assertEqual(sources, [])
        self.assertNotIn("worth assessing", text)
        retry_scope = next(row for row in traces if row["operation"] == "product_fit_retry_scope")
        self.assertFalse(retry_scope["retry_available"])
        self.assertEqual(retry_scope["admissible_fit_excerpt_ids"], [])

    def test_repeatedly_unestablished_fit_returns_insufficient_without_candidate_claims(self):
        class RejectingModel(FakeModel):
            def invoke(self, system, *args, **kwargs):
                if system == FIT_REVIEW_SYSTEM:
                    return {"quote_role": "problem_statement", "relevant": True}, {"synthetic": True}
                return super().invoke(system, *args, **kwargs)
        result = Assistant(model=RejectingModel([routed("product")], [answer(), answer()]),
            tools={"product": lambda args: evidence()}).turn("What product could help?")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("worth assessing", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_declared_specification_renders_reviewed_facts_even_for_mixed_wording(self):
        tool = evidence()
        tool["data"].update(request_kind="mixed", facts=[{"fact_id": "f1", "source_id": "B1", "doc_id": "synthetic",
            "parameter": "Height", "configuration": "H1", "unit": "mm", "value": 450,
            "trusted": True, "review_status": "verified", "status": "reported"}])
        model = FakeModel([routed("product")], plans=[request_plan("Describe the dimensions of Example.", "specification")])
        result = Assistant(model=model, tools={"product": lambda args: tool}).turn("Describe the dimensions of Example.")
        self.assertEqual(result["status"], "answered")
        self.assertIn("Height = 450 mm", result["answer"])
        self.assertEqual(model.invocations, [])

    def test_separately_supported_hypothesis_and_fit_are_both_kept(self):
        tool = evidence()
        tool["sources"].append({"id": "B2", "file": "synthetic.pdf", "page": 2,
                                 "excerpt": "Traditional recipes can be difficult to adjust when slurry conditions change."})
        generated = answer()
        generated["claims"].insert(0, {"kind": "hypothesis", "text": HYPOTHESIS_INTRO,
                                      "source_id": "B2", "quote": "recipes can be difficult to adjust when slurry conditions change", "fact_ids": []})
        assistant = Assistant(model=FakeModel([routed("product")], [generated]), tools={"product": lambda args: tool})
        result = assistant.turn("Our press loses performance when feed changes. What can help?")
        self.assertEqual(result["status"], "answered")
        self.assertIn(HYPOTHESIS_INTRO, result["answer"])
        self.assertIn("recipes can be difficult to adjust", result["answer"])
        self.assertIn("not an established cause", result["answer"])
        generation = next(row for row in result["trace"] if row["operation"] == "grounded_product_generation")
        self.assertEqual(len(generation["raw"]["claims"]), 2)
        self.assertNotIn("[B2]", result["answer"])

    def test_malformed_optional_index_returns_an_error_without_starting_inference(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index.json"
            index.write_text('{"model":null}', encoding="utf-8")
            model = FakeModel([routed("product")])
            result = Assistant(model=model, retrieval_mode="hybrid", semantic_index=index).turn("help")
        self.assertEqual(result["status"], "error")
        self.assertIn("embedding-model binding", result["error"])
        self.assertEqual(model.invocations, [])

    def test_ambiguous_quantity_explains_source_problem_without_exposing_parser_codes(self):
        tool = evidence()
        tool.update(status="insufficient_evidence", warnings=["configuration_boundary_ambiguity: internal diagnostic"])
        tool["data"]["unavailable_facts"] = [{"source_id": "B1", "parameter": "Cloth length", "configuration": "synthetic", "review_status": "ambiguous"}]
        model = FakeModel([routed("product")])
        result = Assistant(model=model, tools={"product": lambda args: tool}).turn("cloth length")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("does not clearly associate", result["answer"])
        self.assertNotIn("configuration_boundary_ambiguity", result["answer"])
        self.assertIn("Synthetic brochure, p. 2", result["answer"])
        self.assertEqual(result["sources"][0]["id"], "B1")
        self.assertEqual(model.invocations, [])

    def test_optional_retrieval_releases_models_before_switching_and_traces_it(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        events = []
        class ResourceModel(FakeModel):
            def route(self, history):
                events.append("route")
                return super().route(history)
            def unload(self):
                events.append("release_chat")
                return {"operation": "local_model_unload", "model": "synthetic:chat", "confirmed": True}
            def invoke(self, *args, **kwargs):
                events.append("generate")
                return super().invoke(*args, **kwargs)
        def retrieve(*args, **kwargs):
            events.append("retrieve")
            self.assertEqual(kwargs["retrieval_mode"], "hybrid")
            return evidence()
        def release(endpoint, tag):
            events.append("release_embed")
            self.assertEqual(tag, "synthetic:embedding")
            return {"operation": "local_model_unload", "model": tag, "confirmed": True}
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index.json"
            index.write_text(json.dumps({"model": {"tag": "synthetic:embedding"}}), encoding="utf-8")
            model = ResourceModel([routed("product")], [answer()])
            with patch("metso_assistant.product_tool.retrieve_product", side_effect=retrieve), patch("metso_assistant.agent.unload_local_model", side_effect=release):
                result = Assistant(model=model, retrieval_mode="hybrid", semantic_index=index).turn("help")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(events, ["route", "release_chat", "retrieve", "release_embed", "generate",
                                  "release_chat", "retrieve", "release_embed", "generate", "generate"])
        self.assertEqual(len([t for t in result["trace"] if t["operation"] == "local_model_unload"]), 4)

    def test_visual_trace_hash_is_for_the_exact_attached_bytes_read_once(self):
        # Independently invented one-pixel PNG, never a brochure/source image.
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=")
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp); image_path = raw / "invented.png"; image_path.write_bytes(png)
            tool = evidence()
            tool["data"].update(request_kind="visual", needs_visual_inspection=True,
                                visual_candidates=[{"source_id": "B2", "image_file": str(image_path)}])
            tool["sources"].append({"id": "B2", "file": "synthetic.pdf", "page": 2, "kind": "full_page", "excerpt": ""})
            generated = answer(recommendation=None)
            generated["claims"][0].update(kind="documented", text="The pictured tank connects to the controller.", source_id="B2", quote="Tank")
            model = FakeModel([routed("product", query="Explain the schematic")], [generated, {"complete": True, "grounded": True}],
                              plans=[request_plan("Explain the schematic", "visual")])
            original_read = Path.read_bytes
            with patch.object(Path, "read_bytes", autospec=True, side_effect=original_read) as reads:
                result = Assistant(model=model, tools={"product": lambda args: tool}, raw_dir=raw).turn("Explain the schematic")
            self.assertEqual(result["status"], "answered")
            self.assertEqual(reads.call_count, 1)
            attached = base64.b64decode(model.invocations[0][1][0]["images"][0])
            self.assertEqual(attached, png)
            self.assertNotIn("adjusts setpoints", model.invocations[0][1][0]["content"])
            self.assertEqual(model.invocations[0][2]["properties"]["recommendation"]["enum"], [None])
            trace = next(t for t in result["trace"] if t["operation"] == "grounded_product_generation")
            self.assertEqual(trace["visual_attachments"], [{"source_id": "B2", "relative_path": "invented.png",
                                                          "sha256": hashlib.sha256(attached).hexdigest(), "bytes": len(attached)}])
            self.assertEqual(trace["visual_source_ids"], ["B2"])

    def test_selected_product_context_excludes_other_machines(self):
        tool = evidence()
        tool["data"]["passages"] = [{"source_id": "B1", "doc_id": "example-optimizer", "text": tool["sources"][0]["excerpt"]},
                                     {"source_id": "B2", "doc_id": "example-pump", "text": "Example pump handles changing slurry characteristics."}]
        tool["sources"].append({"id": "B2", "file": "pump.pdf", "excerpt": "Example pump handles changing slurry characteristics."})
        model = FakeModel([routed("product")], [{"product_name": "Example optimizer"}, answer(recommendation="Example optimizer")])
        result = Assistant(model=model, tools={"product": lambda args: tool}).turn("Help with changing slurry characteristics")
        self.assertEqual(result["status"], "answered")
        self.assertIn("Example pump", model.invocations[0][1][0]["content"])
        self.assertNotIn("Example pump", model.invocations[1][1][0]["content"])

    def test_clarification_then_computation_and_topic_switch(self):
        calls = []
        def cost(args):
            calls.append(deepcopy(args))
            if args["cloths_supplied"] is None:
                return {"tool": "service_cost", "status": "clarify", "data": {}, "sources": [],
                        "question": "Are the cloths already supplied?", "missing_fields": ["cloths_supplied"], "warnings": []}
            return {"tool": "service_cost", "status": "ok", "data": {"summary": "Computed example budget: XCU 123.40", "amount": Decimal("123.40")},
                    "sources": [{"id": "S1", "file": "synthetic.txt"}], "warnings": [], "question": None, "missing_fields": []}
        model = FakeModel([routed("service_cost", country="Exampleland", equipment="Example Press", task="cloth replacement", cloth_count=7),
                           routed("service_cost", country="Exampleland", equipment="Example Press", task="cloth replacement", cloth_count=7, cloths_supplied=True, round_trip_travel_hours=0),
                           routed("product", query="Changing slurry", country="Exampleland", cloth_count=7)], [answer()])
        assistant = Assistant(model=model, tools={"service_cost": cost, "product": lambda args: evidence()})
        self.assertEqual(assistant.turn("Replace seven cloths")["status"], "clarify")
        computed = assistant.turn("Yes, and no travel")
        self.assertIn("123.40", computed["answer"])
        product = assistant.turn("What about changing slurry?")
        self.assertEqual(product["tool_arguments"], {"query": "What about changing slurry?"})
        self.assertEqual(product["status"], "answered")
        self.assertIs(calls[-1]["cloths_supplied"], True)
        self.assertEqual(calls[-1]["round_trip_travel_hours"], 0)
        self.assertIn("Are the cloths already supplied?", model.histories[1][-2]["content"])

    def test_invalid_generation_repaired_without_changed_evidence(self):
        invalid = answer(recommendation="Imaginary item")
        model = FakeModel([routed("product")], [invalid, answer()])
        result = Assistant(model=model, tools={"product": lambda args: evidence()}).turn("help")
        self.assertEqual(result["status"], "answered")
        generations = [x for x in result["trace"] if x["operation"] == "grounded_product_generation"]
        self.assertEqual(len(generations), 2)
        self.assertTrue(generations[0]["validation_errors"])

    def test_hypothesis_must_be_explicitly_conditional_before_it_is_rendered(self):
        invalid = answer()
        invalid["claims"][0].update(kind="hypothesis", text="The cause is slow adjustment.")
        model = FakeModel([routed("product")], [invalid, answer()])
        result = Assistant(model=model, tools={"product": lambda args: evidence()}).turn("help")
        self.assertEqual(result["status"], "answered")
        first = next(x for x in result["trace"] if x["operation"] == "grounded_product_generation")
        self.assertTrue(any("hypotheses" in x for x in first["validation_errors"]))
        generation = [row for row in result["trace"] if row["operation"] == "grounded_product_generation"][-1]
        self.assertEqual(len(generation["raw"]["claims"]), 1)

    def test_why_followup_keeps_customer_problem_and_does_not_recommend_again(self):
        queries = []
        def retrieve(arguments):
            queries.append(arguments["query"])
            tool = evidence()
            tool["sources"][0]["excerpt"] += " Traditional recipes can be difficult to adjust."
            return tool
        explanation = answer(recommendation=None)
        explanation["claims"][0].update(kind="hypothesis", text=HYPOTHESIS_INTRO, quote="Traditional recipes can be difficult to adjust.")
        initial = answer()
        initial["claims"].insert(0, deepcopy(explanation["claims"][0]))
        model = FakeModel([routed("product", query="Example press performance changes with feed"),
                           routed("product", query="why did performance drop though")], [initial, explanation], plans=[
                               request_plan("Our Example press loses performance when feed changes"),
                               request_plan("why did performance drop though", "explain", continuation=True)])
        assistant = Assistant(model=model, tools={"product": retrieve})
        assistant.turn("Our Example press loses performance when feed changes")
        second = assistant.turn("why did performance drop though")
        self.assertEqual(second["status"], "answered")
        self.assertEqual(second["response_intent"], "explain")
        self.assertIn("Example press", queries[-1])
        self.assertIn("Example optimizer", queries[-1])
        self.assertNotIn("Worth assessing", second["answer"])
        self.assertIn("not an established cause", second["answer"])
        self.assertNotIn("[B1]", second["answer"])
        self.assertEqual(second["sources"][0]["id"], "B1")

    def test_capability_question_repairs_unrequested_customer_cause_hypothesis(self):
        invalid = answer(recommendation=None)
        invalid["claims"][0].update(kind="hypothesis", text="One possibility is that the plant needs a new recipe.")
        valid = answer(recommendation=None)
        valid["claims"][0]["kind"] = "documented"
        model = FakeModel([routed("product")], [invalid, valid], plans=[
            request_plan("Explain how the Example optimizer helps changing slurry.", "explain")])
        result = Assistant(model=model, tools={"product": lambda args: evidence()}).turn("Explain how the Example optimizer helps changing slurry.")
        self.assertEqual(result["status"], "answered")
        first = next(x for x in result["trace"] if x["operation"] == "grounded_product_generation")
        self.assertTrue(any("No customer-cause hypothesis" in x for x in first["validation_errors"]))
        self.assertNotIn("established cause", result["answer"])

    def test_unavailable_numerical_answer_uses_customer_language_and_preserves_diagnostic(self):
        diagnostic = "No reviewed, reported and unblocked numerical fact answers the selected request. Qualitative passages cannot supply a numerical answer."
        model = FakeModel([routed("product")], [answer(uncertainties=[diagnostic])])
        result = Assistant(model=model, tools={"product": lambda args: evidence()}).turn("help without invented gains")
        self.assertIn("does not establish the requested numerical answer", result["answer"])
        self.assertNotIn("unblocked", result["answer"])
        generation = next(row for row in result["trace"] if row["operation"] == "grounded_product_generation")
        self.assertEqual(generation["raw"]["uncertainties"], [diagnostic])

    def test_failures_and_unsupported_requests_never_invent_answers(self):
        model = FakeModel([routed("unsupported"), routed("product")], [answer(recommendation="Bad"), answer(recommendation="Bad")])
        assistant = Assistant(model=model, tools={"product": lambda args: evidence()})
        self.assertEqual(assistant.turn("weather")["status"], "unsupported")
        self.assertEqual(assistant.turn("help")["status"], "error")
        assistant.reset()
        self.assertEqual(assistant.history, [])


if __name__ == "__main__":
    unittest.main()
