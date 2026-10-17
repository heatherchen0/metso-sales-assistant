from copy import deepcopy
import json
from pathlib import Path
import unittest

from metso_assistant.agent import Assistant
from metso_assistant.model import ModelError
from metso_assistant.product_request import named_detail_recovery, plan_product_request, scope_candidates, task_query
from tests.test_model import routed


def previous():
    return {"subject_query": "Our Atlas pressure filters lose performance when feed properties change.",
            "equipment": "Atlas pressure filters", "recommendation": "cycle controller",
            "discussed_subject": "cycle controller", "recommendations": ["cycle controller"]}


CATALOG = [{"name": name, "doc_id": name.replace(" ", "-"), "title": name}
           for name in ("cycle controller", "wash controller", "bubble column")]


def plan(current, tasks, *, status="continue_need", equipment=None):
    return {"context_status": status, "need_quote": current if status == "new_need" else None,
            "equipment_quote": equipment, "tasks": tasks}


def task(kind, quote, subject=None):
    return {"kind": kind, "question_quote": quote, "subject_product": subject}


class Model:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def invoke(self, system, messages, schema, **kwargs):
        from metso_assistant.agent import FIT_REVIEW_SYSTEM, TEXT_REVIEW_SYSTEM
        if system == FIT_REVIEW_SYSTEM:
            return {"quote_role": "product_capability", "relevant": True}, {"synthetic": True}
        if system == TEXT_REVIEW_SYSTEM:
            return {"substantive": True, "addresses_request": True}, {"synthetic": True}
        self.calls.append({"system": system, "messages": deepcopy(messages), "schema": deepcopy(schema)})
        return deepcopy(next(self.outputs)), {"synthetic": True}

    def route(self, history):
        return routed("product", query="stale routing guess"), []


def evidence():
    rows = [("cycle-controller", "S1", "Cycle controller adjusts pressure filter recipes when feed properties change."),
            ("wash-controller", "S2", "Wash controller adjusts pressure filter washing when slurry properties change."),
            ("bubble-column", "S3", "Bubble column offers flexible flotation spargers for concentrate grade.")]
    return {"tool": "product", "status": "ok", "data": {
        "passages": [{"doc_id": doc, "source_id": sid, "text": text} for doc, sid, text in rows], "facts": []},
        "sources": [{"id": sid, "file": doc + ".pdf", "page": 2, "excerpt": text} for doc, sid, text in rows],
        "warnings": [], "missing_fields": [], "question": None}


def response(kind, text, sid, quote, recommendation=None):
    return {"status": "answered", "recommendation": recommendation,
            "claims": [{"kind": kind, "text": text, "source_id": sid, "quote": quote, "fact_ids": []}],
            "uncertainties": []}


class ProductRequestTests(unittest.TestCase):
    @staticmethod
    def source_catalog():
        return [{"name": "Helix-X controller", "aliases": ["controller", "Helix"], "doc_id": "helix-x",
                 "status": "source_supported", "source_sha256": "a" * 64,
                 "identity_evidence": [{"doc_id": "helix-x", "page": 2, "source_sha256": "a" * 64,
                                        "quote": "Helix-X controller", "authority": "source_identity_only"}]}]

    def test_named_detail_recovery_uses_full_canonical_identity_and_exact_source_proof(self):
        catalog = self.source_catalog()
        for current in ("Describe Helix-X controller.", "How does Helix X controller work?",
                        "Show the Helix-X controller schematic.", "What specifications does HelixX controller have?"):
            with self.subTest(current=current):
                recovered = named_detail_recovery(current, catalog)
                self.assertEqual(recovered["matches"][0]["name"], "Helix-X controller")
                self.assertIn(recovered["matches"][0]["literal_quote"], current)
                self.assertEqual(recovered["matches"][0]["identity_evidence"], catalog[0]["identity_evidence"])
                self.assertEqual(recovered["authority"], "source_identity_only")
        catalog[0]["identity_evidence"][0]["source_sha256"] = "b" * 64
        self.assertIsNone(named_detail_recovery("Describe Helix-X controller.", catalog))

    def test_named_detail_recovery_does_not_promote_aliases_unknown_names_or_non_detail_requests(self):
        for current in ("Describe this controller.", "How does Helix work?", "Describe Unknown controller.",
                        "Describe SuperHelix-X controller.", "We have a Helix-X controller.",
                        "Recommend Helix-X controller.", "Describe Helix-X controller and suggest other options.",
                        "What is the replacement cost for Helix-X controller?",
                        "Explain labour and travel for a Helix-X controller service visit."):
            with self.subTest(current=current):
                self.assertIsNone(named_detail_recovery(current, self.source_catalog()))

    def test_named_detail_recovery_requires_source_identity_not_catalogue_spelling_alone(self):
        catalog = self.source_catalog()
        for change in ({"status": "unresolved"}, {"identity_evidence": []}, {"source_sha256": None}):
            with self.subTest(change=change):
                altered = [{**catalog[0], **change}]
                self.assertIsNone(named_detail_recovery("Describe Helix-X controller.", altered))

    def test_initial_problem_exposes_no_unnamed_product_subject_or_unrequested_alternative(self):
        current = "Our Atlas pressure filters lose performance when feed changes."
        wanted = plan(current, [task("recommend", current)], status="new_need", equipment="Atlas pressure filters")
        model = Model([wanted])
        _, context, _ = plan_product_request(model, [current], {}, CATALOG, equipment_candidates=["Atlas pressure filters", "guessed variant"])
        schema = model.calls[0]["schema"]
        properties = schema["properties"]["tasks"]["items"]["properties"]
        self.assertEqual(properties["subject_product"]["enum"], [None])
        self.assertNotIn("alternatives", properties["kind"]["enum"])
        self.assertEqual(schema["properties"]["need_quote"]["enum"], [None, current])
        self.assertEqual(schema["properties"]["equipment_quote"]["enum"], [None, "Atlas pressure filters"])
        self.assertEqual(json.loads(model.calls[0]["messages"][0]["content"])["catalogue"], [])
        self.assertEqual(context["equipment"], "Atlas pressure filters")

    def test_compound_plan_preserves_need_and_builds_different_task_queries(self):
        current = "Explain that controller, and suggest additional options."
        wanted = plan(current, [task("describe", "Explain that controller", "cycle controller"),
                                task("alternatives", "suggest additional options")])
        actual, context, trace = plan_product_request(Model([wanted]), [current], previous(), CATALOG)
        self.assertEqual(actual, wanted)
        self.assertEqual(context["subject_query"], previous()["subject_query"])
        description, alternatives = [task_query(item, context) for item in actual["tasks"]]
        self.assertIn("cycle controller", description)
        self.assertNotIn("Atlas", description)
        self.assertIn("Atlas pressure filters", alternatives)
        self.assertIn("feed properties change", alternatives)
        self.assertFalse(trace[0]["validation_errors"])

    def test_named_other_product_changes_discussion_without_recommending_it(self):
        current = "Tell me more about bubble column."
        wanted = plan(current, [task("describe", current, "bubble column")], status="subject_only")
        _, context, _ = plan_product_request(Model([wanted]), [current], previous(), CATALOG, equipment_candidates=["bubble column"])
        self.assertEqual(context["discussed_subject"], "bubble column")
        self.assertEqual(context["recommendation"], "cycle controller")
        self.assertEqual(context["subject_query"], previous()["subject_query"])
        self.assertEqual(context["equipment"], previous()["equipment"])
        self.assertNotIn("Atlas", task_query(wanted["tasks"][0], context))

    def test_new_customer_need_retires_old_recommendations(self):
        current = "Our lift pump cavitates at low flow."
        wanted = plan(current, [task("recommend", current)], status="new_need", equipment="lift pump")
        _, context, _ = plan_product_request(Model([wanted]), [current], previous(), CATALOG, equipment_candidates=["lift pump"])
        self.assertEqual(context["subject_query"], current)
        self.assertEqual(context["equipment"], "lift pump")
        self.assertIsNone(context["recommendation"])
        self.assertEqual(context["recommendations"], [])
        self.assertEqual(context["need_provenance"]["quote"], current)

    def test_fresh_request_payload_excludes_cross_domain_history_and_retired_product_state(self):
        current = "Our vessel loses throughput whenever the feed changes."
        state = previous()
        state["followup"] = False
        messages = ["Budget for a repair in Exampleland.", "Compare the new assay readings.", current]
        model = Model([plan(current, [task("recommend", current)], status="new_need")])
        _, context, _ = plan_product_request(model, messages, state, CATALOG)
        payload = json.loads(model.calls[0]["messages"][0]["content"])
        self.assertEqual(payload["actual_user_messages"], [current])
        self.assertEqual(payload["previous_product_context"], {
            "subject_query": current, "equipment": None, "followup": False})
        self.assertNotIn("Atlas", model.calls[0]["messages"][0]["content"])
        self.assertNotIn("cycle controller", model.calls[0]["messages"][0]["content"])
        self.assertEqual(context["subject_query"], current)

    def test_implicit_problem_cannot_be_replaced_with_named_equipment_description(self):
        current = "Our Kappa pressure filters lose throughput when the feed changes."
        catalog = [{"doc_id": name.replace(" ", "-"), "name": name, "aliases": [name, "Kappa"]}
                   for name in ("Kappa P", "Kappa Q", "Kappa R")]
        state = {"subject_query": current, "equipment": None, "followup": False}
        invalid = plan(current, [task("describe", current, "Kappa Q")], status="subject_only")
        valid = plan(current, [task("recommend", current)], status="new_need")
        model = Model([invalid, valid])
        actual, context, trace = plan_product_request(model, [current], state, catalog)
        self.assertEqual(actual, valid)
        self.assertEqual(context["subject_query"], current)
        self.assertTrue(any("Uncovered requested operation: recommend" in error
                            for error in trace[0]["validation_errors"]))
        coverage = json.loads(model.calls[0]["messages"][0]["content"])["requested_operation_coverage"]
        self.assertEqual(coverage, [{"question_quote": current, "allowed_kinds": ["recommend"]}])

    def test_initial_recommendation_cannot_preselect_a_named_equipment_variant(self):
        current = "Our Kappa line loses throughput during feed changes."
        catalog = [{"doc_id": "kappa-q", "name": "Kappa Q", "aliases": ["Kappa Q", "Kappa"]}]
        invalid = plan(current, [task("recommend", current, "Kappa Q")], status="new_need")
        valid = plan(current, [task("recommend", current)], status="new_need")
        actual, _, trace = plan_product_request(Model([invalid, valid]), [current], {}, catalog)
        self.assertIsNone(actual["tasks"][0]["subject_product"])
        self.assertTrue(any("must not preselect" in error for error in trace[0]["validation_errors"]))

    def test_soft_followup_hint_does_not_forbid_new_problem_without_equipment_guess(self):
        current = "Our mixing bank oscillates whenever upstream conditions change."
        state = previous()
        state["followup"] = True
        wanted = plan(current, [task("recommend", current)], status="new_need")
        model = Model([wanted])
        _, context, trace = plan_product_request(model, [state["subject_query"], current], state, CATALOG)
        self.assertIn("new_need", model.calls[0]["schema"]["properties"]["context_status"]["enum"])
        self.assertEqual(context["subject_query"], current)
        self.assertIsNone(context["equipment"])
        self.assertIsNone(context["recommendation"])
        self.assertEqual(trace[0]["validation_errors"], [])

    def test_local_pronoun_inside_owned_problem_does_not_lock_old_need(self):
        for current in ("Our vessel loses pressure and its alarm keeps tripping.",
                        "What could help our centrifuge that trips under load?"):
            with self.subTest(current=current):
                state = previous()
                state["followup"] = True
                wanted = plan(current, [task("recommend", current)], status="new_need")
                _, context, _ = plan_product_request(Model([wanted]), [current], state, CATALOG)
                self.assertEqual(context["subject_query"], current)
                self.assertEqual(context["recommendations"], [])

    def test_continuation_history_starts_at_the_verified_product_need(self):
        state = previous()
        state["followup"] = True
        current = "Why might that happen?"
        relevant = [state["subject_query"], "Tell me more about that controller.", current]
        model = Model([plan(current, [task("explain", current, "cycle controller")])])
        _, context, _ = plan_product_request(model,
            ["Budget the unrelated repair.", "Assess the assay readings.", *relevant], state, CATALOG)
        payload = json.loads(model.calls[0]["messages"][0]["content"])
        self.assertEqual(payload["actual_user_messages"], relevant)
        self.assertEqual(context["subject_query"], state["subject_query"])

    def test_missing_history_anchor_does_not_pass_unrelated_messages_to_followup_planner(self):
        state = previous()
        state["followup"] = True
        current = "What specifications does it have?"
        model = Model([plan(current, [task("specification", current, "cycle controller")])])
        _, context, _ = plan_product_request(model, ["Budget the unrelated repair.", current], state, CATALOG)
        payload = json.loads(model.calls[0]["messages"][0]["content"])
        self.assertEqual(payload["actual_user_messages"], [current])
        self.assertEqual(payload["previous_product_context"]["recommendation"], "cycle controller")
        self.assertEqual(context["subject_query"], state["subject_query"])

    def test_bare_source_alias_remains_a_description_in_a_fresh_conversation(self):
        current = "Helix"
        catalog = [{"doc_id": "helix", "name": "Helix upgrade", "aliases": ["Helix upgrade", "Helix"]}]
        wanted = plan(current, [task("describe", current, "Helix upgrade")], status="subject_only")
        model = Model([wanted])
        actual, context, _ = plan_product_request(model, [current], {}, catalog)
        self.assertEqual(actual["tasks"][0]["kind"], "describe")
        self.assertEqual(context["discussed_subject"], "Helix upgrade")
        self.assertEqual(json.loads(model.calls[0]["messages"][0]["content"])["requested_operation_coverage"], [])

    def test_named_capability_questions_do_not_default_to_recommendation(self):
        catalog = [{"doc_id": "helix", "name": "Helix upgrade", "aliases": ["Helix upgrade", "Helix"]}]
        for current in ("How does the Helix upgrade work?", "How can Helix help with unstable feed?",
                        "How does the Helix upgrade control the cycle?"):
            with self.subTest(current=current):
                wanted = plan(current, [task("explain", current, "Helix upgrade")], status="subject_only")
                model = Model([wanted])
                actual, context, trace = plan_product_request(model, [current], {}, catalog)
                self.assertEqual(actual["tasks"][0]["kind"], "explain")
                self.assertEqual(context["discussed_subject"], "Helix upgrade")
                self.assertEqual(trace[0]["validation_errors"], [])
                coverage = json.loads(model.calls[0]["messages"][0]["content"])["requested_operation_coverage"]
                self.assertEqual(coverage[0]["allowed_kinds"], ["describe", "explain"])

    def test_first_person_task_only_alternatives_cannot_retire_the_active_need(self):
        for current in ("Can we have other recommendations?", "We would like additional options."):
            with self.subTest(current=current):
                state = previous()
                state["followup"] = True
                invalid = plan(current, [task("alternatives", current)], status="new_need")
                valid = plan(current, [task("alternatives", current)])
                model = Model([invalid, valid])
                _, context, trace = plan_product_request(model, [current], state, CATALOG)
                self.assertNotIn("new_need", model.calls[0]["schema"]["properties"]["context_status"]["enum"])
                self.assertTrue(trace[0]["validation_errors"])
                self.assertEqual(context["subject_query"], state["subject_query"])

    def test_first_person_request_with_a_new_owned_condition_can_still_establish_a_need(self):
        current = "Can we have another option for our dryer that overheats under load?"
        state = previous()
        state["followup"] = True
        wanted = plan(current, [task("alternatives", current)], status="new_need")
        model = Model([wanted])
        _, context, trace = plan_product_request(model, [current], state, CATALOG)
        self.assertIn("new_need", model.calls[0]["schema"]["properties"]["context_status"]["enum"])
        self.assertEqual(context["subject_query"], current)
        self.assertEqual(context["recommendations"], [])
        self.assertEqual(trace[0]["validation_errors"], [])

    def test_authoritative_new_boundary_cannot_reuse_old_need_even_for_short_request(self):
        current = "Help our vessel recover fines."
        state = previous()
        state["followup"] = False
        invalid = plan(current, [task("recommend", current)], status="subject_only")
        valid = plan(current, [task("recommend", current)], status="new_need")
        model = Model([invalid, valid])
        _, context, trace = plan_product_request(model, [current], state, CATALOG)
        self.assertNotIn("continue_need", model.calls[0]["schema"]["properties"]["context_status"]["enum"])
        self.assertTrue(any("retired application" in error for error in trace[0]["validation_errors"]))
        self.assertEqual(context["subject_query"], current)
        self.assertIsNone(context["recommendation"])
        self.assertNotIn("cycle controller", model.calls[0]["schema"]["properties"]["tasks"]["items"]["properties"]["subject_product"]["enum"])

    def test_reconciled_fresh_context_accepts_terse_current_need(self):
        current = "Help"
        state = {"subject_query": current, "equipment": None, "followup": False}
        wanted = plan(current, [task("recommend", current)], status="new_need")
        _, context, trace = plan_product_request(Model([wanted]), [current], state, CATALOG)
        self.assertEqual(context["subject_query"], current)
        self.assertEqual(trace[0]["validation_errors"], [])

    def test_verified_substantive_followup_cannot_retire_active_need(self):
        current = "Explain how that controller responds to fluctuating slurry characteristics."
        state = previous()
        state["followup"] = True
        wanted = plan(current, [task("explain", current, "cycle controller")])
        model = Model([wanted])
        _, context, _ = plan_product_request(model, [current], state, CATALOG)
        self.assertEqual(context["subject_query"], previous()["subject_query"])
        self.assertNotIn("new_need", model.calls[0]["schema"]["properties"]["context_status"]["enum"])

    def test_missing_half_of_numeric_and_alternative_request_is_repaired(self):
        current = "What is the controller capacity, and what other recommendations do you have?"
        incomplete = plan(current, [task("specification", current, "cycle controller")])
        complete = plan(current, [task("specification", "What is the controller capacity", "cycle controller"),
                                  task("alternatives", "what other recommendations do you have")])
        model = Model([incomplete, complete])
        actual, context, trace = plan_product_request(model, [current], previous(), CATALOG)
        self.assertEqual(len(actual["tasks"]), 2)
        self.assertTrue(any("Uncovered requested operation: alternatives" in error for error in trace[0]["validation_errors"]))
        self.assertEqual(context["subject_query"], previous()["subject_query"])
        self.assertEqual(model.calls[0]["schema"]["properties"]["tasks"]["minItems"], 2)

    def test_description_and_visual_are_separate_tasks_without_flow_rate_confusion(self):
        current = "Describe bubble column and explain its flow diagram."
        incomplete = plan(current, [task("describe", current, "bubble column")], status="subject_only")
        complete = plan(current, [task("describe", "Describe bubble column", "bubble column"),
                                  task("visual", "explain its flow diagram", "bubble column")], status="subject_only")
        actual, _, trace = plan_product_request(Model([incomplete, complete]), [current], previous(), CATALOG)
        self.assertEqual([item["kind"] for item in actual["tasks"]], ["describe", "visual"])
        self.assertTrue(any("Uncovered requested operation: visual" in error for error in trace[0]["validation_errors"]))

    def test_comparison_keeps_both_literal_product_names_without_recommending_either(self):
        current = "Compare cycle controller and wash controller."
        wanted = plan(current, [task("compare", current)], status="subject_only")
        actual, context, trace = plan_product_request(Model([wanted]), [current], previous(), CATALOG)
        query = task_query(actual["tasks"][0], context)
        self.assertIn("Compare cycle controller and wash controller", query)
        self.assertFalse(trace[0]["validation_errors"])

    def test_source_alias_authorizes_canonical_subject_without_inventing_a_name(self):
        current = "Describe Helix."
        catalog = [{"doc_id": "helix-upgrade", "name": "Helix\u00ae upgrade", "aliases": ["Helix\u00ae upgrade", "Helix"]}]
        wanted = plan(current, [task("describe", current, "Helix\u00ae upgrade")], status="subject_only")
        actual, context, trace = plan_product_request(Model([wanted]), [current], previous(), catalog)
        self.assertEqual(actual["tasks"][0]["subject_product"], "Helix\u00ae upgrade")
        self.assertEqual(context["discussed_subject"], "Helix\u00ae upgrade")
        self.assertFalse(trace[0]["validation_errors"])

    def test_equipment_suffix_does_not_change_which_source_candidates_survive(self):
        first = previous()
        first.update(equipment="ExistingVessel", subject_query="Recover fines below 17 microns in ExistingVessel")
        second = deepcopy(first)
        second["equipment"] = "ExistingVessel circuit"
        one, _ = scope_candidates(evidence(), first)
        two, _ = scope_candidates(evidence(), second)
        self.assertEqual(one["sources"], two["sources"])
        self.assertEqual(len(one["data"]["passages"]), 3)

    def test_invented_variant_and_invented_task_quote_are_repaired(self):
        current = "What does it do?"
        invalid = plan(current, [task("describe", "Does it measure feed grade?", "cycle controller")], equipment="Atlas PF-999")
        valid = plan(current, [task("describe", current, "cycle controller")])
        model = Model([invalid, valid])
        _, context, trace = plan_product_request(model, [current], previous(), CATALOG)
        self.assertEqual(context["equipment"], "Atlas pressure filters")
        self.assertTrue(trace[0]["validation_errors"])
        self.assertEqual(trace[1]["validation_errors"], [])
        self.assertIn("actual user spans", model.calls[1]["messages"][-1]["content"])

    def test_short_alternative_request_cannot_retire_customer_need_even_if_model_calls_it_new(self):
        current = "Any additional recommendations?"
        invalid = plan(current, [task("alternatives", current)], status="new_need")
        valid = plan(current, [task("alternatives", current)])
        _, context, trace = plan_product_request(Model([invalid, valid]), [current], previous(), CATALOG)
        self.assertEqual(context["subject_query"], previous()["subject_query"])
        self.assertTrue(any("retains the active customer need" in error for error in trace[0]["validation_errors"]))

    def test_followup_schema_cannot_erase_need_with_task_only_or_named_description(self):
        for current, tasks in [
                ("Describe that controller and give other recommendations.", [task("describe", "Describe that controller", "cycle controller"), task("alternatives", "give other recommendations")]),
                ("Tell me more about bubble column.", [task("describe", "Tell me more about bubble column.", "bubble column")])]:
            model = Model([plan(current, tasks)])
            plan_product_request(model, [current], previous(), CATALOG)
            schema = model.calls[0]["schema"]
            self.assertNotIn("new_need", schema["properties"]["context_status"]["enum"])
            self.assertEqual(schema["properties"]["need_quote"]["enum"], [None])

    def test_event_cause_anaphor_uses_active_symptom_even_if_planner_names_candidate(self):
        current = "Why might that occur?"
        wanted = plan(current, [task("explain", current, "cycle controller")], status="subject_only")
        model = Model([wanted])
        actual, context, _ = plan_product_request(model, [current], previous(), CATALOG)
        query = task_query(actual["tasks"][0], context)
        self.assertIn("Atlas pressure filters lose performance", query)
        self.assertIn("cycle controller", query)
        self.assertNotIn("new_need", model.calls[0]["schema"]["properties"]["context_status"]["enum"])
        capability = task_query(task("explain", "How does it work?", "cycle controller"), context)
        self.assertNotIn("Atlas", capability)

    def test_cause_followup_with_auxiliary_words_cannot_start_a_new_need(self):
        current = "Why did performance drop though?"
        wanted = plan(current, [task("explain", current, "cycle controller")])
        model = Model([wanted])
        plan_product_request(model, [current], previous(), CATALOG)
        self.assertNotIn("new_need", model.calls[0]["schema"]["properties"]["context_status"]["enum"])

    def test_unverified_subject_or_unavailable_prior_need_fails_closed(self):
        current = "Any other options?"
        invalid = plan(current, [task("alternatives", current, "invented device")])
        with self.assertRaises(ModelError) as raised:
            plan_product_request(Model([invalid, invalid]), [current], {}, CATALOG)
        self.assertEqual(len(raised.exception.traces), 2)

    def test_alternative_scope_excludes_previous_identity_without_pretending_lexical_fit_is_semantic_proof(self):
        scoped, trace = scope_candidates(evidence(), previous(), alternatives=True)
        self.assertEqual([item["doc_id"] for item in scoped["data"]["passages"]], ["wash-controller", "bubble-column"])
        self.assertEqual([source["id"] for source in scoped["sources"]], ["S2", "S3"])
        rejected = {item["doc_id"]: item for item in trace["decisions"] if not item["accepted"]}
        self.assertIn("previously recommended", rejected["cycle-controller"]["reason"])
        self.assertTrue(all(item["application_fit"] == "not_decided" for item in trace["decisions"]))

    def test_generic_need_words_and_valid_citation_do_not_establish_fit(self):
        source = evidence()
        source["data"]["passages"] = source["data"]["passages"][-1:]
        source["sources"] = source["sources"][-1:]
        scoped, trace = scope_candidates(source, previous(), alternatives=True)
        self.assertEqual(scoped["sources"], source["sources"])
        self.assertEqual(trace["decisions"][0]["application_fit"], "not_decided")

    def test_shared_equipment_adjective_cannot_authorize_a_different_process(self):
        source = evidence()
        source["sources"][-1]["excerpt"] += " The flotation sparger uses pressure control."
        source["data"]["passages"] = source["data"]["passages"][-1:]
        source["sources"] = source["sources"][-1:]
        scoped, trace = scope_candidates(source, previous(), alternatives=True)
        self.assertNotIn("equipment_head", trace)
        self.assertEqual(scoped["sources"], source["sources"])
        self.assertEqual(trace["decisions"][0]["application_fit"], "not_decided")

    def test_equipment_head_without_documented_customer_need_remains_an_undecided_candidate(self):
        source = evidence()
        source["data"]["passages"].append({"doc_id": "cloth-press", "source_id": "S4", "text": "Cloth press is a pressure filter with a robust frame."})
        source["sources"].append({"id": "S4", "file": "cloth-press.pdf", "page": 1,
                                  "excerpt": "Cloth press is a pressure filter with a robust frame."})
        scoped, trace = scope_candidates(source, previous(), alternatives=True)
        self.assertEqual([row["doc_id"] for row in scoped["data"]["passages"]], ["wash-controller", "bubble-column", "cloth-press"])
        unrelated = next(row for row in trace["decisions"] if row["doc_id"] == "cloth-press")
        self.assertTrue(unrelated["accepted"])
        self.assertEqual(unrelated["application_fit"], "not_decided")
        self.assertEqual(unrelated["matched_non_equipment_need_groups"], [])

    def test_source_synonyms_can_support_need_without_copying_exact_customer_wording(self):
        source = evidence()
        source["sources"][1]["excerpt"] = "Wash controller adjusts pressure filter operation when slurry characteristics vary."
        source["data"]["passages"][1]["text"] = source["sources"][1]["excerpt"]
        scoped, trace = scope_candidates(source, previous(), alternatives=True)
        self.assertIn("wash-controller", [row["doc_id"] for row in scoped["data"]["passages"]])
        accepted = next(row for row in trace["decisions"] if row["doc_id"] == "wash-controller")
        self.assertGreaterEqual(len(accepted["matched_non_equipment_need_groups"]), 2)

    def test_differently_qualified_grade_label_is_not_support_for_changing_feed_grade(self):
        source = evidence()
        source["data"]["passages"] = [{"doc_id": "clean-filter", "source_id": "S4", "text": "Clean filter has medical-grade steel and automatic cycles for varying conditions."}]
        source["sources"] = [{"id": "S4", "excerpt": source["data"]["passages"][0]["text"]}]
        context = previous()
        context["subject_query"] = "Our pressure filters lose performance when feed grade changes."
        scoped, trace = scope_candidates(source, context, alternatives=True)
        self.assertEqual(scoped["sources"], source["sources"])
        self.assertEqual(len(trace["decisions"][0]["matched_non_equipment_need_groups"]), 1)
        self.assertEqual(trace["decisions"][0]["application_fit"], "not_decided")

    def test_compound_agent_executes_both_tasks_and_alternative_uses_active_need(self):
        current = "Describe this controller and give other recommendations."
        requested = plan(current, [task("describe", "Describe this controller", "cycle controller"),
                                   task("alternatives", "give other recommendations")])
        outputs = [requested,
                   response("documented", "Cycle controller adjusts pressure filter recipes when feed properties change.", "S1",
                            "Cycle controller adjusts pressure filter recipes when feed properties change."),
                   {"product_name": "Wash controller"},
                   response("fit", "Wash controller adjusts pressure filter washing when slurry properties change.", "S2",
                            "Wash controller adjusts pressure filter washing when slurry properties change.", "Wash controller")]
        calls = []
        def retrieve(arguments):
            calls.append(arguments["query"])
            return evidence()
        model = Model(outputs)
        assistant = Assistant(model=model, tools={"product": retrieve}, store_path=Path("missing-synthetic.sqlite"))
        assistant.context = {"last_route": "product", "product": previous()}
        result = assistant.turn(current)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["response_intent"], "compound")
        self.assertIn("Cycle controller", result["answer"])
        self.assertIn("Other options:", result["answer"])
        self.assertIn("Wash controller", result["answer"])
        self.assertNotIn("Bubble column", result["answer"])
        self.assertIn("Atlas pressure filters", calls[-1])
        tasks = result["tool_result"]["data"]["task_results"]
        self.assertEqual([row["status"] for row in tasks], ["answered", "answered"])
        self.assertTrue(all(row["executed"] for row in tasks))
        self.assertEqual(result["conversation_context"]["product"]["subject_query"], previous()["subject_query"])
        selection = next(call for call in model.calls if "candidates" in json.loads(call["messages"][0]["content"]))
        self.assertIn("Bubble column", selection["messages"][0]["content"])
        self.assertNotIn("Cycle controller", selection["messages"][0]["content"])

    def test_no_supported_alternative_is_explicit_and_does_not_repeat_recommendation(self):
        current = "Any other recommendations?"
        wanted = plan(current, [task("alternatives", current)])
        tool = evidence()
        tool["data"]["passages"] = [row for row in tool["data"]["passages"] if row["doc_id"] != "wash-controller"]
        tool["sources"] = [row for row in tool["sources"] if row["id"] != "S2"]
        model = Model([wanted, {"product_name": None}])
        assistant = Assistant(model=model, tools={"product": lambda args: tool}, store_path=Path("missing-synthetic.sqlite"))
        assistant.context = {"last_route": "product", "product": previous()}
        result = assistant.turn(current)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("did not find a source-supported product recommendation", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result["conversation_context"]["product"]["recommendation"], "cycle controller")


if __name__ == "__main__":
    unittest.main()
