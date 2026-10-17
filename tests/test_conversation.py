from copy import deepcopy
import unittest

from metso_assistant.conversation import reconcile_route, update_context, _travel_value
from metso_assistant.routing import FIELD_TYPES, validate_route


def routed(route, query="model summary", **fields):
    return {"route": route, "query": query,
            "known_fields": {name: fields.get(name) for name in FIELD_TYPES}}


def messages(user, assistant=None):
    return ([{"role": "assistant", "content": assistant}] if assistant else []) + [
        {"role": "user", "content": user}]


def product_context():
    question = "Our Example pressure filters lose performance when the feed changes."
    route, context, _ = reconcile_route(
        routed("product", equipment="Example pressure filters"), messages(question))
    return update_context(context, route, recommendation="Example optimizer")


def service_context(*, pending=("round_trip_travel_hours",), question="What are the expected round-trip travel hours?"):
    message = "Budget a visit in Exampleland to replace 7 cloths on an Example press. Customer already has the cloths."
    route, context, _ = reconcile_route(
        routed("service_cost", country="Exampleland", equipment="Example press",
               task="cloth replacement", cloth_count=7, cloths_supplied=True), messages(message))
    return update_context(context, route, {"missing_fields": list(pending), "question": question})


class ProductConversationTests(unittest.TestCase):
    def test_contextual_new_label_does_not_erase_verified_reference_only_product_need(self):
        for question in ("Why could that happen?", "Do you have other recommendations?"):
            with self.subTest(question=question):
                original = product_context()
                _, state, trace = reconcile_route(routed("product"), messages(question), original,
                    interpretation={"route": "product", "relation": "new_request", "context_requirement": "self_contained",
                                    "initial_context_requirement": "needs_context"})
                self.assertEqual(state["product"]["subject_query"], original["product"]["subject_query"])
                self.assertTrue(state["product"]["followup"])
                self.assertFalse(any(item.get("operation") == "current_request_boundary" for item in trace))

    def test_contextual_new_label_does_not_erase_proven_service_addition(self):
        question = "Also include 6 locking clips. Total billed travel will be 8 hours."
        history = [{"role": "user", "content": "Budget a visit in Exampleland to replace 7 cloths on an Example press."},
                   {"role": "user", "content": question}]
        route, state, trace = reconcile_route(routed("service_cost"), history, service_context(),
            interpretation={"route": "service_cost", "relation": "new_request", "context_requirement": "self_contained",
                            "initial_context_requirement": "needs_context"})
        self.assertEqual(state["service"]["fields"]["cloth_count"], 7)
        self.assertEqual(state["service"]["fields"]["country"], "Exampleland")
        self.assertEqual(state["service"]["fields"]["round_trip_travel_hours"], 8)
        self.assertFalse(any(item.get("operation") == "current_request_boundary" for item in trace))

    def test_hours_in_independent_equipment_problem_do_not_answer_old_service_question(self):
        question = "Our reactor loses pressure after 2 hours. What product could help?"
        for initial in ("self_contained", "needs_context"):
            with self.subTest(initial=initial):
                route, state, trace = reconcile_route(routed("product", equipment="reactor"), messages(question), service_context(),
                    interpretation={"route": "product", "relation": "new_request", "context_requirement": "self_contained",
                                    "initial_context_requirement": initial})
                self.assertEqual(route["route"], "product")
                self.assertEqual(state["product"]["subject_query"], question)
                self.assertNotIn("service", state)
                self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])
                self.assertTrue(any(item.get("operation") == "current_request_boundary" for item in trace))

    def test_local_reference_in_self_contained_question_preserves_new_boundary(self):
        question = "Can you suggest something for our fluctuating vessel level? Our current control cannot keep up with it."
        _, state, trace = reconcile_route(routed("product"), messages(question), product_context(),
            interpretation={"route": "product", "relation": "new_request", "context_requirement": "self_contained"})
        self.assertFalse(state["product"]["followup"])
        self.assertEqual(state["product"]["subject_query"], question)
        self.assertIsNone(state["product"]["recommendation"])
        self.assertTrue(any(item.get("operation") == "current_request_boundary" for item in trace))

    def test_alternative_request_retains_the_customer_need_despite_new_label(self):
        original = product_context()
        route, state, trace = reconcile_route(routed("clarify"), messages("Do you have other recommendations?"), original,
            interpretation={"route": "clarify", "relation": "new_request"})
        self.assertEqual(route["route"], "product")
        self.assertEqual(state["product"]["subject_query"], original["product"]["subject_query"])
        self.assertFalse(any(item.get("operation") == "current_request_boundary" for item in trace))

    def test_local_pronouns_in_new_customer_problem_do_not_restore_old_need(self):
        question = "Now consider a different plant: its flotation cells have unstable pulp levels. Recommend something for that problem."
        route, state, trace = reconcile_route(routed("product"), messages(question), product_context(),
            interpretation={"route": "product", "relation": "new_request"})
        self.assertFalse(state["product"]["followup"])
        self.assertEqual(state["product"]["subject_query"], question)
        self.assertIsNone(state["product"]["recommendation"])
        self.assertTrue(any(item.get("operation") == "current_request_boundary" for item in trace))

    def test_explicit_request_for_reason_and_recommendation_can_answer_both(self):
        _, context, _ = reconcile_route(routed("product"), messages("Why does performance drop, and what product do you recommend?"))
        self.assertEqual(context["product"]["intent"], "recommend")
        _, context, _ = reconcile_route(routed("product"), messages("Why did you recommend that?"), product_context())
        self.assertEqual(context["product"]["intent"], "explain")

    def test_generic_specs_allow_features_but_specific_ratings_remain_numerical(self):
        generic, state, _ = reconcile_route(routed("product", "Specs"), messages("What specifications does it have?"), product_context())
        self.assertTrue(state["product"]["broad_specification"])
        self.assertIn("Example optimizer", generic["query"])
        self.assertNotIn("pressure filters", generic["query"])
        specific, state, _ = reconcile_route(routed("product", "Rating"), messages("What is its rated voltage?"), product_context())
        self.assertFalse(state["product"]["broad_specification"])
        self.assertNotIn("help describe", specific["query"])

    def test_why_followup_cannot_replace_customer_subject_with_router_guess(self):
        context = product_context()
        route, context, trace = reconcile_route(
            routed("product", "Why do AC filters lose performance?", equipment="AC filter"),
            messages("Why did performance drop though?"), context)
        self.assertIn("Example pressure filters", route["query"])
        self.assertIn("feed changes", route["query"])
        self.assertIn("Example optimizer", route["query"])
        self.assertNotIn("AC filter", route["query"])
        self.assertEqual(context["product"]["intent"], "explain")
        self.assertEqual(route["known_fields"]["equipment"], "Example pressure filters")
        self.assertTrue(trace)
        self.assertEqual(validate_route(route), [])

    def test_specs_of_it_targets_validated_candidate(self):
        route, context, _ = reconcile_route(
            routed("product", "Specifications of pressure filters", equipment="Example pressure filters"),
            messages("What specifications does it have?"), product_context())
        self.assertEqual(route["known_fields"]["equipment"], "Example optimizer")
        self.assertIn("Previously discussed candidate product: Example optimizer", route["query"])
        self.assertNotIn("Example pressure filters", route["query"])
        self.assertEqual(context["product"]["intent"], "specification")

    def test_new_named_equipment_does_not_inherit_old_symptom(self):
        route, context, _ = reconcile_route(
            routed("product", "How does Other pump work?", equipment="Other pump"),
            messages("How does Other pump work?"), product_context())
        self.assertEqual(route["query"], "How does Other pump work?")
        self.assertNotIn("feed changes", route["query"])
        self.assertFalse(context["product"]["followup"])
        self.assertIsNone(context["product"]["recommendation"])

    def test_explicit_topic_switch_prevents_pronoun_reuse(self):
        route, context, _ = reconcile_route(
            routed("product", "New equipment: why is this pump vibrating?", equipment="Other pump"),
            messages("A different machine: why is this pump vibrating?"), product_context())
        self.assertNotIn("Example optimizer", route["query"])
        self.assertEqual(context["product"]["intent"], "explain")

    def test_question_intent_does_not_require_an_existing_recommendation(self):
        _, context, _ = reconcile_route(routed("product"), messages("Why does this filter lose performance?"))
        self.assertEqual(context["product"]["intent"], "explain")
        self.assertIsNone(context["product"]["recommendation"])

    def test_only_caller_validated_recommendation_enters_memory(self):
        route, context, _ = reconcile_route(routed("product", "Example miracle recommendation"), messages("Help with a filter."))
        self.assertIsNone(context["product"]["recommendation"])
        self.assertIsNone(update_context(context, route, {"data": {"recommendation": "unchecked"}})["product"]["recommendation"])

    def test_product_switch_discards_service_job_from_router_query_and_fields(self):
        question = "Instead, explain whether Example optimizer could help changing feed properties."
        raw = routed("product", "Replace 7 cloths on Example press in Exampleland and explain Example optimizer.",
                     equipment="Example press", country="Exampleland", task="cloth replacement",
                     cloth_count=7, cloths_supplied=True, round_trip_travel_hours=4)
        route, context, _ = reconcile_route(raw, messages(question), service_context())
        self.assertEqual(route["query"], question)
        self.assertIsNone(route["known_fields"]["equipment"])
        for name in ("country", "task", "cloth_count", "cloths_supplied", "round_trip_travel_hours"):
            self.assertIsNone(route["known_fields"][name])
        self.assertNotIn("service", context)
        self.assertEqual(context["product"]["intent"], "explain")

    def test_self_contained_product_question_retains_only_current_named_equipment(self):
        question = "How does Other pump control work?"
        route, _, _ = reconcile_route(routed("product", "Example press service; " + question,
                                              equipment="Other pump"), messages(question), service_context())
        self.assertEqual(route["query"], question)
        self.assertEqual(route["known_fields"]["equipment"], "Other pump")

    def test_unprovided_product_variant_cannot_enter_new_question_context(self):
        route, context, _ = reconcile_route(routed("product", "Help with Example PF-DS", equipment="Example PF-DS"),
                                           messages("Our Example pressure filter loses performance."))
        self.assertNotIn("PF-DS", route["query"])
        self.assertIsNone(context["product"]["equipment"])


class ServiceProvenanceTests(unittest.TestCase):
    def test_confirmed_basis_correction_and_additions_cannot_retire_active_job(self):
        prior = "Budget replacing 7 cloths on an Example press. Travel will be 8 hours."
        context = service_context()
        for message in ("Yes, that includes the return journey.",
                        "Actually make that 9 cloths, not 7.",
                        "Also include 3 fastening clips."):
            with self.subTest(message=message):
                route, state, trace = reconcile_route(routed("clarify"),
                    [{"role": "user", "content": prior}, {"role": "user", "content": message}], context,
                    interpretation={"route": "clarify", "relation": "new_request"})
                self.assertEqual(route["route"], "service_cost")
                self.assertEqual(state["service"]["fields"]["country"], "Exampleland")
                self.assertFalse(any(item.get("operation") == "current_request_boundary" for item in trace))

    def test_counted_customer_replacement_job_overrides_brochure_misroute(self):
        question = "A customer in Exampleland needs 7 filter cloths replaced on an Example press."
        route, context, trace = reconcile_route(routed("product", country="Exampleland", equipment="Example press",
                                                      task="cloth replacement", cloth_count=7), messages(question))
        self.assertEqual(route["route"], "service_cost")
        self.assertEqual(route["known_fields"]["cloth_count"], 7)
        self.assertEqual(route["known_fields"]["task"], "cloth replacement")
        self.assertEqual(context["last_route"], "service_cost")
        self.assertTrue(any("counted cloth replacement" in t.get("reason", "") for t in trace))
        self.assertEqual(validate_route(route), [])

    def test_explicit_job_count_comes_from_user_not_router_default(self):
        route, _, _ = reconcile_route(routed("clarify", cloth_count=99),
                                     messages("We need to replace eight cloths on an Example press."))
        self.assertEqual(route["route"], "service_cost")
        self.assertEqual(route["known_fields"]["cloth_count"], 8)
        route, _, _ = reconcile_route(routed("product", cloth_count=8),
                                     messages("A customer needs advice about their filter."))
        self.assertEqual(route["route"], "product")
        self.assertIsNone(route["known_fields"]["cloth_count"])

    def test_specification_suitability_and_instruction_questions_are_not_service_jobs(self):
        questions = ["What cloth length does Example PF have?",
                     "A customer has 7 cloths: which product is suitable for replacement?",
                     "Is Example PF suitable for a customer replacing 7 cloths?",
                     "How do I replace 7 cloths on an Example press?",
                     "The brochure mentions replacement of 7 cloths; what does it mean?",
                     "A customer had 7 cloths replaced. Why might the old cloths have failed?",
                     "A customer needs 7-9 cloths replaced; which filter should they choose?"]
        for question in questions:
            with self.subTest(question=question):
                route, _, _ = reconcile_route(routed("product", cloth_count=7), messages(question))
                self.assertEqual(route["route"], "product")

    def test_unprovided_false_and_travel_values_are_removed_without_mutating_raw_route(self):
        raw = routed("service_cost", country="Exampleland", equipment="Example press",
                     task="cloth replacement", cloth_count=7, cloths_supplied=False,
                     round_trip_travel_hours=3)
        original = deepcopy(raw)
        route, context, trace = reconcile_route(raw, messages("Replace 7 cloths on an Example press in Exampleland."))
        self.assertIsNone(route["known_fields"]["cloths_supplied"])
        self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])
        self.assertEqual(raw, original)
        self.assertEqual(len([t for t in trace if t.get("action") == "withheld"]), 2)
        self.assertEqual(validate_route(route), [])
        self.assertFalse(context["scenario_mode"])

    def test_explicit_supply_scope_is_used_instead_of_router_boolean(self):
        cases = [("The customer already has the replacement cloths.", True),
                 ("The customer will provide the cloths.", True),
                 ("The customer doesn't have any cloths.", False),
                 ("The technician must supply the replacement cloths.", False),
                 ("We need to buy the replacement cloths.", False)]
        for message, expected in cases:
            with self.subTest(message=message):
                route, context, _ = reconcile_route(
                    routed("service_cost", cloths_supplied=not expected), messages(message))
                self.assertIs(route["known_fields"]["cloths_supplied"], expected)
                self.assertEqual(context["service"]["provenance"]["cloths_supplied"]["user_text"], message)

    def test_historical_material_entry_does_not_establish_current_supply_scope(self):
        route, _, _ = reconcile_route(routed("service_cost", cloths_supplied=False),
                                     messages("The historical report lists cloths as a material cost."))
        self.assertIsNone(route["known_fields"]["cloths_supplied"])

    def test_some_customer_spares_do_not_establish_all_cloths_are_supplied(self):
        route, _, trace = reconcile_route(routed("service_cost", cloths_supplied=True),
                                          messages("The customer has some spare cloths."))
        self.assertIsNone(route["known_fields"]["cloths_supplied"])
        self.assertTrue(any("partial cloth stock" in t.get("reason", "") for t in trace))

    def test_negated_customer_stock_cannot_be_read_as_available_stock(self):
        for message in ("The customer has no replacement cloths.", "No customer-supplied cloths.",
                        "The cloths are not customer-supplied."):
            route, _, _ = reconcile_route(routed("service_cost", cloths_supplied=True), messages(message))
            self.assertIs(route["known_fields"]["cloths_supplied"], False, message)

    def test_yes_to_bundled_customer_supply_question_is_provenance(self):
        context = service_context(pending=("cloths_supplied", "round_trip_travel_hours"),
                                  question="Are the cloths already supplied, and what is the expected round-trip travel time?")
        route, _, _ = reconcile_route(routed("service_cost", cloths_supplied=False),
                                     messages("Yes. Travel there and back will take about four hours."), context)
        self.assertIs(route["known_fields"]["cloths_supplied"], True)
        self.assertEqual(route["known_fields"]["round_trip_travel_hours"], 4)

    def test_yes_to_service_supply_question_has_opposite_meaning(self):
        context = service_context(pending=("cloths_supplied",), question="Should the service technician supply the cloths?")
        route, _, _ = reconcile_route(routed("service_cost", cloths_supplied=True), messages("Yes"), context)
        self.assertIs(route["known_fields"]["cloths_supplied"], False)

    def test_prior_user_confirmed_supply_survives_unrelated_reply_and_router_default(self):
        route, context, _ = reconcile_route(routed("service_cost", cloths_supplied=False),
                                            messages("Travel there and back is 5 hours."), service_context())
        self.assertIs(route["known_fields"]["cloths_supplied"], True)
        self.assertEqual(context["service"]["provenance"]["cloths_supplied"]["basis"], "explicit customer cloth supply")

    def test_round_trip_is_not_doubled_but_explicit_each_way_is(self):
        for text, expected in [("Travel there and back is 4 hours.", 4),
                               ("Driving six hours each way.", 12),
                               ("The one-way travel is 2.5 hours.", 5),
                               ("No travel is required.", 0)]:
            with self.subTest(text=text):
                route, _, _ = reconcile_route(routed("service_cost", round_trip_travel_hours=999), messages(text))
                self.assertEqual(route["known_fields"]["round_trip_travel_hours"], expected)

    def test_one_trip_is_ambiguous_even_after_round_trip_question(self):
        route, _, trace = reconcile_route(routed("service_cost", round_trip_travel_hours=6),
                                         messages("One trip takes about 6 hours."), service_context())
        self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])
        self.assertTrue(any("one trip" in t.get("reason", "") for t in trace))

    def test_bare_numeric_reply_requires_only_travel_to_be_pending(self):
        route, _, _ = reconcile_route(routed("clarify", cloth_count=12), messages("12"), service_context())
        self.assertEqual(route["route"], "service_cost")
        self.assertEqual(route["known_fields"]["round_trip_travel_hours"], 12)
        self.assertEqual(route["known_fields"]["cloth_count"], 7)
        context = service_context(pending=("equipment", "round_trip_travel_hours"), question="Which model, and how many round-trip travel hours?")
        route, _, _ = reconcile_route(routed("service_cost", round_trip_travel_hours=12), messages("12"), context)
        self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])

    def test_unknown_travel_requests_scenarios_instead_of_repeating_question(self):
        route, context, trace = reconcile_route(routed("clarify", round_trip_travel_hours=4),
                                              messages("I don't know yet."), service_context())
        self.assertEqual(route["route"], "service_cost")
        self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])
        self.assertTrue(context["scenario_mode"])
        self.assertEqual(context["service"]["unknown_fields"], ["round_trip_travel_hours"])
        self.assertTrue(any(t.get("scenario_mode") for t in trace))

    def test_unknown_model_and_travel_are_both_respected(self):
        context = service_context(pending=("equipment", "round_trip_travel_hours"))
        route, context, _ = reconcile_route(routed("service_cost", equipment="Guessed model", round_trip_travel_hours=4),
                                           messages("I'm not sure what model or travel time yet."), context)
        self.assertIsNone(route["known_fields"]["equipment"])
        self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])
        self.assertEqual(set(context["service"]["unknown_fields"]), {"equipment", "round_trip_travel_hours"})

    def test_unknown_travel_does_not_erase_customer_cloth_confirmation(self):
        route, context, _ = reconcile_route(routed("service_cost", cloths_supplied=False),
                                           messages("I don't know the travel time, but the customer has cloths."), service_context())
        self.assertIs(route["known_fields"]["cloths_supplied"], True)
        self.assertEqual(context["service"]["unknown_fields"], ["round_trip_travel_hours"])

    def test_explicit_later_value_resolves_unknown_and_disables_scenarios(self):
        route, context, _ = reconcile_route(routed("service_cost"), messages("I don't know the travel time."), service_context())
        context = update_context(context, route, {"missing_fields": [], "question": None})
        route, context, _ = reconcile_route(routed("service_cost", round_trip_travel_hours=None),
                                           messages("Now confirmed: round-trip travel is 3 hours."), context)
        self.assertEqual(route["known_fields"]["round_trip_travel_hours"], 3)
        self.assertFalse(context["scenario_mode"])

    def test_new_visit_clears_old_provenance_and_stale_fields(self):
        context = service_context()
        route, context, _ = reconcile_route(
            routed("service_cost", country="Exampleland", equipment="Other machine", cloth_count=7,
                   cloths_supplied=True, round_trip_travel_hours=4),
            messages("A different customer needs a visit for an Other machine."), context)
        for field in ("country", "cloth_count", "cloths_supplied", "round_trip_travel_hours"):
            self.assertIsNone(route["known_fields"][field], field)
        self.assertEqual(context["service"]["provenance"], {})

    def test_self_contained_customer_request_resets_previous_verified_inputs(self):
        context = service_context()
        route, context, _ = reconcile_route(routed("service_cost", country="Otherland", equipment="Other press",
                                                  cloths_supplied=True, round_trip_travel_hours=4),
                                           messages("A customer in Otherland needs a visit for an Other press."), context)
        self.assertEqual(route["known_fields"]["country"], "Otherland")
        self.assertIsNone(route["known_fields"]["cloths_supplied"])
        self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])
        self.assertEqual(context["service"]["provenance"], {})

    def test_unfinished_service_does_not_capture_new_product_or_weather_question(self):
        for route_name, question in [("product", "What specifications does the Other pump have?"),
                                     ("unsupported", "What is the weather in Paris?")]:
            route, context, _ = reconcile_route(routed(route_name, question, equipment="Other pump"), messages(question), service_context())
            self.assertEqual(route["route"], route_name)
            self.assertNotIn("service", context)

    def test_ranges_and_multiple_durations_do_not_silently_choose_a_value(self):
        for message in ("Round-trip travel is 4-6 hours.", "Travel is 2 hours there and 3 hours back."):
            route, _, _ = reconcile_route(routed("service_cost", round_trip_travel_hours=4), messages(message))
            self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])

    def test_shared_unit_ranges_alternatives_and_negation_are_withheld_in_memory_as_in_billing(self):
        from metso_assistant.service_request import resolve_duration_inputs
        for message in ("Total travel is between 4 and 8 hours.",
                        "Total travel is either 4 or 8 hours.",
                        "Total travel is not 8 hours.",
                        "Travel there and back takes at least four hours."):
            with self.subTest(message=message):
                route, context, trace = reconcile_route(routed("service_cost", round_trip_travel_hours=8),
                    messages(message), service_context())
                self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])
                self.assertNotIn("round_trip_travel_hours",context["service"]["provenance"])
                self.assertTrue(any(item.get("action")=="withheld" and item.get("field")=="round_trip_travel_hours" for item in trace))
                record=next(record for record in resolve_duration_inputs([message]) if record.role=="travel_hours")
                self.assertIsNone(record.total_hours)

    def test_nominal_approximate_hours_are_accepted_with_estimate_provenance(self):
        route,context,_=reconcile_route(routed("service_cost"),
            messages("Travel there and back takes about four hours."),service_context())
        self.assertEqual(route["known_fields"]["round_trip_travel_hours"],4)
        self.assertIn("user-estimated nominal",context["service"]["provenance"]["round_trip_travel_hours"]["basis"])
        route,_,_=reconcile_route(routed("service_cost"),messages("Use 6 total travel hours."),service_context())
        self.assertEqual(route["known_fields"]["round_trip_travel_hours"],6)

    def test_unrelated_year_and_labour_hours_do_not_become_or_erase_confirmed_travel(self):
        self.assertEqual(_travel_value("The report was written in June 2017.",["round_trip_travel_hours"],"How many total travel hours?"),(None,None))
        route,context,_=reconcile_route(routed("service_cost"),messages("Total travel is 4 hours."),service_context())
        route,_,_=reconcile_route(routed("service_cost",round_trip_travel_hours=3),messages("Labour is 3 hours."),context)
        self.assertEqual(route["known_fields"]["round_trip_travel_hours"],4)

    def test_unrelated_equipment_on_site_is_not_proof_of_zero_travel(self):
        value,reason=_travel_value("The separator is already on site.",[],"")
        self.assertIsNone(value);self.assertIsNone(reason)

    def test_no_travel_budget_does_not_mean_zero_travel_duration(self):
        route, _, _ = reconcile_route(routed("service_cost", round_trip_travel_hours=0),
                                     messages("We have no travel budget information."))
        self.assertIsNone(route["known_fields"]["round_trip_travel_hours"])


if __name__ == "__main__":
    unittest.main()
