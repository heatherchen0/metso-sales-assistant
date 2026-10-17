"""Current request authority and field provenance across topic transitions."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from metso_assistant.model import LocalModel, ModelError, ACTION_ROUTES
from tests.test_model import routed


def subject(route="service_cost", relation="new_request"):
    return {"action": next(action for action, value in ACTION_ROUTES.items() if value == route), "relation": relation}, {}


def request(route="service_cost", *, needs_context=False):
    return {"action": next(action for action, value in ACTION_ROUTES.items() if value == route),
            "context_requirement": "needs_context" if needs_context else "self_contained"}, {}


class TaskMetadataTests(unittest.TestCase):
    def test_route_label_cannot_replace_the_actual_work_description(self):
        bad = routed("service_cost", task="service_cost")
        good = deepcopy(bad)
        good["known_fields"]["task"] = "replace the requested drive couplings"
        model = LocalModel()
        with patch.object(model, "invoke", side_effect=[request(), (bad, {}), (good, {})]) as invoke:
            result, traces = model.route([{"role": "user", "content": "Replace the drive couplings."}])
        self.assertEqual(result["known_fields"]["task"], good["known_fields"]["task"])
        self.assertEqual(invoke.call_count, 3)
        self.assertTrue(traces[1]["validation_errors"])
        self.assertFalse(traces[2]["validation_errors"])

    def test_followup_subject_is_interpreted_before_extracting_fields(self):
        model = LocalModel()
        history = [{"role": "user", "content": "Compare the recorded service visit costs."},
                   {"role": "assistant", "content": "PRIVATE source prices and unavailable conversion."},
                   {"role": "user", "content": "Can you combine those into another currency?"}]
        with patch.object(model, "invoke", side_effect=[request(needs_context=True), subject(relation="followup"), (routed("service_cost"), {})]) as invoke:
            result, traces = model.route(history)
        self.assertEqual(result["route"], "service_cost")
        self.assertEqual(result["query"], history[-1]["content"])
        self.assertNotIn("PRIVATE", str(invoke.call_args_list[0]))
        self.assertEqual(traces[0]["raw"]["relation"], "followup")

    def test_new_subject_fields_are_extracted_without_previous_job(self):
        model = LocalModel()
        history = [{"role": "user", "content": "Budget replacement parts in Exampleland."},
                   {"role": "assistant", "content": "PRIVATE historical cost subtotal"},
                   {"role": "user", "content": "Our flotation equipment loses fine particles. What can help?"}]
        with patch.object(model, "invoke", side_effect=[request("product"), (routed("product", query="stale job"), {})]) as invoke:
            result, _ = model.route(history)
        extraction = json.loads(invoke.call_args_list[1].args[1][0]["content"])
        self.assertEqual(extraction["user_messages"], [history[-1]["content"]])
        self.assertEqual(extraction["last_assistant_reply"], "")
        self.assertEqual(result["query"], history[-1]["content"])
        self.assertEqual(invoke.call_args_list[1].args[2]["properties"]["route"]["enum"], ["product"])
        first = json.loads(invoke.call_args_list[0].args[1][0]["content"])
        self.assertEqual(first, {"current_question": history[-1]["content"]})

    def test_only_an_unresolved_request_consults_verified_subject(self):
        model = LocalModel()
        history = [{"role": "user", "content": "Price a service visit."},
                   {"role": "user", "content": "That includes the return journey."}]
        state = {"last_route": "service_cost", "pending_fields": ["round_trip_travel_hours"],
                 "pending_question": "Does that include the return journey?"}
        with patch.object(model, "invoke", side_effect=[request("clarify", needs_context=True),
                subject("service_cost", "followup"), (routed("service_cost"), {})]) as invoke:
            result, traces = model.route(history, context=state)
        self.assertEqual(result["route"], "service_cost")
        self.assertNotIn("active_subject", json.loads(invoke.call_args_list[0].args[1][0]["content"]))
        self.assertEqual(json.loads(invoke.call_args_list[1].args[1][0]["content"])["active_subject"]["route"], "service_cost")
        self.assertEqual(traces[1]["phase"], "resolve_reference")

    def test_schema_valid_but_stale_domain_is_not_accepted(self):
        model = LocalModel()
        with patch.object(model, "invoke", side_effect=[request("product"), (routed("service_cost"), {}), (routed("product"), {})]):
            result, traces = model.route([{"role": "user", "content": "Our installed equipment has a new performance problem."}])
        self.assertEqual(result["route"], "product")
        self.assertIn("authoritative", traces[1]["validation_errors"][0])

    def test_unrelated_followup_can_remain_unsupported(self):
        model = LocalModel()
        with patch.object(model, "invoke", side_effect=[request("unsupported"), (routed("unsupported"), {})]):
            result, _ = model.route([{"role": "user", "content": "Price some service work."},
                                     {"role": "user", "content": "Tell me tomorrow's weather."}])
        self.assertEqual(result["route"], "unsupported")

    def test_followup_keeps_actual_users_and_only_a_bounded_latest_reply(self):
        model = LocalModel()
        history = [{"role": "user", "content": "Requested scope"},
                   {"role": "assistant", "content": "OLD PRIVATE EVIDENCE"},
                   {"role": "user", "content": "A correction"},
                   {"role": "assistant", "content": "X" * 12000},
                   {"role": "user", "content": "Add the next component."}]
        with patch.object(model, "invoke", side_effect=[request(needs_context=True), subject(relation="followup"), (routed("service_cost"), {})]) as invoke:
            model.route(history)
        payload = json.loads(invoke.call_args.args[1][0]["content"])
        self.assertEqual(payload["user_messages"], ["Requested scope", "A correction", "Add the next component."])
        self.assertEqual(len(payload["last_assistant_reply"]), 1000)
        self.assertNotIn("OLD PRIVATE EVIDENCE", str(payload))

    def test_truncated_routing_is_retried_once_and_failure_evidence_is_retained(self):
        model = LocalModel()
        failure = ModelError("Output budget exceeded")
        failure.partial_output, failure.metrics = '{"query":"unfinished', {"done_reason": "length"}
        with patch.object(model, "invoke", side_effect=[request(), failure, (routed("service_cost"), {})]) as invoke:
            result, traces = model.route([{"role": "user", "content": "Price the requested parts."}])
        self.assertEqual(result["route"], "service_cost")
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(traces[1]["raw_text"], failure.partial_output)
        with patch.object(model, "invoke", side_effect=[request(), failure, failure]):
            with self.assertRaises(ModelError) as caught:
                model.route([{"role": "user", "content": "Price the requested parts."}])
        self.assertEqual(len(caught.exception.traces), 3)

    def test_unresolved_subject_cannot_silently_inherit_old_route(self):
        model = LocalModel()
        with patch.object(model, "invoke", side_effect=[({"action": [], "context_requirement": "self_contained"}, {}),
                request("clarify", needs_context=True), subject("clarify", "followup"), (routed("clarify"), {})]):
            result, traces = model.route([{"role": "user", "content": "Compare some products."},
                                         {"role": "user", "content": "Do the other thing."}])
        self.assertEqual(result["route"], "clarify")
        self.assertTrue(traces[0]["validation_errors"])

    def test_standalone_request_uses_identical_classifier_at_any_position(self):
        question = "Our installed equipment has an unstable control response."
        current = {"role": "user", "content": question}
        state = {"last_route": "service_cost", "service": {"fields": {"country": "old country"}}}
        inputs = []
        for history, context in [([current], None), ([{"role": "user", "content": "An unrelated old job"}, current], state)]:
            model = LocalModel()
            with patch.object(model, "invoke", side_effect=[request("product"), (routed("product"), {})]) as invoke:
                model.route(history, context=context)
            inputs.append(invoke.call_args_list[0])
            extraction = json.loads(invoke.call_args_list[1].args[1][0]["content"])
            self.assertEqual(extraction["user_messages"], [question])
            self.assertEqual(extraction["last_assistant_reply"], "")
            self.assertNotIn("old country", str(extraction))
        self.assertEqual(inputs[0], inputs[1])

    def test_first_message_reference_has_no_invented_referent(self):
        model = LocalModel()
        with patch.object(model, "invoke", side_effect=[request("product", needs_context=True),
                subject("product", "followup"), (routed("clarify"), {})]) as invoke:
            result, traces = model.route([{"role": "user", "content": "What are its specifications?"}])
        self.assertEqual(result["route"], "clarify")
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(traces[1]["raw"]["relation"], "new_request")
        self.assertEqual(traces[1]["raw"]["context_requirement"], "needs_context")

    def test_standalone_subject_can_recover_from_dependency_uncertainty_without_prior_chat(self):
        model = LocalModel()
        with patch.object(model, "invoke", side_effect=[request("product", needs_context=True),
                subject("product", "new_request"), (routed("product"), {})]) as invoke:
            result, traces = model.route([{"role": "user", "content": "Tell me more about Example controller."}])
        self.assertEqual(result["route"], "product")
        resolution = json.loads(invoke.call_args_list[1].args[1][0]["content"])
        self.assertEqual(resolution["previous_user_messages"], [])
        self.assertIsNone(resolution["active_subject"]["route"])
        self.assertEqual(traces[1]["raw"]["context_requirement"], "self_contained")

    def test_product_reference_resolves_verified_subject_before_extraction(self):
        model = LocalModel()
        state = {"last_route": "product", "product": {"recommendation": "Example controller",
                 "subject_query": "Our equipment loses throughput."}}
        history = [{"role": "user", "content": state["product"]["subject_query"]},
                   {"role": "assistant", "content": "Assess Example controller."},
                   {"role": "user", "content": "What are its specifications?"}]
        with patch.object(model, "invoke", side_effect=[request("product", needs_context=True),
                subject("product", "followup"), (routed("product"), {})]) as invoke:
            result, traces = model.route(history, context=state)
        self.assertEqual(result["route"], "product")
        resolution = json.loads(invoke.call_args_list[1].args[1][0]["content"])
        self.assertEqual(resolution["active_subject"]["product"]["recommendation"], "Example controller")
        extraction = json.loads(invoke.call_args_list[2].args[1][0]["content"])
        self.assertEqual(extraction["user_messages"], [message["content"] for message in history if message["role"] == "user"])
        self.assertEqual(traces[1]["raw"]["relation"], "followup")

    def test_context_resolution_failure_is_bounded_and_retains_evidence(self):
        model = LocalModel()
        failure = ModelError("Output budget exceeded")
        failure.partial_output, failure.metrics = '{"action":', {"done_reason": "length"}
        history = [{"role": "user", "content": "Price the work"}, {"role": "user", "content": "Eight hours"}]
        with patch.object(model, "invoke", side_effect=[request(needs_context=True), failure, failure]) as invoke:
            with self.assertRaises(ModelError) as caught:
                model.route(history, context={"last_route": "service_cost"})
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(len(caught.exception.traces), 3)
        self.assertEqual(caught.exception.traces[-1]["raw_text"], failure.partial_output)
        self.assertEqual(caught.exception.traces[-1]["phase"], "resolve_reference")

    def test_unresolved_reply_cannot_invent_an_unrequested_action(self):
        model = LocalModel()
        history = [{"role": "user", "content": "Assess our process readings against the plant history."},
                   {"role": "user", "content": "Only one reading has changed. Keep the others."}]
        with patch.object(model, "invoke", side_effect=[request("clarify", needs_context=True),
                subject("product", "followup"), subject("flotation", "followup"), (routed("flotation"), {})]) as invoke:
            result, traces = model.route(history, context={"last_route": "flotation"})
        self.assertEqual(result["route"], "flotation")
        self.assertEqual(invoke.call_args_list[1].args[2]["properties"]["action"]["enum"],
                         ["plant_data_analysis", "clarification"])
        self.assertTrue(traces[1]["validation_errors"])
        self.assertFalse(traces[2]["validation_errors"])

    def test_explicit_dependent_operation_can_change_the_action(self):
        model = LocalModel()
        history = [{"role": "user", "content": "Describe the recommended equipment."},
                   {"role": "user", "content": "What would a service visit for it cost?"}]
        with patch.object(model, "invoke", side_effect=[request(needs_context=True),
                subject("service_cost", "followup"), (routed("service_cost"), {})]) as invoke:
            result, _ = model.route(history, context={"last_route": "product"})
        self.assertEqual(result["route"], "service_cost")
        self.assertEqual(invoke.call_args_list[1].args[2]["properties"]["action"]["enum"], list(ACTION_ROUTES))

    def test_unresolved_action_retains_subject_until_a_source_identity_is_verified(self):
        model = LocalModel()
        history = [{"role": "user", "content": "Our vessel pressure fluctuates."},
                   {"role": "user", "content": "Tell me more about Example pump."}]
        with patch.object(model, "invoke", side_effect=[request("clarify", needs_context=True),
                subject("product", "followup"), (routed("product"), {})]) as invoke:
            result, traces = model.route(history, context={"last_route": "product"})
        self.assertEqual(result["route"], "product")
        self.assertEqual(invoke.call_args_list[1].args[2]["properties"]["relation"]["enum"], ["followup"])
        self.assertEqual(traces[1]["raw"]["context_requirement"], "needs_context")
        self.assertEqual(traces[1]["raw"]["initial_context_requirement"], "needs_context")
        extraction = json.loads(invoke.call_args_list[2].args[1][0]["content"])
        self.assertEqual(extraction["user_messages"], [message["content"] for message in history])


if __name__ == "__main__":
    unittest.main()
