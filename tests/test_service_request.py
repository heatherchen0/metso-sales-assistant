"""Literal durations retain ambiguity until their billing interpretation is known."""
from dataclasses import FrozenInstanceError
from decimal import Decimal
import unittest

from metso_assistant.service_request import DurationInput, bind_duration_facts, duration_question, resolve_duration_inputs, duration_literal_spans


def travel(messages, **kwargs):
    return next(record for record in resolve_duration_inputs(messages, **kwargs) if record.role == "travel_hours")


class ServiceRequestTests(unittest.TestCase):
    def test_ambiguous_eight_is_retained_with_hour_unit_quote_and_a_focused_question(self):
        message = "The travel time will be 8 hours."
        records = resolve_duration_inputs([message]); value = next(record for record in records if record.role == "travel_hours")
        self.assertEqual((value.value,value.unit,value.basis,value.status),(Decimal(8),"hour","unspecified","ambiguous"))
        self.assertEqual(value.quote,message)
        self.assertIsNone(value.total_hours); self.assertIsNone(value.to_fact("U8"))
        self.assertIn("8 hours",duration_question(records));self.assertIn("return journey",duration_question(records))

    def test_explicit_total_and_one_way_hours_have_checked_total_operands(self):
        total = travel(["Total travel will be 8 hours."])
        self.assertEqual((total.status,total.basis,total.total_hours),("known","total",Decimal(8)))
        self.assertEqual(total.to_fact("U8")["value"],Decimal(8))
        billed = travel(["The customer already has the cartridges. Total billed travel will be 8 hours."])
        self.assertEqual((billed.status,billed.basis,billed.total_hours),("known","total",Decimal(8)))
        one = travel(["Travel will be 8 hours each way."])
        self.assertEqual((one.value,one.basis,one.total_hours),(Decimal(8),"oneway",Decimal(16)))
        self.assertTrue(one.to_fact("U8")["assumptions"])

    def test_confirmation_preserves_original_value_and_literal_confirmation_provenance(self):
        first="Travel will be 8 hours."; question=duration_question(resolve_duration_inputs([first]))
        for reply in ("yes","total","including return","round-trip","Yes, including the return journey","Yes, that includes the return journey.","It includes the return trip."):
            with self.subTest(reply=reply):
                value=travel([first,reply],pending_fields=["round_trip_travel_hours"],pending_question=question)
                self.assertEqual(value.total_hours,Decimal(8));self.assertEqual(value.quote,reply)
                self.assertIn(first,value.to_fact("U8")["user_text"])
                self.assertEqual(value.to_fact("U8")["scope"]["input_quote"],reply)

    def test_bare_numeric_reply_is_known_only_after_a_total_travel_question(self):
        value=travel(["8"],pending_fields=["round_trip_travel_hours"],pending_question="How many total travel hours, including the return?")
        self.assertEqual(value.total_hours,Decimal(8))
        self.assertEqual(travel(["8"]).status,"missing")
        self.assertEqual(travel(["8"],pending_fields=["travel_hours"],pending_question="How long is one trip?").status,"ambiguous")
        self.assertEqual(travel(["8 total"],pending_fields=["travel_hours"],pending_question="What travel time?").total_hours,Decimal(8))
        self.assertEqual(travel(["8 each way"],pending_fields=["travel_hours"],pending_question="What total travel time?").total_hours,Decimal(16))

    def test_duration_binding_consumes_only_the_proven_literal_span_and_keeps_operand_ids(self):
        from metso_assistant.adaptive import request_facts
        messages = ['Replace 8 packing rings.', '  8  ']
        raw = request_facts(messages, 'service_cost')
        kwargs = {'pending_fields': ['round_trip_travel_hours'], 'pending_question': 'How many total travel hours?'}
        records = resolve_duration_inputs(messages, **kwargs)
        result = bind_duration_facts(raw, messages, records, **kwargs)
        self.assertEqual([fact['id'] for fact in result], ['U001', 'U_duration_travel_hours'])
        self.assertEqual(result[0], raw[0])
        self.assertEqual(result[1]['value'], Decimal(8))
        self.assertEqual(result[1]['scope']['input_quote'], messages[1])
        self.assertEqual(raw[1]['scope']['role'], 'requested_quantity')

    def test_bare_quantity_without_a_unique_pending_duration_is_not_consumed(self):
        from metso_assistant.adaptive import request_facts
        for fields in ([], ['round_trip_travel_hours', 'labour_hours']):
            with self.subTest(fields=fields):
                messages = ['8']
                raw = request_facts(messages, 'service_cost')
                kwargs = {'pending_fields': fields, 'pending_question': 'How many hours?'}
                result = bind_duration_facts(raw, messages, resolve_duration_inputs(messages, **kwargs), **kwargs)
                self.assertEqual(result, raw)

    def test_pending_duration_does_not_consume_an_explicit_material_addition(self):
        from metso_assistant.adaptive import request_facts
        messages = ['Add 8 packing rings.']
        raw = request_facts(messages, 'service_cost')
        kwargs = {'pending_fields': ['round_trip_travel_hours'], 'pending_question': 'How many total travel hours?'}
        self.assertEqual(bind_duration_facts(raw, messages, resolve_duration_inputs(messages, **kwargs), **kwargs), raw)

    def test_each_way_or_unresolved_duration_reply_is_not_a_material_quantity(self):
        from metso_assistant.adaptive import request_facts
        for reply, question, expected in [('8 each way', 'How many total travel hours?', Decimal(16)),
                                          ('8', 'How long is the journey?', None)]:
            with self.subTest(reply=reply):
                messages = [reply]
                kwargs = {'pending_fields': ['travel_hours'], 'pending_question': question}
                result = bind_duration_facts(request_facts(messages, 'service_cost'), messages,
                                             resolve_duration_inputs(messages, **kwargs), **kwargs)
                self.assertFalse(any(fact.get('scope', {}).get('role') == 'requested_quantity' for fact in result))
                self.assertEqual([fact['value'] for fact in result], [] if expected is None else [expected])

    def test_earlier_duration_reply_stays_consumed_after_an_update_and_later_component_request(self):
        from metso_assistant.adaptive import request_facts
        messages = ['Replace 8 packing rings.', '8', 'Total travel is 6 hours.', 'Also add 8 quartz seals.']
        history = {1: {'fields': ['round_trip_travel_hours'], 'question': 'How many total travel hours?'}}
        records = resolve_duration_inputs(messages, clarifications=history)
        raw = request_facts(messages, 'service_cost')
        result = bind_duration_facts(raw, messages, records, clarifications=history)
        quantities = [fact for fact in result if fact.get('scope', {}).get('role') == 'requested_quantity']
        self.assertEqual([(fact['id'], fact['scope']['entity_text']) for fact in quantities],
                         [(raw[0]['id'], 'packing rings'), (raw[-1]['id'], 'quartz seals')])
        self.assertEqual(result[-1]['value'], Decimal(6))

    def test_a_question_about_total_travel_is_not_confirmation_of_the_previous_eight(self):
        value=travel(["Travel will be 8 hours.","Can you show the total travel cost?"],pending_fields=["round_trip_travel_hours"],pending_question="How many total travel hours?")
        self.assertEqual(value.status,"ambiguous");self.assertIsNone(value.to_fact("U8"))

    def test_explicit_unknown_overrides_an_old_duration_and_is_distinct_from_missing(self):
        value=travel(["Round-trip travel is 8 hours.","I don't know the travel time now."])
        self.assertEqual(value.status,"unknown");self.assertIsNone(value.value)
        self.assertEqual(travel([]).status,"missing")
        self.assertIsNone(duration_question([value]))

    def test_missing_requested_travel_gets_one_total_return_question_but_excluded_categories_do_not(self):
        records=resolve_duration_inputs(["Please budget for a service visit."])
        question=duration_question(records,{"travel"})
        self.assertIn("total billable travel hours",question)
        self.assertIn("return journey",question)
        self.assertIsNone(duration_question(records,set()))
        self.assertIsNone(duration_question(records,{"materials"}))
        resolved=travel(["Please budget for a service visit.","The travel time will be 8 hours."],pending_fields=["round_trip_travel_hours"],pending_question=question)
        self.assertEqual(resolved.total_hours,Decimal(8))

    def test_recorded_question_history_keeps_confirmed_duration_across_later_work_followups(self):
        messages=["Please budget a service visit.","The travel time will be 8 hours.","Also include an inspection."]
        question=duration_question(resolve_duration_inputs(messages[:1]),{"travel"})
        for clarification_history in ([None,{"fields":["round_trip_travel_hours"],"question":question},None],
                                      {1:{"fields":["round_trip_travel_hours"],"question":question}}):
            with self.subTest(history=clarification_history):
                value=travel(messages,clarifications=clarification_history)
                self.assertEqual((value.status,value.basis,value.total_hours),("known","total",Decimal(8)))
                self.assertEqual(value.quote,messages[1]);self.assertEqual(value.turn,1)
        self.assertEqual(travel(messages).status,"ambiguous")

    def test_latest_duration_replaces_old_value_without_changing_another_billing_role(self):
        records=resolve_duration_inputs(["Labour is 3 hours and total travel is 4 hours.","Total travel will be 8 hours."])
        values={r.role:r.total_hours for r in records}
        self.assertEqual(values,{"labour_hours":Decimal(3),"travel_hours":Decimal(8)})

    def test_parts_and_unknown_budgets_are_not_duration_values_or_zero_travel(self):
        self.assertEqual(travel(["Replace 8 packing rings."]).status,"missing")
        self.assertEqual(travel(["We have no travel budget yet."]).status,"missing")
        zero=travel(["Travel is not required."])
        self.assertEqual(zero.total_hours,Decimal(0))

    def test_range_conflicting_basis_and_multiple_travel_durations_remain_unpriced(self):
        for message in ("Total travel is 2 to 4 hours.","Round-trip travel is 8 hours each way.","Travel is 8 hours or 10 hours."):
            with self.subTest(message=message):
                value=travel([message]);self.assertEqual(value.status,"ambiguous");self.assertIsNone(value.to_fact("U8"))

    def test_shared_unit_ranges_alternatives_and_negation_never_select_a_billable_endpoint(self):
        for text, reason, candidates in (
                ("Total travel is between 4 and 8 hours.", "range", (Decimal(4),Decimal(8))),
                ("Total travel is either 4 or 8 hours.", "alternatives", (Decimal(4),Decimal(8))),
                ("Total travel is not 8 hours.", "negated", (Decimal(8),)),
                ("Total travel is 4-8 hours.", "range", (Decimal(4),Decimal(8))),
                ("Total travel of 8 hours is not correct.", "negated", (Decimal(8),)),
                ("Labour will be between two and five hours.", "range", (Decimal(2),Decimal(5)))):
            with self.subTest(text=text):
                records=resolve_duration_inputs([text])
                record=next(value for value in records if value.status == "ambiguous")
                self.assertEqual((record.uncertainty,record.candidates),(reason,candidates))
                self.assertIsNone(record.value);self.assertIsNone(record.total_hours)
                self.assertIsNone(record.to_fact("Ubad"));self.assertTrue(duration_question(records))
                self.assertEqual(record.quote,text)

    def test_range_endpoints_are_duration_spans_and_not_material_counts(self):
        text="Replace 6 couplers; total travel is between four and 8 hours."
        self.assertEqual([text[a:b] for a,b in duration_literal_spans(text)],["four","8"])
        self.assertEqual(duration_literal_spans("Replace 6 couplers and 8 seals."),())

    def test_basis_only_confirmation_cannot_promote_an_uncertain_or_denied_duration(self):
        for first in ("Total travel is not 8 hours.","Travel is between 4 and 8 hours.","Travel may be 8 hours."):
            with self.subTest(first=first):
                value=travel([first,"yes"],pending_fields=["round_trip_travel_hours"],pending_question="How many total travel hours?")
                self.assertEqual(value.status,"ambiguous");self.assertIsNone(value.to_fact("Ubad"))
        value=travel(["Total travel is between 4 and 8 hours.","Total travel is 6 hours."])
        self.assertEqual(value.total_hours,Decimal(6))

    def test_missing_requested_labour_is_asked_when_travel_is_already_known(self):
        records=resolve_duration_inputs(["Total travel is 4 hours."])
        self.assertEqual(duration_question(records,{"labour","travel"}),"How many total on-site labour hours should I use?")
        records=resolve_duration_inputs(["Total travel is 4 hours. Labour hours are unknown."])
        self.assertIsNone(duration_question(records,{"labour","travel"}))

    def test_modifier_phrases_resolve_literal_total_duration_and_replace_an_earlier_range(self):
        first="Total travel is between 4 and 8 hours."
        for reply in ("Use 6 total travel hours.","Use six total billable travel hours.","Use 6 estimated total travel hours."):
            with self.subTest(reply=reply):
                value=travel([first,reply])
                self.assertEqual(value.total_hours,Decimal(6));self.assertEqual(value.quote,reply)
                self.assertIsNone(value.uncertainty)
                self.assertEqual([reply[a:b] for a,b in duration_literal_spans(reply)],["six" if "six" in reply else "6"])

    def test_nominal_user_estimates_are_usable_only_with_billing_basis_and_keep_an_assumption(self):
        for message in ("Round-trip travel takes about four hours.","Use 4 estimated total travel hours.",
                        "Total travel takes approximately 4 hours.","Total travel is roughly 4 hours."):
            with self.subTest(message=message):
                value=travel([message]);fact=value.to_fact("Unominal")
                self.assertEqual(value.total_hours,Decimal(4));self.assertTrue(value.estimated)
                self.assertTrue(fact['scope']['duration_estimated'])
                self.assertTrue(any('approximate nominal estimate' in item for item in fact['assumptions']))
        first="Travel is about 4 hours."
        value=travel([first]);self.assertEqual(value.status,"ambiguous");self.assertTrue(value.estimated)
        value=travel([first,"including return"])
        self.assertEqual(value.total_hours,Decimal(4));self.assertTrue(value.estimated)
        value=travel(["about 4"],pending_fields=['round_trip_travel_hours'],pending_question='How many total travel hours?')
        self.assertEqual(value.total_hours,Decimal(4));self.assertTrue(value.estimated)

    def test_bounds_and_modals_remain_non_billable_even_with_duration_modifiers(self):
        for message in ('Total travel takes at least 4 hours.','Use between 4 and 8 total travel hours.',
                        'Total travel may be 4 hours.','Total travel is at most 4 hours.',
                        'Use either four or eight estimated travel hours.'):
            with self.subTest(message=message):
                value=travel([message]);self.assertEqual(value.status,'ambiguous')
                self.assertIsNone(value.total_hours);self.assertIsNone(value.to_fact('Ubad'))

    def test_duration_records_are_immutable_and_do_not_accept_nonfinite_known_values(self):
        record=DurationInput("travel_hours",Decimal(8),basis="total",status="known")
        with self.assertRaises(FrozenInstanceError):record.value=Decimal(9)
        with self.assertRaises(ValueError):DurationInput("travel_hours",Decimal("NaN"),basis="total",status="known")
        with self.assertRaises(ValueError):DurationInput("travel_hours",Decimal(-1),basis="total",status="known")


if __name__ == "__main__":unittest.main()
