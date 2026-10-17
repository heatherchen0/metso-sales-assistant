"""Literal service updates retain scope without authorizing model facts."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from metso_assistant.adaptive import request_facts
from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_execution import execute_service_request
from metso_assistant.service_request import resolve_duration_inputs, service_continuation_proof
from metso_assistant.services import parse_report
from metso_assistant.store import build_store


FIELDS = {'equipment':'Helix Controller','task':'replace Quartz modules','country':'Chile'}
FIRST = 'A customer in Chile needs 5 Quartz modules replaced on a Helix Controller. Materials only.'


class ServiceContinuationTests(unittest.TestCase):
    def proof(self, prior, current, **kwargs):
        return service_continuation_proof(prior,current,service_fields=FIELDS,**kwargs)

    def test_return_journey_confirmation_uses_the_actual_prior_literal_hours(self):
        prior=FIRST+' Travel time will be 8 hours.'
        reply='Yes, that includes the return journey.'
        pending={'pending_fields':['round_trip_travel_hours'],
                 'pending_question':'You mentioned 8 hours. How many total travel hours including return?'}
        self.assertTrue(self.proof([prior],reply,**pending)['proven'])
        record=next(record for record in resolve_duration_inputs([prior,reply],**pending) if record.role=='travel_hours')
        self.assertEqual(record.total_hours,Decimal(8))
        self.assertEqual(record.quote,reply)
        self.assertEqual(record.origin_quote,prior)

    def test_basis_acknowledgement_can_continue_a_range_without_authorizing_its_endpoint(self):
        prior=FIRST+' Total travel is between 4 and 8 hours.'
        reply='Yes, including return.'
        pending={'pending_fields':['round_trip_travel_hours'],'pending_question':'How many total travel hours including return?'}
        self.assertTrue(self.proof([prior],reply,**pending)['proven'])
        record=next(record for record in resolve_duration_inputs([prior,reply],**pending) if record.role=='travel_hours')
        self.assertIsNone(record.total_hours)

    def test_bare_yes_to_a_bundled_question_is_not_a_travel_basis_proof(self):
        result=self.proof([FIRST+' Travel time is 8 hours.'],'yes',
            pending_fields=['round_trip_travel_hours','cloths_supplied'],
            pending_question='Are materials available and how many total travel hours?')
        self.assertFalse(result['proven'])

    def test_pending_hours_do_not_turn_unrelated_owned_equipment_observations_into_service_updates(self):
        pending={'pending_fields':['round_trip_travel_hours'],
                 'pending_question':'How many total billable travel hours, including the return journey?'}
        for observation in ('Our dryer overheats after 2 hours. What product could help?',
                            'Our separator forms a coating after 3 hours. Which product can help?',
                            'Our mixer can operate for 2 hours before it needs attention.',
                            'The drum has 2 hours total operating time. Tell me about its design.'):
            with self.subTest(observation=observation):
                self.assertFalse(self.proof([FIRST],observation,**pending)['proven'])

    def test_explicit_duration_updates_and_genuine_field_replies_still_continue_the_job(self):
        pending={'pending_fields':['round_trip_travel_hours'],
                 'pending_question':'How many total billable travel hours, including the return journey?'}
        for reply in ('8','8 hours','It will take 8 hours.','Use about 8 hours.',
                      'Total billed travel will be 8 hours.','Labour will take 3 hours.',
                      'Travel is 4 hours one way.','Also include 3 signal leads. Total billed travel is 8 hours.'):
            with self.subTest(reply=reply):
                self.assertTrue(self.proof([FIRST],reply,**pending)['proven'])
        self.assertFalse(self.proof([FIRST],'8 hours',pending_fields=['round_trip_travel_hours','labour_hours'],
            pending_question='What are the labour and travel hours?')['proven'])

    def test_basis_only_one_way_reply_preserves_resolver_owned_hours(self):
        prior=FIRST+' Travel time is 4 hours.'
        pending={'pending_fields':['round_trip_travel_hours'],'pending_question':'Does that include the return journey?'}
        self.assertTrue(self.proof([prior],'One way.',**pending)['proven'])
        record=next(record for record in resolve_duration_inputs([prior,'One way.'],**pending) if record.role=='travel_hours')
        self.assertEqual(record.total_hours,Decimal(8))

    def test_quantity_correction_is_proven_without_a_repeated_equipment_or_replacement_verb(self):
        result=self.proof([FIRST],'Actually make that 7 modules, not 5. Distinguish estimated task labour from confirmed hours.')
        self.assertTrue(result['proven'])
        self.assertIn('proven correction',result['reasons'][0])
        self.assertFalse(self.proof([FIRST],'Actually make that 7 rotors, not 5.')['proven'])

    def test_scope_additions_work_without_a_pending_question_and_with_a_new_component(self):
        for reply in ('Also include 3 signal leads. Give the material subtotal.',
                      'Also add 2 connector packs. What is the subtotal now?',
                      'Could you also include four signal leads?'):
            with self.subTest(reply=reply):
                self.assertTrue(self.proof([FIRST],reply)['proven'])

    def test_new_jobs_symptoms_and_physical_measurements_are_not_service_additions(self):
        for reply in ('A different customer needs 7 modules replaced.',
                      'Our flotation vessel loses particles below 20 microns. What can help?',
                      'The separator is already on site. What are its features?',
                      'Also include 20 microns of fine particles in that assessment.',
                      'Also include 2.5% silica in the historical comparison.',
                      'Actually the concentrate iron is 66%, not 65%.'):
            with self.subTest(reply=reply):
                self.assertFalse(self.proof([FIRST],reply)['proven'])

    def test_assistant_text_is_not_literal_input_and_current_tail_is_not_duplicated(self):
        reply='Also include 3 signal leads.'
        history=[{'role':'user','content':FIRST},
                 {'role':'assistant','content':'Invented user duration: 90 hours.'},
                 {'role':'user','content':reply}]
        self.assertTrue(self.proof(history,reply)['proven'])
        self.assertFalse(self.proof([{'role':'assistant','content':FIRST}],reply)['proven'])
        self.assertFalse(service_continuation_proof([FIRST],reply,
            service_fields={'equipment':'Unrelated trommel','task':'Trommel overhaul'})['proven'])


class CumulativeMaterialScopeTests(unittest.TestCase):
    def test_additions_and_corrections_retain_every_active_component_and_remove_only_obsolete_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'source.txt'
            source.write_text('''Reference: FSR-9876
Site: Invented facility, Chile
Equipment: Helix Controller
| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Quartz modules | 2 | 11.50/ea | 23.00 |
| Signal leads | 2 | 2.25/ea | 4.50 |
| Connector packs | 2 | 4.75/ea | 9.50 |
''',encoding='utf-8')
            store=root/'prepared.sqlite'
            build_store(store,[],[],[],[parse_report(source)])
            messages=[FIRST]
            replies=['Also include 3 signal leads. Give the material subtotal.',
                     'Actually make that 7 leads, not 3.',
                     'Also add 2 connector packs. What is the subtotal now?']
            expected=[Decimal(5)*Decimal('11.50')+Decimal(3)*Decimal('2.25'),
                      Decimal(5)*Decimal('11.50')+Decimal(7)*Decimal('2.25'),
                      Decimal(5)*Decimal('11.50')+Decimal(7)*Decimal('2.25')+Decimal(2)*Decimal('4.75')]
            for index,(reply,total) in enumerate(zip(replies,expected)):
                proof=service_continuation_proof(messages,reply,service_fields=FIELDS)
                self.assertTrue(proof['proven'])
                messages.append(reply)
                facts=request_facts(messages,'service_cost')
                workspace=ServiceWorkspace(store,root/'index.sqlite',inputs=facts)
                result,_=execute_service_request(workspace,{
                    'purpose':'requested_work','requested_categories':['material'],
                    'quantity_structure':'direct','equipment_count_ref':None},messages,FIELDS)
                subtotal=workspace.facts[result['data']['subtotal_refs_by_currency']['USD']]
                self.assertEqual(subtotal['value'],total)
                self.assertNotIn('Still unpriced',result['data']['summary'])
                if index>0:
                    self.assertTrue(any(fact['scope']['role']=='superseded_input' and fact['value']==3 for fact in facts))


if __name__ == '__main__':unittest.main()
