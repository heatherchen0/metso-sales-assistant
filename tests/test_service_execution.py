"""Request execution across short service conversations, with invented records."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from metso_assistant.adaptive import request_facts
from metso_assistant.service_execution import execute_service_request, evidence_rank, _installation_matches
from metso_assistant.service_request import bind_duration_facts, resolve_duration_inputs
from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.services import parse_report
from metso_assistant.store import build_store


class ServiceExecutionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        text = """Reference: FSR-9881
Site: Invented facility, Chile
Equipment: Helix Separator
On-site hours: 5
Travel hours: 1.5
| Task | Time Spent (hrs) |
|------|------------------|
| Replace Helix coupler cartridges (3 pcs) | 2 |
| Inspect Helix Separator | 1 |
| Preparation and testing | 2 |

| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Helix coupler cartridges | 3 | 7.15/ea | 21.45 |
| Labour | 5 hrs | 50/hr | 250 |
| Travel | 1.5 hrs | 20/hr | 30 |

Total Cost: USD 301.45
"""
        source = self.root / 'source.txt'
        source.write_text(text, encoding='utf-8')
        self.store = self.root / 'store.sqlite'
        build_store(self.store, [], [], [], [parse_report(source)])
        self.first = 'A customer needs 6 Helix coupler cartridges replaced on a Helix Separator. What should I budget?'

    def execute(self, messages, *, pending=(), question='', clarifications=(), categories=None):
        durations = resolve_duration_inputs(messages, pending_fields=pending, pending_question=question,
                                            clarifications=clarifications)
        inputs = bind_duration_facts(request_facts(messages, 'service_cost'), messages, durations,
            pending_fields=pending, pending_question=question, clarifications=clarifications)
        workspace = ServiceWorkspace(self.store, self.root / 'index.sqlite', inputs=inputs)
        plan = {'purpose': 'requested_work', 'requested_categories': categories if categories is not None else ['material', 'labour', 'travel'],
                'quantity_structure': 'direct', 'equipment_count_ref': None}
        result, trace = execute_service_request(workspace, plan, messages,
            {'country': 'Chile', 'equipment': 'Helix Separator'}, duration_inputs=durations)
        return result, trace, workspace

    def total(self, result, workspace):
        return workspace.facts[result['data']['subtotal_refs_by_currency']['USD']]['value']

    def test_supplied_return_hours_recalculate_without_model_arithmetic_loop(self):
        first, _, workspace = self.execute([self.first])
        self.assertEqual(self.total(first, workspace), Decimal('242.90'))
        second, trace, workspace = self.execute([self.first, 'Total billed travel will be 8 hours.'])
        self.assertEqual(self.total(second, workspace), Decimal('402.90'))
        self.assertFalse(any(item['operation'] == 'adaptive_action_selection' for item in trace))
        self.assertEqual(sum(item['action']['action'] == 'read_report' for item in trace if item['operation'] == 'adaptive_tool'), 1)

    def test_modified_travel_suffix_cannot_replace_source_scaled_replacement_labour(self):
        message = self.first + ' Only need replacement labour plus 6 hours of total billed travel.'
        result, _, workspace = self.execute([message], categories=['labour', 'travel'])
        # Source task: 2h/3 cartridges, requested6 =>4h. Travel: literal6h.
        expected = Decimal(2) / Decimal(3) * Decimal(6) * Decimal(50) + Decimal(6) * Decimal(20)
        self.assertEqual(self.total(result, workspace), expected)
        durations = {record['role']: record for record in result['data']['duration_inputs']}
        self.assertEqual(durations['labour_hours']['status'], 'missing')
        self.assertEqual(durations['travel_hours']['total_hours'], Decimal(6))
        self.assertNotIn('travel not yet priced', result['data']['summary'])

    def test_bare_travel_reply_is_priced_once_without_a_phantom_component(self):
        first, _, _ = self.execute([self.first])
        result, _, workspace = self.execute([self.first, '8'],
            pending=first['missing_fields'], question=first['question'])
        self.assertEqual(self.total(result, workspace), Decimal('402.90'))
        self.assertNotIn('Still unpriced:', result['data']['summary'])
        quantities = [fact for fact in workspace.facts.values() if fact.get('kind') == 'user_input'
                      and fact.get('scope', {}).get('role') == 'requested_quantity']
        self.assertEqual([fact['value'] for fact in quantities], [Decimal(6)])
        travel = [fact for fact in workspace.facts.values() if fact.get('kind') == 'user_input'
                  and fact.get('scope', {}).get('role') == 'travel_hours']
        self.assertEqual([(fact['value'], fact['unit']) for fact in travel], [(Decimal(8), 'hour')])

    def test_duration_reply_stays_bound_when_an_explicit_same_value_component_is_added(self):
        first, _, _ = self.execute([self.first])
        history = [self.first, '8', 'Also include 8 quartz seals.']
        result, _, workspace = self.execute(history, clarifications={
            1: {'fields': first['missing_fields'], 'question': first['question']}})
        self.assertEqual(self.total(result, workspace), Decimal('402.90'))
        quantities = [fact for fact in workspace.facts.values() if fact.get('kind') == 'user_input'
                      and fact.get('scope', {}).get('role') == 'requested_quantity']
        self.assertEqual([fact['value'] for fact in quantities], [Decimal(6), Decimal(8)])
        self.assertEqual(quantities[1]['scope']['entity_text'], 'quartz seals')
        self.assertIn('8 quartz seals', result['data']['summary'])
        self.assertNotIn('unspecified component', result['data']['summary'])

    def test_ambiguous_hours_stay_unpriced_then_confirmation_uses_original_value(self):
        first, _, workspace = self.execute([self.first + ' Travel time is 8 hours.'])
        self.assertEqual(self.total(first, workspace), Decimal('242.90'))
        self.assertIn('8', first['question'])
        self.assertIn('round_trip_travel_hours', first['missing_fields'])
        second, _, workspace = self.execute([self.first + ' Travel time is 8 hours.', 'Yes, including the return journey.'],
            pending=first['missing_fields'], question=first['question'])
        self.assertEqual(self.total(second, workspace), Decimal('402.90'))

    def test_owned_material_still_supports_work_and_extra_inspection_is_distinct(self):
        messages = [self.first, 'The customer already has the cartridges. Total billed travel is 8 hours.']
        result, _, workspace = self.execute(messages)
        self.assertEqual(self.total(result, workspace), Decimal('360.00'))
        result, _, workspace = self.execute([*messages, 'Also include an inspection.'])
        self.assertEqual(self.total(result, workspace), Decimal('410.00'))
        self.assertIn('Inspect Helix Separator', result['data']['summary'])

    def test_unknown_hours_do_not_trigger_repeated_question_or_guess(self):
        result, _, workspace = self.execute([self.first, 'I do not know travel hours.'])
        self.assertEqual(self.total(result, workspace), Decimal('242.90'))
        self.assertIsNone(result['question'])
        self.assertIn('20.00', result['data']['summary'])

    def test_question_confirmed_travel_persists_for_a_related_work_followup(self):
        first, _, _ = self.execute([self.first])
        prompts = [{}, {'fields': first['missing_fields'], 'question': first['question']}, {}]
        messages = [self.first, 'The travel time will be 8 hours.', 'Also include an inspection.']
        result, _, workspace = self.execute(messages, clarifications=prompts)
        self.assertEqual(self.total(result, workspace), Decimal('452.90'))
        self.assertNotIn('travel', ' '.join(result['missing_fields']))

    def test_uncertain_or_negated_travel_never_adds_a_travel_cost_or_phantom_material(self):
        for followup in ('Total travel is between 4 and 8 hours.',
                         'Total travel is either 4 or 8 hours.',
                         'Total travel is not 8 hours.'):
            with self.subTest(followup=followup):
                result,_,workspace=self.execute([self.first,followup])
                self.assertEqual(self.total(result,workspace),Decimal('242.90'))
                self.assertIn('round_trip_travel_hours',result['missing_fields'])
                self.assertNotIn('Still unpriced:',result['data']['summary'])
                self.assertFalse(any(fact.get('scope',{}).get('role') == 'requested_quantity'
                                     for fact in request_facts([followup],'service_cost')))

    def test_first_turn_replacement_and_inspection_match_the_same_work_split_across_turns(self):
        first,_,first_workspace=self.execute([self.first+' Also inspect the housing.'],categories=['labour'])
        split,_,split_workspace=self.execute([self.first,'Also inspect the housing.'],categories=['labour'])
        self.assertEqual(self.total(first,first_workspace),Decimal('250.00'))
        self.assertEqual(self.total(first,first_workspace),self.total(split,split_workspace))
        scope=first['data']['execution_work_scope']
        self.assertEqual(scope['priced_modes'],['inspection','replacement'])
        self.assertFalse(scope['unpriced_modes'])
        self.assertIn('original task example',first['data']['summary'])
        self.assertNotIn('Still unpriced:',first['data']['summary'])
        self.assertIn('labour task subtotal',first['data']['summary'])

    def test_explicitly_excluded_inspection_is_not_added_to_requested_replacement(self):
        result,_,workspace=self.execute([self.first+' Do not inspect the housing.'],categories=['labour'])
        self.assertEqual(self.total(result,workspace),Decimal('200.00'))
        self.assertNotIn('original task example',result['data']['summary'])
        self.assertEqual(result['data']['execution_work_scope']['requested_modes'],['replacement'])

    def test_known_travel_without_supported_work_duration_asks_for_labour_and_keeps_rates_visible(self):
        result,_,workspace=self.execute(['Budget a Helix Separator visit. Total travel is 4 hours.'],
                                         categories=['labour','travel'])
        self.assertEqual(self.total(result,workspace),Decimal('80.00'))
        self.assertEqual(result['question'],'How many total on-site labour hours should I use?')
        self.assertEqual(result['missing_fields'],['labour_hours'])
        self.assertIn('50.00/h',result['data']['summary'])
        self.assertIn('labour not yet priced',result['data']['summary'])

    def test_total_hours_modifier_reply_replaces_range_and_recalculates_known_components(self):
        messages=[self.first,'Total travel is between 4 and 8 hours.','Use 6 total travel hours.']
        result,_,workspace=self.execute(messages)
        self.assertEqual(self.total(result,workspace),Decimal('362.90'))
        self.assertIsNone(result['question'])
        self.assertNotIn('Still unpriced:',result['data']['summary'])
        unresolved,_,_=self.execute(messages[:2])
        confirmed,_,workspace=self.execute([*messages[:2],'4 hours'],
            pending=unresolved['missing_fields'],question=unresolved['question'])
        self.assertEqual(self.total(confirmed,workspace),Decimal('322.90'))
        self.assertIsNone(confirmed['question'])

    def test_nominal_travel_hours_are_priced_and_visibly_qualified_as_a_user_estimate(self):
        result,_,workspace=self.execute([self.first,'Total travel is about 4 hours.'])
        self.assertEqual(self.total(result,workspace),Decimal('322.90'))
        self.assertIsNone(result['question'])
        self.assertIn('billed travel for about 4 h',result['data']['summary'])
        self.assertIn('Uses your nominal hour estimate',result['data']['summary'])
        self.assertTrue(next(record for record in result['data']['duration_inputs'] if record['role']=='travel_hours')['estimated'])

    def test_checked_conditional_task_with_different_action_wording_is_not_also_reported_excluded(self):
        source=self.root/'source.txt'
        source.write_text(source.read_text(encoding='utf-8').replace(
            'Replace Helix coupler cartridges (3 pcs)','Helix coupler cartridge change (3 pcs)'),encoding='utf-8')
        self.store=self.root/'change-task.sqlite'
        build_store(self.store,[],[],[],[parse_report(source)])
        result,_,workspace=self.execute([self.first],categories=['labour'])
        self.assertEqual(self.total(result,workspace),Decimal('200.00'))
        self.assertEqual(result['data']['execution_work_scope']['priced_modes'],['replacement'])
        self.assertFalse(result['data']['execution_work_scope']['unpriced_modes'])
        self.assertNotIn('No supported duration for requested replacement',result['data']['summary'])
        self.assertNotIn('labour_hours',result['missing_fields'])
        self.assertIn('other visit work excluded',result['data']['summary'])

    def test_priced_replacement_does_not_hide_a_separate_unavailable_inspection(self):
        source=self.root/'source.txt'
        source.write_text(source.read_text(encoding='utf-8').replace('| Inspect Helix Separator | 1 |\n',''),encoding='utf-8')
        self.store=self.root/'no-inspection.sqlite'
        build_store(self.store,[],[],[],[parse_report(source)])
        result,_,workspace=self.execute([self.first+' Also inspect the housing.'],categories=['labour'])
        self.assertEqual(self.total(result,workspace),Decimal('200.00'))
        self.assertEqual(result['data']['execution_work_scope']['unpriced_modes'],['inspection'])
        self.assertIn('No supported duration for requested inspection',result['data']['summary'])

    def test_corrected_count_keeps_work_identity_and_a_companion_material_cannot_add_another_report_task(self):
        sources=[]
        for name,text in [('nominal-change','''Reference: FSR-9871
Site: Invented facility, Chile
Equipment: Helix Separator
On-site hours: 2
| Task | Time Spent (hrs) |
|------|------------------|
| Quartz module change (full set) | 2 |

| Item | Quantity | Unit Rate (CAD) | Total (CAD) |
|------|----------|-----------------|-------------|
| Quartz modules | 2 | 11.50/ea | 23.00 |
| Labour | 2 hrs | 50/hr | 100 |
'''),('alternative-task-and-pins','''Reference: FSR-9872
Site: Invented facility, Sweden
Equipment: Helix Separator
On-site hours: 1
| Task | Time Spent (hrs) |
|------|------------------|
| Quartz module replacement (3 pcs) | 1 |

| Item | Quantity | Unit Rate (EUR) | Total (EUR) |
|------|----------|-----------------|-------------|
| Locking pins | 2 | 2.60/ea | 5.20 |
| Labour | 1 hr | 40/hr | 40 |
''')]:
            path=self.root/(name+'.txt');path.write_text(text,encoding='utf-8');sources.append(parse_report(path))
        self.store=self.root/'two-task-examples.sqlite'
        build_store(self.store,[],[],[],sources)
        messages=['A customer needs 3 Quartz modules replaced on a Helix Separator. Price materials and labour.',
                  'Actually make that 6 modules, not 3. Distinguish estimated task labour from confirmed hours.']
        for history in (messages,[*messages,'Also include 2 locking pins; keep the material subtotal separate.']):
            with self.subTest(history=history):
                result,_,workspace=self.execute(history,categories=['material','labour'])
                service=result['data']['category_subtotal_refs_by_currency']['service']
                self.assertEqual(set(service),{'CAD'})
                self.assertEqual(workspace.facts[service['CAD']]['value'],Decimal(6)*Decimal(2)/Decimal(2)*Decimal(50))
                self.assertEqual(result['data']['execution_work_scope']['priced_modes'],['replacement'])
                self.assertFalse(result['data']['execution_work_scope']['unpriced_modes'])
                self.assertFalse(result['data']['execution_work_scope']['original_task_refs'])
                self.assertNotIn('No supported duration for requested replacement',result['data']['summary'])
                self.assertIsNone(result['question'])
                if len(history)==3:
                    materials=result['data']['category_subtotal_refs_by_currency']['material']
                    self.assertEqual(workspace.facts[materials['CAD']]['value'],Decimal(6)*Decimal('11.50'))
                    self.assertEqual(workspace.facts[materials['EUR']]['value'],Decimal(2)*Decimal('2.60'))
                    self.assertNotIn('original task example',result['data']['summary'])

    def test_source_model_identity_precedes_generic_component_word_fit(self):
        known = {'equipment': 'ZX900', 'country': 'Chile'}
        requested = {'id': 'U1', 'value': Decimal(2), 'kind': 'user_input', 'unit': 'each',
                     'user_text': '2 replacement couplers', 'scope': {'span': [0, 1], 'entity_text': 'replacement couplers'}}
        exact_model = {'label': 'Coupler, ZX900', 'scope': {'equipment': 'Separator family',
                       'entity_tokens': ['coupler', 'zx900'], 'country': 'Chile'}}
        other_model = {'label': 'Replacement coupler', 'scope': {'equipment': 'Another separator',
                        'entity_tokens': ['coupler'], 'country': 'Chile'}}
        self.assertLess(evidence_rank(exact_model, known, requested, {'U1': requested}),
                        evidence_rank(other_model, known, requested, {'U1': requested}))

    def test_component_ranking_keeps_part_nouns_inside_extracted_equipment_for_either_id_order(self):
        requested = {'id': 'U1', 'value': Decimal(2), 'kind': 'user_input', 'unit': 'each',
                     'user_text': '2 ZX700 pump rotors', 'scope': {'span': [0, 1]}}
        known = {'equipment': 'ZX700 pump rotors', 'country': 'Finland'}
        for right_id, wrong_id in (('SF00001', 'SF99999'), ('SF99999', 'SF00001')):
            with self.subTest(right_id=right_id):
                right = {'id': right_id, 'label': 'Rotor, ZX700', 'report_id': 'FSR-9873',
                         'scope': {'equipment': 'Apex transfer pump', 'equipment_family': 'pump',
                                   'entity_tokens': ['rotor', 'zx700'], 'country': 'Finland'}}
                wrong = deepcopy(right)
                wrong.update(id=wrong_id, label='Gasket, ZX700')
                wrong['scope']['entity_tokens'] = ['gasket', 'zx700']
                self.assertLess(evidence_rank(right, known, requested, {'U1': requested}),
                                evidence_rank(wrong, known, requested, {'U1': requested}))

    def test_material_selection_survives_source_path_relocation_with_contaminated_equipment(self):
        text = """Reference: FSR-9873
Site: Invented facility, Finland
Equipment: Apex transfer pump
| Item | Quantity | Unit Rate (EUR) | Total (EUR) |
|------|----------|-----------------|-------------|
| Rotor, ZX700 | 1 | 235.75/ea | 235.75 |
| Gasket, ZX700 | 1 | 9.20/ea | 9.20 |
"""
        source = self.root / 'FSR-9873.txt'; source.write_text(text, encoding='utf-8')
        original = source.read_bytes()
        message = 'A customer needs 2 ZX700 pump rotors replaced.'
        plan = {'purpose': 'requested_work', 'requested_categories': ['material'],
                'quantity_structure': 'direct', 'equipment_count_ref': None}
        chosen_ids = set()
        for index, path in enumerate(('legacy/data/HistoricalServices/FSR-9873.txt', 'FSR-9873.txt', str(source))):
            with self.subTest(path=path):
                record = parse_report(source); record['source']['path'] = path
                store = self.root / f'relocated-{index}.sqlite'
                build_store(store, [], [], [], [record])
                workspace = ServiceWorkspace(store, inputs=request_facts([message], 'service_cost'), source_dir=self.root)
                result, trace = execute_service_request(workspace, plan, [message],
                    {'country': 'Finland', 'equipment': 'ZX700 pump rotors'})
                subtotal = result['data']['category_subtotal_refs_by_currency']['material']['EUR']
                self.assertEqual(workspace.facts[subtotal]['value'], Decimal('471.50'))
                selected = [item['action']['items'][0]['rate_ref'] for item in trace
                            if item.get('operation') == 'adaptive_tool' and item['action']['action'] == 'price_items']
                self.assertEqual(len(selected), 1)
                self.assertEqual(workspace.facts[selected[0]]['scope']['description'], 'Rotor, ZX700')
                chosen_ids.add(selected[0])
                self.assertNotIn('Gasket', result['data']['summary'])
                self.assertEqual(source.read_bytes(), original)
        self.assertEqual(len(chosen_ids), 3)

    def test_inspection_duration_is_not_an_installation_duration(self):
        quantity = {'user_text': 'We need installation of 6 couplers; hours are unknown.'}
        self.assertFalse(_installation_matches({'label': 'Removal and inspection of couplers'}, quantity, {}))
        self.assertTrue(_installation_matches({'label': 'Installation of new couplers'}, quantity, {}))


if __name__ == '__main__':
    unittest.main()
