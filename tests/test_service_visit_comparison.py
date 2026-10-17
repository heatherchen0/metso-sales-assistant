"""Original whole-visit labour is an unscaled comparison, never a new bill."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from metso_assistant.adaptive import request_facts
from metso_assistant.service_answer import service_answer
from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_execution import execute_service_request
from metso_assistant.service_request import bind_duration_facts, resolve_duration_inputs
from metso_assistant.services import parse_report
from metso_assistant.store import build_store


class ServiceVisitComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first = 'A customer in Chile needs 3 quartz cartridges replaced on an Arc Mixer M10. Budget the visit.'

    def workspace(self, *, reported='4.5', tasks=True, incomplete=False, mixed=False, equipment='Arc Mixer M10'):
        task_rows = '''| Site safety induction | 0.5 |
| Machine lock-out and preparation | 0.25 |
| Replace quartz cartridges (2 pcs) | 2 |
| Inspect hydraulic seals | 0.75 |
| Functional testing | 1 |
| Clean-up and documentation | 0.5 |'''
        if incomplete:
            task_rows = task_rows.replace('| Functional testing | 1 |', '| Functional testing | — |')
        if mixed:
            task_rows = task_rows.replace('Functional testing', 'Preparation and testing').replace('Clean-up and documentation', 'Operator handover briefing')
        if not tasks:
            task_rows = '| Replace quartz cartridges (2 pcs) | 2 |'
        text = f'''Reference: FSR-9661
Site: Invented facility, Sweden
Equipment: {equipment}
On-site hours: {reported}
| Task | Time Spent (hrs) |
|------|------------------|
{task_rows}

| Item | Quantity | Unit Rate (EUR) | Total (EUR) |
|------|----------|-----------------|-------------|
| Quartz cartridges | 2 | 7/ea | 14 |
| Labour | {reported} hrs | 40/hr | {Decimal(reported)*40} |
'''
        source = self.root / 'source.txt'; source.write_text(text, encoding='utf-8')
        local = self.root / 'travel.txt'
        local.write_text('''Reference: FSR-9662
Site: Invented second facility, Chile
Equipment: Experimental transfer pump
| Item | Quantity | Unit Rate (USD) | Total (USD) |
|------|----------|-----------------|-------------|
| Travel | 2 hrs | 10/hr | 20 |
''', encoding='utf-8')
        store = self.root / 'prepared.sqlite'
        build_store(store, [], [], [], [parse_report(source), parse_report(local)])
        workspace = ServiceWorkspace(store, self.root / 'index.sqlite')
        workspace.read_report('FSR-9661')
        return workspace

    def references(self, workspace):
        task = next(f for f in workspace.facts.values() if f['kind'] == 'task_duration' and 'Replace' in f['label'])
        rate = next(f for f in workspace.facts.values() if f['kind'] == 'hourly_rate' and f['scope']['category'] == 'labour')
        return task, rate

    def execute(self, workspace, messages):
        durations = resolve_duration_inputs(messages)
        inputs = bind_duration_facts(request_facts(messages, 'service_cost'), messages, durations)
        current = ServiceWorkspace(workspace.store_path, workspace.index_path, inputs=inputs)
        result, trace = execute_service_request(current, {
            'purpose': 'requested_work', 'requested_categories': ['material', 'labour', 'travel'],
            'quantity_structure': 'direct', 'equipment_count_ref': None}, messages,
            {'equipment': 'Arc Mixer M10', 'country': 'Chile'}, duration_inputs=durations)
        return result, trace, current

    def test_discrepant_visit_preserves_both_original_bases_and_grouped_task_provenance(self):
        workspace = self.workspace()
        task, rate = self.references(workspace)
        comparison = workspace.visit_labour_comparison(task['id'], rate['id'])
        self.assertEqual(comparison['status'], 'available')
        data = comparison['data']; fact = lambda key: workspace.facts[data[key]]
        self.assertEqual(fact('itemized_hours_ref')['value'], Decimal(5))
        self.assertEqual(fact('other_hours_ref')['value'], Decimal(3))
        self.assertEqual(fact('itemized_amount_ref')['value'], Decimal(5)*rate['value'])
        self.assertEqual(data['country'], 'Sweden')
        reported = data['reported_bases'][0]
        self.assertEqual(workspace.facts[reported['hours_ref']]['value'], Decimal('4.5'))
        self.assertEqual(workspace.facts[reported['amount_ref']]['value'], Decimal('4.5')*rate['value'])
        self.assertFalse(reported['matches_itemized'])
        groups = {g['label']: workspace.facts[g['hours_ref']]['value'] for g in data['groups']}
        self.assertEqual(groups, {'Matched task': Decimal(2), 'Preparation and safety': Decimal('.75'),
                                  'Inspections': Decimal('.75'), 'Testing': Decimal(1),
                                  'Cleanup and documentation': Decimal('.5')})
        self.assertEqual(set(fact('itemized_hours_ref')['parents']), set(data['task_refs']))
        self.assertTrue(all(f['source_ids'] and f['scope']['comparison_only'] for f in comparison['facts']))

    def test_reconciled_report_from_a_different_product_keeps_original_hours_unscaled(self):
        workspace = self.workspace(reported='5', equipment='Orion Experimental dosing pump')
        task, rate = self.references(workspace)
        data = workspace.visit_labour_comparison(task['id'], rate['id'])['data']
        self.assertEqual(data['equipment'], 'Orion Experimental dosing pump')
        self.assertTrue(all(row['matches_itemized'] for row in data['reported_bases']))
        self.assertEqual(workspace.facts[data['itemized_amount_ref']]['value'], Decimal(5)*rate['value'])

    def test_reconciled_display_collapses_equal_bases_without_repeating_the_amount(self):
        workspace = self.workspace(reported='5')
        result, _, _ = self.execute(workspace, [self.first])
        text = result['data']['summary']
        self.assertIn('Listed tasks (matches reported/billed)', text)
        self.assertEqual(text.count('EUR 200.00'), 1)
        self.assertNotIn('differs from listed tasks', text)

    def test_mixed_activity_rows_are_retained_once_without_misleading_single_activity_groups(self):
        workspace = self.workspace(mixed=True); task, rate = self.references(workspace)
        data = workspace.visit_labour_comparison(task['id'], rate['id'])['data']
        other = next(group for group in data['groups'] if group['label'] == 'Other work')
        self.assertEqual(workspace.facts[other['hours_ref']]['value'], Decimal('1.5'))
        self.assertEqual({workspace.facts[ref]['label'] for ref in other['task_refs']},
                         {'Preparation and testing', 'Operator handover briefing'})
        self.assertEqual(workspace.facts[data['itemized_hours_ref']]['value'], Decimal(5))
        self.assertEqual(len(data['task_refs']), len(set(data['task_refs'])))

    def test_missing_task_hours_or_no_additional_tasks_do_not_create_a_whole_visit_cost(self):
        for option, reason in [({'incomplete': True}, 'incomplete_task_hours'),
                               ({'tasks': False}, 'no_additional_tasks')]:
            with self.subTest(option=option):
                # Each source/store is independent, as in a fresh preparation.
                nested = self.root / reason; nested.mkdir()
                original = self.root; self.root = nested
                workspace = self.workspace(**option)
                task, rate = self.references(workspace)
                result = workspace.visit_labour_comparison(task['id'], rate['id'])
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['data']['reason'], reason)
                self.assertFalse(result['facts'])
                self.root = original

    def test_comparison_only_values_cannot_be_reused_as_customer_budget_operands(self):
        workspace = self.workspace(); task, rate = self.references(workspace)
        data = workspace.visit_labour_comparison(task['id'], rate['id'])['data']
        self.assertIn('comparison-only', workspace.calculate('multiply', [rate['id'], data['itemized_hours_ref']])['error'])
        self.assertIn('comparison-only', workspace.calculate('sum', [data['itemized_amount_ref'], data['reported_bases'][0]['amount_ref']])['error'])
        travel = workspace.read_report('FSR-9662')['facts'][0]
        invalid = workspace.visit_labour_comparison(task['id'], travel['id'])
        self.assertTrue(invalid.get('error'))

    def test_replacement_budget_and_travel_followup_do_not_absorb_the_comparison(self):
        workspace = self.workspace()
        first, _, current = self.execute(workspace, [self.first])
        follow, _, updated = self.execute(workspace, [self.first, 'Total billed travel will be 8 hours.'])
        for result, registry in [(first, current), (follow, updated)]:
            refs = result['data']['category_subtotal_refs_by_currency']
            self.assertEqual(registry.facts[refs['service']['EUR']]['value'], Decimal(3)*40)
            self.assertEqual(registry.facts[refs['material']['EUR']]['value'], Decimal(3)*7)
            self.assertEqual(registry.facts[result['data']['subtotal_refs_by_currency']['EUR']]['value'], Decimal(3)*47)
            self.assertEqual(len(result['data']['historical_visit_labour_comparisons']), 1)
            text = result['data']['summary']
            self.assertIn('requested replacement-task labour only', text)
            self.assertIn('Historical whole-visit labour — FSR-9661 (Sweden)', text)
            self.assertIn('Other listed work: 3 h', text)
            self.assertIn('not added to this task budget', text)
            self.assertIn('EUR 200.00', text); self.assertIn('EUR 180.00', text)
            self.assertNotIn('EUR 320.00', text)
            self.assertNotIn('source task totals do not reconcile', text)
        self.assertIn('8 h', follow['data']['summary'])
        self.assertEqual(follow['data']['category_subtotal_ranges_by_currency']['service']['USD']['minimum'], Decimal(8)*10)
        # Calling the renderer again cannot discover comparison amounts as bills.
        repeated, _ = service_answer(updated, [], [self.first, 'Total billed travel will be 8 hours.'],
            {'country': 'Chile'}, {'purpose': 'requested_work', 'requested_categories': ['material', 'labour']})
        self.assertEqual(updated.facts[repeated['data']['subtotal_refs_by_currency']['EUR']]['value'], Decimal(3)*47)


if __name__ == '__main__':
    unittest.main()
