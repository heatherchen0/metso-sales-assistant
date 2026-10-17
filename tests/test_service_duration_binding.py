"""Billing roles attach to literal duration phrases, not an earlier category."""
from decimal import Decimal
import unittest

from metso_assistant.service_request import bind_duration_facts, resolve_duration_inputs


class DurationRoleBindingTests(unittest.TestCase):
    def records(self, message, **kwargs):
        return {record.role: record for record in resolve_duration_inputs([message], **kwargs)}

    def test_modified_following_roles_bind_in_either_category_order(self):
        for message, role, hours in (
                ('Need replacement labour plus 6 hours of total billed travel.', 'travel_hours', 6),
                ('Need installation work and 5 hours for total billable driving.', 'travel_hours', 5),
                ('Need labour plus 7 hours total chargeable journey.', 'travel_hours', 7),
                ('Need travel plus 3 hours of total on-site labour.', 'labour_hours', 3),
                ('Need travel and 4 hours for estimated installation.', 'labour_hours', 4),
                ('Need driving plus 2 hours of billed repair work.', 'labour_hours', 2)):
            with self.subTest(message=message):
                records = self.records(message)
                record = records[role]
                self.assertEqual((record.status, record.total_hours), ('known', Decimal(hours)))
                other = 'labour_hours' if role == 'travel_hours' else 'travel_hours'
                self.assertEqual(records[other].status, 'missing')
                fact = record.to_fact('DURATION')
                self.assertEqual(fact['scope']['role'], role)
                self.assertEqual(fact['user_text'], message)

    def test_both_separate_role_values_survive_prefix_modifier_and_suffix_placements(self):
        for message in (
                'Labour is 3 hours and total billed travel is 6 hours.',
                'Total billed travel is 6 hours and labour is 3 hours.',
                'Use 3 labour hours plus 6 hours of total billed travel.',
                'Use 3 labour hours and 6 total travel hours.',
                'Use 6 total travel hours and 3 labour hours.',
                'Use 6 total travel hours plus 3 hours for on-site work.',
                'Use 3 hours of labour; use 6 hours of total billed travel.',
                'Use 6 hours of total billed travel, and 3 hours of labour.'):
            with self.subTest(message=message):
                self.assertEqual({role: record.total_hours for role, record in self.records(message).items()},
                                 {'labour_hours': Decimal(3), 'travel_hours': Decimal(6)})

    def test_direct_duration_role_wins_over_general_job_context(self):
        for message in ('Budget the repair with 6 total travel hours.',
                        'The work needs 6 hours of total billed travel.',
                        'Installation requires 6 total billed travel hours.',
                        'The labour budget includes 6 hours of total billed travel.'):
            with self.subTest(message=message):
                records = self.records(message)
                self.assertEqual(records['travel_hours'].total_hours, Decimal(6))
                self.assertEqual(records['labour_hours'].status, 'missing')

    def test_one_way_total_and_nominal_modifiers_belong_to_the_same_role(self):
        for message, total, estimated in (
                ('Need labour plus 3 hours of one-way travel.', 6, False),
                ('Need work plus 3 hours for travel each way.', 6, False),
                ('Need repair labour plus 6 hours of round-trip travel.', 6, False),
                ('Need labour plus about 6 hours of total billed travel.', 6, True),
                ('Need labour plus 6 estimated total travel hours.', 6, True)):
            with self.subTest(message=message):
                record = self.records(message)['travel_hours']
                self.assertEqual(record.total_hours, Decimal(total))
                self.assertEqual(record.estimated, estimated)
                self.assertEqual(self.records(message)['labour_hours'].status, 'missing')

    def test_billing_basis_does_not_leak_between_role_clauses(self):
        for message in ('Labour is 3 hours total and travel is 6 hours.',
                        'Labour is 3 hours total plus travel is 6 hours.',
                        'Labour is 3 hours total. Travel is 6 hours.'):
            with self.subTest(message=message):
                records = self.records(message)
                self.assertEqual(records['labour_hours'].total_hours, Decimal(3))
                self.assertEqual((records['travel_hours'].value, records['travel_hours'].status),
                                 (Decimal(6), 'ambiguous'))
                self.assertIsNone(records['travel_hours'].to_fact('TRAVEL'))
        for message in ('Labour is 3 hours one-way and total travel is 6 hours.',
                        'Total travel is 6 hours and labour is 3 hours one-way.'):
            with self.subTest(message=message):
                self.assertEqual(self.records(message)['travel_hours'].total_hours, Decimal(6))

    def test_explicit_role_conflicts_cannot_be_overridden_by_pending_questions(self):
        for message in ('Use 3 labour hours for travel.',
                        'Travel will take 3 labour hours.',
                        'Use 3 travel hours for on-site work.'):
            for pending in ((), ('round_trip_travel_hours',), ('labour_hours',)):
                with self.subTest(message=message, pending=pending):
                    records = self.records(message, pending_fields=pending,
                                           pending_question='How many total billable hours?')
                    for record in records.values():
                        self.assertEqual((record.status, record.candidates), ('ambiguous', (Decimal(3),)))
                        self.assertIsNone(record.to_fact('CONFLICT'))

    def test_modified_following_role_preserves_uncertainty_instead_of_billing_an_endpoint(self):
        for message, uncertainty in (
                ('Need labour plus between 2 and 6 hours of total billed travel.', 'range'),
                ('Need labour plus either 2 or 6 hours of total billed travel.', 'alternatives'),
                ('Need labour plus not 6 hours of total billed travel.', 'negated'),
                ('Need labour plus at least 6 hours of total billed travel.', 'qualified'),
                ('Need labour plus 6 hours of round-trip travel each way.', None)):
            with self.subTest(message=message):
                records = self.records(message)
                record = records['travel_hours']
                self.assertEqual(record.status, 'ambiguous')
                self.assertEqual(record.uncertainty, uncertainty)
                self.assertIsNone(record.to_fact('UNRESOLVED'))
                self.assertEqual(records['labour_hours'].status, 'missing')

    def test_role_negation_and_nominal_estimate_do_not_cross_clause_boundaries(self):
        records = self.records('Labour is not needed plus 6 hours of total billed travel.')
        self.assertEqual(records['travel_hours'].total_hours, Decimal(6))
        records = self.records('Labour is about 3 hours plus total billed travel is 6 hours.')
        self.assertTrue(records['labour_hours'].estimated)
        self.assertFalse(records['travel_hours'].estimated)
        records = self.records('Labour is 3 hours. The drum operates for 2 hours.')
        self.assertEqual(records['labour_hours'].total_hours, Decimal(3))

    def test_range_and_there_and_back_are_not_split_by_conjunction_boundaries(self):
        for message in ('Travel is between 2 and 6 hours.', 'Travel is 2 hours and 6 hours.'):
            with self.subTest(message=message):
                record = self.records(message)['travel_hours']
                self.assertEqual(record.status, 'ambiguous')
                self.assertIsNone(record.to_fact('RANGE'))
        record = self.records('Travel takes 6 hours there and back.')['travel_hours']
        self.assertEqual(record.total_hours, Decimal(6))

    def test_bound_fact_uses_suffix_role_and_does_not_invent_labour_hours(self):
        from metso_assistant.adaptive import request_facts

        message = 'Replace 5 quartz seals. Only need replacement labour plus 6 hours of total billed travel.'
        durations = resolve_duration_inputs([message])
        facts = bind_duration_facts(request_facts([message], 'service_cost'), [message], durations)
        typed = [fact for fact in facts if fact.get('unit') == 'hour']
        self.assertEqual([(fact['value'], fact['scope']['role']) for fact in typed],
                         [(Decimal(6), 'travel_hours')])
        self.assertEqual([fact['value'] for fact in facts if fact.get('scope', {}).get('role') == 'requested_quantity'],
                         [Decimal(5)])


if __name__ == '__main__':
    unittest.main()
