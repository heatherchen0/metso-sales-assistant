"""Source-free topic-order fixture integrity; no runtime models or source data."""
import ast
import inspect
import itertools
import json
from pathlib import Path
import unittest

from metso_assistant import evaluate


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'benchmarks/regression-cases.json'


def assignment_prompts():
    # Keep the three demonstration questions independent of the private brief.
    # Exact text matters: these checks protect every ordering of the same turns.
    return {
        'product': 'We have Larox pressure filters but performance drops whenever our feed grade changes',
        'service': 'A customer in Chile needs 12 filter cloths replaced on a Larox PF-DS tower press. What should I budget for the service visit?',
        'process': "The customer is getting 3.1% silica and 64.5% iron in their concentrate. They think that's normal. What do you think?",
    }


class TopicOrderCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads(PATH.read_text(encoding='utf-8'))
        cls.cases=[case for case in cls.fixture['cases'] if case['id'].startswith('topic-order-')]
        cls.prompts=assignment_prompts()
        cls.demo_cases=[case for case in cls.cases if case['id'].startswith('topic-order-')
                        and 'tankcell' not in case['id'] and 'pid-level' not in case['id']]

    def test_case_ids_counts_and_every_exact_assignment_permutation(self):
        self.assertEqual(len(self.cases),14)
        self.assertEqual(sum(len(case['messages']) for case in self.cases),46)
        self.assertEqual(len({case['id'] for case in self.cases}),len(self.cases))
        lookup={text:kind for kind,text in self.prompts.items()}
        for reply in (False,True):
            group=[case for case in self.demo_cases if case['id'].endswith('-with-service-reply')==reply]
            self.assertEqual(len(group),6)
            orders={tuple(lookup[message] for message in case['messages'] if message!='8') for case in group}
            self.assertEqual(orders,set(itertools.permutations(('product','service','process'))))
            self.assertEqual({len(case['messages']) for case in group},{4 if reply else 3})

    def test_reply_is_immediately_after_an_actual_service_question_and_is_not_a_new_request(self):
        for case in self.demo_cases:
            with self.subTest(case=case['id']):
                messages=case['messages']; service=messages.index(self.prompts['service'])
                self.assertEqual(case['requires_clarification_turns'],[service])
                self.assertEqual(case['requires_unpriced_travel_turns'],[service])
                self.assertEqual(case['forbidden_priced_duration_roles_by_turn'],{str(service):['travel_hours']})
                expected_boundaries=[i for i,message in enumerate(messages) if message!='8']
                self.assertEqual(case['requires_new_request_boundary_turns'],expected_boundaries)
                if case['id'].endswith('-with-service-reply'):
                    self.assertEqual(messages[service+1],'8')
                    self.assertEqual(case['expected_routes'][service+1],'service_cost')
                    self.assertEqual(case['requires_travel_range_turns'],[service+1])
                    self.assertNotIn(service+1,case['requires_new_request_boundary_turns'])
                else:
                    self.assertNotIn('8',messages)
                    self.assertNotIn('requires_travel_range_turns',case)

    def test_routes_sources_calculations_and_input_measurements_match_each_turn(self):
        routes={self.prompts['product']:'product',self.prompts['service']:'service_cost',
                self.prompts['process']:'flotation','8':'service_cost'}
        for case in self.demo_cases:
            with self.subTest(case=case['id']):
                expected=[routes[message] for message in case['messages']]
                self.assertEqual(case['expected_routes'],expected)
                self.assertEqual(case['requires_source_turns'],list(range(len(expected))))
                self.assertEqual(case['requires_calculation_turns'],[i for i,route in enumerate(expected) if route!='product'])
                product=expected.index('product');process=expected.index('flotation')
                self.assertEqual(case['requires_recommendation_turns'],[product])
                self.assertEqual(case['required_product_tasks_by_turn'],{str(product):['recommend']})
                self.assertEqual(case['expected_process_readings_by_turn'],
                    {str(process):{'% Silica Concentrate':3.1,'% Iron Concentrate':64.5}})
                self.assertEqual(case['expected_travel_countries'],['Chile' if route=='service_cost' else None for route in expected])
                self.assertEqual(case['must_not_claim_full_estimate_turns'],[i for i,route in enumerate(expected) if route=='service_cost'])

    def test_both_same_domain_orders_are_standalone_needs_requiring_fresh_recommendations(self):
        fine='We\u2019re losing valuable particles below 20 microns to tailings in our existing TankCell circuit.'
        pid='Our flotation-cell pulp levels fluctuate whenever upstream conditions change, despite PID control'
        group=[case for case in self.cases if case not in self.demo_cases]
        self.assertEqual({tuple(case['messages']) for case in group},{(fine,pid),(pid,fine)})
        for case in group:
            self.assertEqual(case['expected_routes'],['product','product'])
            self.assertEqual(case['requires_new_request_boundary_turns'],[0,1])
            self.assertEqual(case['requires_recommendation_turns'],[0,1])
            self.assertEqual(case['required_product_tasks_by_turn'],{'0':['recommend'],'1':['recommend']})

    def test_messages_do_not_help_the_router_with_explicit_topic_switch_markers(self):
        for case in self.cases:
            for message in case['messages']:
                self.assertNotRegex(message.lower(),r'\b(?:new|different|another|instead|now)\b|separate\s+topic|change\s+the\s+topic')

    def test_contracts_are_supported_by_the_existing_evaluator_and_indices_are_in_range(self):
        tree=ast.parse(inspect.getsource(evaluate))
        supported={'id','priority','messages'}
        for node in ast.walk(tree):
            # A few evaluator contracts are dispatched through a loop whose
            # keys are literal strings rather than direct case.get arguments.
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith(('_turns','_by_turn')):
                supported.add(node.value)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='get' and isinstance(node.func.value,ast.Name) and node.func.value.id=='case':
                if node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):supported.add(node.args[0].value)
            if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='case' and isinstance(node.slice,ast.Constant):
                supported.add(node.slice.value)
        for case in self.cases:
            with self.subTest(case=case['id']):
                self.assertFalse(set(case)-supported,'An unrecognized contract would be silently ignored')
                n=len(case['messages'])
                self.assertEqual(len(case['expected_routes']),n)
                self.assertEqual(case['allowed_statuses'],[['answered'] for _ in range(n)])
                for key,value in case.items():
                    if key.endswith('_turns'):
                        self.assertTrue(all(type(index) is int and 0<=index<n for index in value))
                    elif key.endswith('_by_turn'):
                        self.assertTrue(all(index.isdigit() and 0<=int(index)<n for index in value))

    def test_topic_order_subset_has_no_source_answer_price_or_expected_candidate_catalogue(self):
        text=json.dumps(self.cases)
        self.assertNotRegex(text,r'\b(?:USD|EUR|CAD|AUD|SEK|ZAR|FSR-\d+)\b')
        self.assertNotRegex(text,r'"(?:expected_answer|answer_text|expected_product|expected_price|unit_price|historical_mean|required_product_subjects_by_turn)"')
        self.assertEqual(set(self.fixture),{'schema_version','purpose','cases'})


if __name__=='__main__':unittest.main()
