import unittest

from metso_assistant.ingest import comparable_fields, fact_key


def fact(value, label="Height of filter"):
    return {"fact_id": "example", "parameter": label, "configuration": "10",
            "unit": "mm", "value_kind": "scalar", "status": "reported",
            "value": value, "issues": [], "review_status": "pending"}


class IngestionComparisonTests(unittest.TestCase):
    def test_geometry_disagreement_blocks_even_if_candidate_looks_valid(self):
        candidate = {"source": {"sha": "a" * 64, "file": "sample.pdf", "page": 1}, "facts": [fact(748)]}
        comparison = comparable_fields(candidate, {"facts": [fact(7480)]}, independent_geometry=True)
        self.assertEqual(comparison["matching_fields"], 0)
        self.assertEqual(candidate["facts"][0]["issues"][0]["severity"], "blocker")
        self.assertEqual(comparison["conflicts"][0]["alternate"]["value"], 7480)
        self.assertEqual(candidate["facts"][0]["value"], 748)

    def test_equivalent_labels_match_but_qualifiers_remain_distinct(self):
        self.assertEqual(fact_key(fact(1, "Height of filter / 33 mm")), fact_key(fact(1, "Height / 33 mm")))
        self.assertNotEqual(fact_key(fact(1, "Height / 33 mm")), fact_key(fact(1, "Weight / 33 mm")))

    def test_duplicate_alternate_fields_do_not_establish_consensus(self):
        candidate = {"source": {"sha": "a" * 64}, "facts": [fact(1)]}
        comparison = comparable_fields(candidate, {"facts": [fact(1), fact(1)]}, independent_geometry=False)
        self.assertEqual(comparison["comparable_fields"], 0)
        self.assertEqual(comparison["matching_fact_ids"], [])


if __name__ == "__main__":
    unittest.main()
