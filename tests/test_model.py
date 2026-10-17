"""Synthetic regressions for the current local transport and routing boundary.

No supplied Metso documents, local model, HTTP requests, or paid API are needed.
These tests verify the application boundary; they do not score an LLM's answers.
"""

from copy import deepcopy
import unittest
from unittest.mock import patch

from metso_assistant.model import LocalModel, OllamaClient
from metso_assistant.routing import dispatch_arguments, validate_route


def known_fields(**updates):
    """Synthetic input using the public router protocol, independent of its schema."""
    fields = {
        "equipment": None,
        "country": None,
        "site": None,
        "task": None,
        "cloth_count": None,
        "cloths_supplied": None,
        "round_trip_travel_hours": None,
        "iron_pct": None,
        "silica_pct": None,
    }
    fields.update(updates)
    return fields


def routed(route, query="Synthetic request", **fields):
    return {"route": route, "query": query, "known_fields": known_fields(**fields)}


class RouterBoundaryTests(unittest.TestCase):
    def test_typed_inputs_reject_strings_and_boolean_quantities(self):
        # A value hidden in free text or the wrong scalar type cannot reach a
        # numeric service/statistics calculation as a valid structured argument.
        invalid = [
            {"cloth_count": True},
            {"cloth_count": 2.5},
            {"cloths_supplied": "false"},
            {"round_trip_travel_hours": "3.5"},
            {"iron_pct": "66.7"},
            {"silica_pct": float("nan")},
            {"iron_pct": 101},
        ]
        for fields in invalid:
            with self.subTest(fields=fields):
                self.assertTrue(validate_route(routed("service_cost", **fields)))

    def test_omitted_typed_slot_is_invalid_even_if_mentioned_in_task(self):
        response = routed("service_cost", task="Replace seven filter cloths")
        del response["known_fields"]["cloth_count"]
        self.assertTrue(validate_route(response))

    def test_product_dispatch_isolates_stale_service_context(self):
        response = routed(
            "product", query="Explain the synthetic pump's design features",
            country="Exampleland", equipment="Example Press", task="Replace cloths",
            cloth_count=7, cloths_supplied=True, round_trip_travel_hours=9,
        )
        original = deepcopy(response)
        self.assertEqual(validate_route(response), [])
        self.assertEqual(
            dispatch_arguments(response),
            {"query": "Explain the synthetic pump's design features"},
        )
        self.assertEqual(response, original, "Dispatch must retain the raw model response")
        self.assertEqual(response["route"], "product", "Dispatch must not silently reroute")

    def test_flotation_dispatch_uses_only_readings(self):
        response = routed(
            "flotation", country="Exampleland", equipment="Example Press",
            task="Old cloth replacement", cloth_count=7,
            iron_pct=66.7, silica_pct=1.9,
        )
        self.assertEqual(dispatch_arguments(response), {"iron_pct": 66.7, "silica_pct": 1.9})
        response["known_fields"]["silica_pct"] = None
        self.assertEqual(dispatch_arguments(response), {"iron_pct": 66.7, "silica_pct": None})

    def test_false_parts_supplied_and_zero_travel_are_not_missing(self):
        response = routed(
            "service_cost", country="Exampleland", equipment="Example Press",
            task="Replace filter cloths", cloth_count=7,
            cloths_supplied=False, round_trip_travel_hours=0,
        )
        self.assertEqual(validate_route(response), [])
        arguments = dispatch_arguments(response)
        self.assertIs(arguments["cloths_supplied"], False)
        self.assertEqual(arguments["round_trip_travel_hours"], 0)

    def test_noncloth_service_dispatch_excludes_cloth_inputs(self):
        response = routed(
            "service_cost", country="Exampleland", equipment="Example Pump",
            task="Motor inspection",
        )
        arguments = dispatch_arguments(response)
        self.assertIsNone(arguments["round_trip_travel_hours"])
        self.assertNotIn("cloth_count", arguments)
        self.assertNotIn("cloths_supplied", arguments)
        self.assertNotIn("iron_pct", arguments)
        self.assertNotIn("silica_pct", arguments)
        self.assertNotIn("site", arguments, "An unknown optional site must not block a rough estimate")

    def test_cloth_service_dispatch_preserves_unknown_quantity_parts_and_travel(self):
        response = routed(
            "service_cost", country="Exampleland", equipment="Example Press",
            task="Replace filter cloths",
        )
        arguments = dispatch_arguments(response)
        for name in ("cloth_count", "cloths_supplied", "round_trip_travel_hours"):
            self.assertIn(name, arguments)
            self.assertIsNone(arguments[name])


class LocalOnlyTests(unittest.TestCase):
    def test_nonlocal_or_credentialed_endpoints_rejected_before_any_http(self):
        endpoints = [
            "https://api.example.invalid",
            "http://192.0.2.1:11434",
            "http://user:synthetic-token@localhost:11434",
            "http://127.0.0.1:11434?token=synthetic",
            "http://localhost:11434/api",
        ]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    OllamaClient(endpoint, timeout=1)

    def test_loopback_configuration_needs_no_network(self):
        for endpoint in ("http://127.0.0.1:11434", "http://localhost:11435", "http://[::1]:11434"):
            with self.subTest(endpoint=endpoint):
                client = OllamaClient(endpoint, timeout=1)
                self.assertEqual(client.endpoint, endpoint)

    def test_cloud_tag_rejected_before_a_client_request(self):
        with patch("metso_assistant.model.OllamaClient") as client:
            with self.assertRaises(ValueError):
                LocalModel(model="synthetic:cloud")
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
