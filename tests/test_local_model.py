import json
import unittest

from metso_assistant.model import LocalModel, ModelError


class StubClient:
    def __init__(self, message, done="stop"):
        self.message, self.done = message, done
        self.payload = None

    def json(self, route, payload):
        self.payload = payload
        return {"message": {"content": self.message}, "done_reason": self.done}


class LocalModelTests(unittest.TestCase):
    def test_duplicate_fields_and_truncated_output_fail_closed(self):
        for value, done in [(' {"route":"product","route":"unsupported"}', "stop"), ('{"value":1}', "length")]:
            model = LocalModel()
            model.client = StubClient(value, done)
            with self.assertRaises(ModelError):
                model.invoke("System", [], {})

    def test_local_call_is_nonstreaming_and_does_not_request_thinking(self):
        model = LocalModel()
        model.client = StubClient('{"value":2}')
        result, _ = model.invoke("System", [{"role": "user", "content": "query"}], {}, max_tokens=20)
        self.assertEqual(result["value"], 2)
        self.assertIs(model.client.payload["stream"], False)
        self.assertIs(model.client.payload["think"], False)
        self.assertEqual(model.client.payload["options"]["num_predict"], 20)

    def test_hosted_endpoints_and_cloud_tags_are_rejected(self):
        with self.assertRaises(ValueError):
            LocalModel(endpoint="https://provider.example.invalid")
        with self.assertRaises(ValueError):
            LocalModel(model="synthetic:cloud")

    def test_unload_releases_only_the_selected_local_model_and_requires_confirmation(self):
        class Runtime:
            response = {"done": True, "done_reason": "unload"}
            def json(self, route, payload):
                self.call = (route, payload)
                return self.response
        model = LocalModel(model="synthetic:local")
        model.client = Runtime()
        self.assertTrue(model.unload()["confirmed"])
        self.assertEqual(model.client.call, ("/api/generate", {"model": "synthetic:local", "keep_alive": 0}))
        model.client.response = {"done": True, "done_reason": "stop"}
        with self.assertRaises(ModelError):
            model.unload()


if __name__ == "__main__":
    unittest.main()
