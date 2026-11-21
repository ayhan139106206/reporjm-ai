import unittest

import numpy as np
from fastapi.testclient import TestClient

import api


class SyntheticDecoderTests(unittest.TestCase):
    def test_normalizes_persian_and_arabic_decimal_digits(self):
        self.assertEqual(
            api.normalize_synthetic_prompt("جمع ۱۲۳ و ٤٥٦"),
            "جمع 123 و 456",
        )

    def test_skips_blank_piece_and_stops_at_eos(self):
        class FakeTokenizer:
            def encode(self, prompt, out_type=int):
                return [7]

            def unk_id(self):
                return 0

            def decode(self, token_ids):
                return "".join({5: "پاسخ"}.get(token_id, "") for token_id in token_ids)

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def run(self, output_names, inputs):
                sequence = inputs["input_ids"].shape[1]
                logits = np.zeros((1, sequence, 8), dtype=np.float32)
                if self.calls == 0:
                    logits[0, -1, 2] = 200.0  # immediate EOS must not produce an empty reply
                    logits[0, -1, 4] = 100.0  # whitespace-only piece
                    logits[0, -1, 5] = 10.0   # visible model token
                else:
                    logits[0, -1, 2] = 100.0  # EOS should remain available
                self.calls += 1
                return [logits]

        runtime = api.SyntheticDemoRuntime.__new__(api.SyntheticDemoRuntime)
        runtime.tokenizer = FakeTokenizer()
        runtime.forbidden_ids = {0, 1, 3}
        runtime.unrenderable_ids = {0, 1, 3, 4}
        runtime.eos_id = 2
        runtime.session = FakeSession()
        runtime.input_name = "input_ids"
        runtime.output_name = "logits"
        runtime.max_sequence_length = 16

        self.assertEqual(runtime.generate("سلام", max_new_tokens=8), "پاسخ")
        self.assertEqual(runtime.session.calls, 2)


class ApiReadinessTests(unittest.TestCase):
    def setUp(self):
        api._runtime = None
        api._runtime_checked = False
        api._runtime_error = "No completed, verified Monster-1 training artifact is installed."
        api._demo_runtime = None
        api._demo_runtime_checked = True
        api._demo_runtime_error = "No verified synthetic demo artifacts are installed."
        self.client = TestClient(api.app)

    def test_health_and_model_do_not_claim_readiness(self):
        health = self.client.get("/v1/health")
        model = self.client.get("/v1/model")
        self.assertEqual(health.status_code, 200)
        self.assertFalse(health.json()["model_ready"])
        self.assertFalse(health.json()["external_model_api_used"])
        self.assertEqual(health.json()["inference_backend"], "not_configured")
        self.assertEqual(model.status_code, 200)
        self.assertFalse(model.json()["engine_ready"])
        self.assertFalse(model.json()["external_model_api_used"])
        self.assertEqual(model.json()["inference_backend"], "not_configured")
        self.assertEqual(model.json()["max_generation_tokens_per_request"], 128)
        self.assertEqual(model.json()["training_status"], "not_trained_or_unverified")
        self.assertEqual(model.json()["parameters"], 5_522_944)

    def test_chat_is_unavailable_without_verified_artifacts(self):
        response = self.client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "سلام"}]},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["error"], "model_not_ready")

    def test_demo_health_is_explicitly_synthetic_and_never_engine_ready(self):
        response = self.client.get("/v1/demo/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["is_synthetic"])
        self.assertTrue(body["experimental"])
        self.assertTrue(body["not_for_production"])
        self.assertFalse(body["engine_ready"])
        self.assertFalse(body["external_model_api_used"])
        self.assertIn("poor or meaningless", body["warning"])

    def test_demo_chat_is_unavailable_without_synthetic_artifacts(self):
        response = self.client.post(
            "/v1/demo/chat",
            json={"messages": [{"role": "user", "content": "سلام"}]},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["error"], "synthetic_demo_unavailable")

    def test_demo_chat_response_is_labeled_experimental(self):
        class FakeDemoRuntime:
            def generate(self, prompt, max_new_tokens):
                self.prompt = prompt
                self.max_new_tokens = max_new_tokens
                return "پاسخ آزمایشی"

        api._demo_runtime = FakeDemoRuntime()
        api._demo_runtime_checked = True
        response = self.client.post(
            "/v1/demo/chat",
            json={
                "messages": [{"role": "user", "content": "سلام"}],
                "max_new_tokens": 12,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["reply"], "پاسخ آزمایشی")
        self.assertEqual(body["generation_status"], "generated")
        self.assertTrue(body["reply_is_model_output"])

        class EmptyDemoRuntime:
            def generate(self, prompt, max_new_tokens):
                return "   "

        api._demo_runtime = EmptyDemoRuntime()
        empty_response = self.client.post(
            "/v1/demo/chat",
            json={"messages": [{"role": "user", "content": "سلام"}]},
        )
        self.assertEqual(empty_response.status_code, 200)
        empty_body = empty_response.json()
        self.assertEqual(empty_body["generation_status"], "empty_model_output")
        self.assertFalse(empty_body["reply_is_model_output"])
        self.assertIn("متن قابل‌نمایشی تولید نکرد", empty_body["reply"])
        self.assertTrue(body["is_synthetic"])
        self.assertTrue(body["experimental"])
        self.assertTrue(body["not_for_production"])
        self.assertFalse(body["engine_ready"])
        self.assertFalse(body["external_model_api_used"])


if __name__ == "__main__":
    unittest.main()
