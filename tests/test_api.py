import unittest

from fastapi.testclient import TestClient

import api


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
