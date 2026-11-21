import unittest

from fastapi.testclient import TestClient

import api


class ApiReadinessTests(unittest.TestCase):
    def setUp(self):
        api._runtime = None
        api._runtime_checked = False
        api._runtime_error = "No completed, verified Monster-1 training artifact is installed."
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


if __name__ == "__main__":
    unittest.main()
