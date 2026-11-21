import tempfile
import unittest
from pathlib import Path

import torch

from export_onnx import export_graph, verify_runtime
from model import Monster1


class OnnxMechanicsTests(unittest.TestCase):
    def test_dynamic_lengths_parity_and_token_loop(self):
        # Test-only initialized fixture: no corpus, optimizer, training, or final artifact.
        torch.manual_seed(11)
        model = Monster1(vocab=64, dim=16, layers=1, heads=2, max_seq_len=32, dropout=0.0).eval()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "test_only.onnx"
            export_graph(model, output, example_length=8)
            report = verify_runtime(
                model,
                output,
                vocab_size=64,
                test_lengths=(1, 2, 8, 16, 32),
                generation_steps=4,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual([row["sequence_length"] for row in report["sequence_tests"]], [1, 2, 8, 16, 32])
            self.assertEqual(report["batch_tests"][0]["batch_size"], 2)
            self.assertEqual(report["onnx_input"]["shape"], ["batch", "sequence"])
            self.assertEqual(report["generation_steps"], 4)
            self.assertFalse((Path(directory) / "model_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
