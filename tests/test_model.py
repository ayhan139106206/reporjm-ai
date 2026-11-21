import unittest

import torch

from model import Monster1, count_parameters, load_config


class Monster1ArchitectureTests(unittest.TestCase):
    def test_config_and_parameter_count_match_checkpoint_contract(self):
        config = load_config("config.json")
        model = Monster1.from_config(config)
        self.assertEqual(config["vocab_size"], 4096)
        self.assertEqual(model.vocab_size, 4096)
        self.assertEqual(count_parameters(model), 5_522_944)

    def test_forward_shape_and_causal_mask(self):
        torch.manual_seed(7)  # deterministic test initialization only; no training occurs
        model = Monster1(vocab=64, dim=16, layers=1, heads=2, max_seq_len=16).eval()
        first = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        changed_future = torch.tensor([[2, 3, 4, 9]], dtype=torch.long)
        with torch.inference_mode():
            output_a = model(first)
            output_b = model(changed_future)
        self.assertEqual(tuple(output_a.shape), (1, 4, 64))
        self.assertTrue(torch.allclose(output_a[:, :3], output_b[:, :3], atol=1e-6, rtol=1e-6))

    def test_rejects_too_long_sequence(self):
        model = Monster1(vocab=32, dim=16, layers=1, heads=2, max_seq_len=4)
        with self.assertRaisesRegex(ValueError, "exceeds max_sequence_length"):
            model(torch.ones((1, 5), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
