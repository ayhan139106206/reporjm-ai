import json
import tempfile
import unittest
from pathlib import Path

from model import Monster1, count_parameters, load_config
from train import MIN_TOKENS_PER_PARAMETER, preflight


class TrainingGateTests(unittest.TestCase):
    def test_real_data_token_threshold_remains_unchanged(self):
        config = load_config("config.json")
        parameters = count_parameters(Monster1.from_config(config))
        self.assertEqual(parameters, 5_522_944)
        self.assertEqual(MIN_TOKENS_PER_PARAMETER, 20)
        self.assertEqual(parameters * MIN_TOKENS_PER_PARAMETER, 110_458_880)

    def test_missing_corpus_stops_before_tokenizer_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "reports" / "preflight.json"
            output_dir = root / "artifacts"
            report = preflight(
                train_file=root / "data" / "train.txt",
                validation_file=root / "data" / "validation.txt",
                test_file=root / "data" / "test.txt",
                manifest_file=root / "data" / "corpus_manifest.json",
                tokenizer_file=output_dir / "monster1_tokenizer.model",
                config_file=Path("config.json"),
                report_file=report_path,
            )
            self.assertFalse(report["training_ready"])
            self.assertFalse(report["training_performed"])
            self.assertEqual(report["status"], "blocked_missing_data")
            self.assertTrue(report_path.is_file())
            self.assertFalse((output_dir / "monster1.pt").exists())
            self.assertFalse((output_dir / "monster1_tokenizer.model").exists())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("train", saved["required_data"])
            self.assertEqual(len(saved["missing_files"]), 4)
            self.assertIn("corpus_manifest.json", saved["missing_files"][-1])


if __name__ == "__main__":
    unittest.main()
