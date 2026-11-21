import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sentencepiece import SentencePieceProcessor

from scripts.generate_synthetic_data import generate_dataset
from scripts.generate_synthetic_scale import SPLITS, generate_corpus
from scripts.run_synthetic_test import _train_synthetic_tokenizer
from train import preflight


class SyntheticScaleTests(unittest.TestCase):
    def test_large_corpus_generator_is_local_explicitly_synthetic_and_meets_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            generate_dataset(source_dir, seed=31)
            tokenizer_prefix = root / "tokenizer"
            _train_synthetic_tokenizer(source_dir / "train.txt", tokenizer_prefix, 4096)
            tokenizer_path = tokenizer_prefix.with_suffix(".model")
            output_dir = root / "scale"

            manifest = generate_corpus(
                output_dir=output_dir,
                source_dir=source_dir,
                tokenizer_path=tokenizer_path,
                train_tokens=1200,
                validation_tokens=300,
                test_tokens=300,
                seed=31,
                progress_interval=10_000,
            )

            saved = json.loads((output_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["is_synthetic"])
            self.assertTrue(saved["synthetic"])
            self.assertFalse(saved["real_data"])
            self.assertFalse(saved["training_use_approved"])
            self.assertFalse(saved["external_api_used"])
            self.assertEqual(saved["mode"], "synthetic_padding")
            self.assertIn("repeats existing", saved["generation_method"])

            processor = SentencePieceProcessor(model_file=str(tokenizer_path))
            for split, target in (("train", 1200), ("validation", 300), ("test", 300)):
                path = output_dir / f"{split}.txt"
                actual = sum(
                    len(processor.encode(line.strip(), out_type=int)) + 1
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                self.assertGreaterEqual(actual, target)
                self.assertEqual(actual, saved["actual_token_counts_sentencepiece_including_eos"][split])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, saved[f"sha256_{split}"])
            self.assertEqual(manifest["target_token_counts_sentencepiece_including_eos"]["train"], 1200)
            self.assertEqual(set(SPLITS), set(manifest["row_counts"]))

    def test_training_preflight_rejects_synthetic_manifest_before_tokenizer_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for split in ("train", "validation", "test"):
                path = root / f"{split}.txt"
                path.write_text("یک نمونه ساختگی برای آزمایش.\n", encoding="utf-8")
                files[split] = path
            manifest_path = root / "corpus_manifest.json"
            manifest = {
                "source_name": "Fabricated synthetic source",
                "source_url": "https://example.invalid/synthetic",
                "license": "CC0-1.0",
                "license_url": "https://example.invalid/license",
                "language": "fa",
                "collection_date": "2026-11-21",
                "rights_to_train": True,
                "privacy_reviewed": True,
                "attribution": "Synthetic test data",
                "is_synthetic": True,
                "synthetic": True,
                "real_data": False,
                "mode": "synthetic_padding",
                **{
                    f"sha256_{split}": hashlib.sha256(files[split].read_bytes()).hexdigest()
                    for split in files
                },
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            tokenizer_path = root / "must-not-be-created.model"
            report = preflight(
                train_file=files["train"],
                validation_file=files["validation"],
                test_file=files["test"],
                manifest_file=manifest_path,
                tokenizer_file=tokenizer_path,
                config_file=Path("config.json"),
                report_file=root / "preflight.json",
            )

            self.assertEqual(report["status"], "blocked_invalid_manifest")
            self.assertFalse(report["training_ready"])
            self.assertFalse(report["training_performed"])
            self.assertTrue(any("Synthetic data is never eligible" in issue for issue in report["corpus_manifest_issues"]))
            self.assertFalse(tokenizer_path.exists())


if __name__ == "__main__":
    unittest.main()
