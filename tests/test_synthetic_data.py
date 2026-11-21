import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_synthetic_data import SPLIT_NAMES, TOPICS, generate_dataset


class SyntheticDataTests(unittest.TestCase):
    def test_deterministic_splits_are_unique_and_cover_required_topics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first_manifest = generate_dataset(first, seed=17)
            second_manifest = generate_dataset(second, seed=17)

            for split in SPLIT_NAMES:
                self.assertEqual(
                    (first / f"{split}.txt").read_bytes(),
                    (second / f"{split}.txt").read_bytes(),
                )
                lines = (first / f"{split}.txt").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), len(set(lines)))
                self.assertGreater(len(lines), 0)
            self.assertEqual(first_manifest["sample_counts"], second_manifest["sample_counts"])
            self.assertEqual(set(first_manifest["topic_sample_counts"]), set(TOPICS))
            self.assertTrue(all(set(first_manifest["split_topic_sample_counts"][split]) == set(TOPICS) for split in SPLIT_NAMES))

            all_lines = [
                line
                for split in SPLIT_NAMES
                for line in (first / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(all_lines), len(set(all_lines)))

    def test_manifest_truthfully_marks_dataset_as_synthetic_and_hashes_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = generate_dataset(root, seed=23)
            saved = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["is_synthetic"])
            self.assertTrue(saved["synthetic"])
            self.assertFalse(saved["real_data"])
            self.assertFalse(saved["external_api_used"])
            for split in SPLIT_NAMES:
                self.assertEqual(len(saved[f"sha256_{split}"].strip()), 64)
                self.assertGreater(saved["sample_counts"][split], 0)
            self.assertEqual(manifest["mode"], "synthetic_test")
            self.assertIn("not a real", manifest["quality_note"])


if __name__ == "__main__":
    unittest.main()
