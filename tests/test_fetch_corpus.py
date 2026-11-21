import tempfile
import unittest
from pathlib import Path
import zipfile
from unittest.mock import patch

from scripts.fetch_corpus import _extract, download_corpus


class CorpusFetchTests(unittest.TestCase):
    def test_rejects_non_https_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "direct HTTPS"):
                download_corpus("http://example.invalid/corpus.zip", "0" * 64, Path(directory))

    def test_rejects_bad_checksum_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                download_corpus("https://example.invalid/corpus.zip", "bad", Path(directory))

    def test_rejects_duplicate_corpus_members(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("train.txt", "first")
                output.writestr("nested/train.txt", "second")
            with self.assertRaisesRegex(ValueError, "Duplicate corpus archive member"):
                _extract(archive, Path(directory) / "out")

    def test_rejects_archive_exceeding_extracted_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "large.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("train.txt", "12345")
            with patch("scripts.fetch_corpus.MAX_EXTRACTED_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "safety limit"):
                    _extract(archive, Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
