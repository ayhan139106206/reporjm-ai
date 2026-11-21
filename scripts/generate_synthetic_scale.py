from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentencepiece import SentencePieceProcessor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "synthetic_test"
DEFAULT_TOKENIZER = ROOT / "artifacts" / "synthetic_test" / "tokenizer.model"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "synthetic_scale"
DEFAULT_TRAIN_TOKENS = 110_458_880
DEFAULT_HELDOUT_TOKENS = 1_000_000
SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_samples(path: Path, processor: SentencePieceProcessor) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            encoded = processor.encode(text, out_type=int)
            if not encoded:
                raise ValueError(f"Synthetic source contains a line with no tokenizer output: {path}")
            samples.append((text, len(encoded) + 1))  # count EOS as the training gate does
    if not samples:
        raise ValueError(f"Synthetic source split is empty: {path}")
    return samples


def _write_repeated_split(
    output_path: Path,
    samples: list[tuple[str, int]],
    target_tokens: int,
    seed: int,
    split: str,
    progress_interval: int,
) -> dict[str, int]:
    if target_tokens < 1:
        raise ValueError(f"Target token count for {split} must be positive")
    ordered = list(samples)
    random.Random(seed ^ (SPLITS.index(split) + 1) * 0x9E3779B1).shuffle(ordered)
    written_tokens = 0
    written_lines = 0
    next_progress = progress_interval
    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024, newline="\n") as stream:
        while written_tokens < target_tokens:
            for text, token_count in ordered:
                stream.write(text)
                stream.write("\n")
                written_tokens += token_count
                written_lines += 1
                if written_tokens >= next_progress:
                    print(
                        f"{split}: {written_tokens:,}/{target_tokens:,} tokens; {written_lines:,} rows",
                        flush=True,
                    )
                    next_progress += progress_interval
                if written_tokens >= target_tokens:
                    break
    return {"tokens": written_tokens, "rows": written_lines, "target_tokens": target_tokens}


def generate_corpus(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    train_tokens: int = DEFAULT_TRAIN_TOKENS,
    validation_tokens: int = DEFAULT_HELDOUT_TOKENS,
    test_tokens: int = DEFAULT_HELDOUT_TOKENS,
    seed: int = 20261121,
    progress_interval: int = 5_000_000,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    source_dir = source_dir.resolve()
    tokenizer_path = tokenizer_path.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output must not overwrite or contain the source synthetic_test corpus")
    if progress_interval < 1:
        raise ValueError("progress_interval must be positive")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Synthetic tokenizer not found: {tokenizer_path}")

    source_manifest_path = source_dir / "corpus_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic source manifest not found: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("mode") != "synthetic_test" or source_manifest.get("is_synthetic") is not True:
        raise ValueError("Refusing source data not explicitly marked as synthetic_test")

    expected_paths = [output_dir / f"{split}.txt" for split in SPLITS]
    expected_paths.append(output_dir / "corpus_manifest.json")
    existing = [str(path) for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing generated files: " + ", ".join(existing))

    processor = SentencePieceProcessor(model_file=str(tokenizer_path))
    if processor.vocab_size() != 4096:
        raise ValueError(f"Expected the existing 4,096-piece synthetic test tokenizer; got {processor.vocab_size()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {"train": train_tokens, "validation": validation_tokens, "test": test_tokens}
    split_counts: dict[str, dict[str, int]] = {}
    hashes: dict[str, str] = {}
    source_row_counts: dict[str, int] = {}
    repetition_counts: dict[str, int] = {}
    for split in SPLITS:
        source_path = source_dir / f"{split}.txt"
        samples = _read_samples(source_path, processor)
        source_row_counts[split] = len(samples)
        result = _write_repeated_split(
            output_dir / f"{split}.txt",
            samples,
            targets[split],
            seed,
            split,
            progress_interval,
        )
        split_counts[split] = result
        repetition_counts[split] = result["rows"]
        hashes[f"sha256_{split}"] = _sha256(output_dir / f"{split}.txt")

    manifest: dict[str, Any] = {
        "project": "Monster-1",
        "dataset_name": "Monster-1 synthetic padding corpus; repeated smoke-test examples",
        "manifest_version": 1,
        "mode": "synthetic_padding",
        "is_synthetic": True,
        "synthetic": True,
        "real_data": False,
        "training_use_approved": False,
        "pretrained_model_used": False,
        "external_corpus_used": False,
        "external_api_used": False,
        "generation_method": (
            "Deterministically shuffles and repeats existing synthetic_test lines until each requested "
            "SentencePiece token count (including EOS) is met. It creates no new factual knowledge, "
            "does not call an external model or API, and is not suitable for real training."
        ),
        "generation_seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": "synthetic_test",
        "source_corpus_manifest_sha256": _sha256(source_manifest_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "tokenizer_vocab_size": processor.vocab_size(),
        "target_token_counts_sentencepiece_including_eos": targets,
        "actual_token_counts_sentencepiece_including_eos": {
            split: split_counts[split]["tokens"] for split in SPLITS
        },
        "row_counts": {split: split_counts[split]["rows"] for split in SPLITS},
        "source_unique_row_counts": source_row_counts,
        "repeated_row_counts": repetition_counts,
        **hashes,
        "warning": (
            "This is repetitive synthetic padding, not 110,458,880 tokens of real Persian knowledge. "
            "It must never be presented as real data, used to pass the production training gate, or used "
            "to claim model quality."
        ),
    }
    manifest_path = output_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a large, local, repetitive synthetic corpus for storage/pipeline stress tests only; "
            "it cannot qualify for real-data training."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--validation-tokens", type=int, default=DEFAULT_HELDOUT_TOKENS)
    parser.add_argument("--test-tokens", type=int, default=DEFAULT_HELDOUT_TOKENS)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--progress-interval", type=int, default=5_000_000)
    args = parser.parse_args()
    try:
        manifest = generate_corpus(
            output_dir=args.output_dir,
            source_dir=args.source_dir,
            tokenizer_path=args.tokenizer,
            train_tokens=args.train_tokens,
            validation_tokens=args.validation_tokens,
            test_tokens=args.test_tokens,
            seed=args.seed,
            progress_interval=args.progress_interval,
        )
    except Exception as exc:
        print(f"Synthetic corpus generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
