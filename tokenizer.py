from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


PERSIAN_RANGE = range(0x0600, 0x06FF + 1)


def inspect_corpus(path: str | Path) -> dict[str, int]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Real Persian corpus not found: {source}")
    total_chars = 0
    persian_chars = 0
    nonempty_lines = 0
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            nonempty_lines += 1
            total_chars += len(line)
            persian_chars += sum(ord(char) in PERSIAN_RANGE for char in line)
    if nonempty_lines == 0 or total_chars == 0:
        raise ValueError("The corpus is empty")
    persian_ratio = persian_chars / total_chars
    if persian_ratio < 0.50:
        raise ValueError(
            f"Corpus is not predominantly Persian: Persian-character ratio={persian_ratio:.3f}"
        )
    return {
        "characters": total_chars,
        "persian_characters": persian_chars,
        "nonempty_lines": nonempty_lines,
    }


def train_tokenizer(
    input_path: str | Path,
    model_prefix: str | Path = "artifacts/model/monster1_tokenizer",
    vocab_size: int = 4096,
) -> tuple[Path, Path]:
    """Train SentencePiece on a supplied, authorized Persian text corpus.

    This function never creates fallback text or synthetic training data.
    """
    source = Path(input_path)
    stats = inspect_corpus(source)
    if vocab_size != 4096:
        raise ValueError("Monster-1 requires a SentencePiece vocabulary of exactly 4096")
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(source),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        hard_vocab_limit=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        num_threads=2,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        minloglevel=2,
    )
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    if processor.vocab_size() != vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary mismatch: expected {vocab_size}, got {processor.vocab_size()}"
        )
    print(
        "Tokenizer trained from supplied corpus:",
        f"{stats['nonempty_lines']} lines, {stats['characters']} characters,",
        f"vocabulary={processor.vocab_size()}",
    )
    return model_path, vocab_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Monster-1 Persian tokenizer")
    parser.add_argument("--input", required=True, help="UTF-8 Persian corpus text file")
    parser.add_argument(
        "--model-prefix",
        default="artifacts/model/monster1_tokenizer",
        help="Output prefix; writes .model and .vocab",
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()
    train_tokenizer(args.input, args.model_prefix, args.vocab_size)


if __name__ == "__main__":
    main()
