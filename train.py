from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

from model import Monster1, count_parameters, load_config
from tokenizer import train_tokenizer


DEFAULT_TRAIN = Path("data/train.txt")
DEFAULT_VALIDATION = Path("data/validation.txt")
DEFAULT_TEST = Path("data/test.txt")
DEFAULT_MANIFEST = Path("data/corpus_manifest.json")
DEFAULT_TOKENIZER = Path("artifacts/model/monster1_tokenizer.model")
DEFAULT_OUTPUT = Path("artifacts/model")
MIN_TOKENS_PER_PARAMETER = 20
MIN_HELDOUT_TOKENS = 1_000_000
REQUIRED_MANIFEST_FIELDS = (
    "source_name",
    "source_url",
    "license",
    "license_url",
    "language",
    "collection_date",
    "rights_to_train",
    "privacy_reviewed",
    "attribution",
    "sha256_train",
    "sha256_validation",
    "sha256_test",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_sentencepiece_tokens(path: str | Path, processor: SentencePieceProcessor) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if text:
                count += len(processor.encode(text, out_type=int)) + 1  # EOS
    return count


def _manifest_issues(manifest_path: Path, files: dict[str, Path]) -> list[str]:
    if not manifest_path.is_file():
        return [f"Missing corpus provenance manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Invalid corpus manifest JSON: {exc}"]
    if (
        manifest.get("is_synthetic") is True
        or manifest.get("synthetic") is True
        or manifest.get("real_data") is False
        or manifest.get("mode") in ("synthetic_test", "synthetic_padding")
    ):
        return [
            "Synthetic data is never eligible for production training; provide a genuinely sourced "
            "corpus with rights and provenance instead."
        ]
    issues = [
        f"corpus_manifest.json is missing or has an empty '{key}'"
        for key in REQUIRED_MANIFEST_FIELDS
        if key not in manifest or (key not in ("rights_to_train", "privacy_reviewed") and not manifest[key])
    ]
    if not issues and manifest.get("language") not in ("fa", "Persian", "Farsi"):
        issues.append("corpus_manifest.json must identify the corpus language as Persian/fa")
    if manifest.get("rights_to_train") is not True:
        issues.append("corpus_manifest.json must explicitly set rights_to_train=true")
    if manifest.get("privacy_reviewed") is not True:
        issues.append("corpus_manifest.json must explicitly set privacy_reviewed=true")
    for key in ("source_url", "license_url"):
        if manifest.get(key) and not str(manifest[key]).startswith("https://"):
            issues.append(f"corpus_manifest.json '{key}' must be an HTTPS URL")
    for label, path in files.items():
        hash_key = f"sha256_{label}"
        if path.is_file() and hash_key in manifest:
            actual = sha256_file(path)
            if str(manifest[hash_key]).lower() != actual:
                issues.append(f"SHA-256 mismatch for {path}: manifest does not match file")
    return issues


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight(
    train_file: Path = DEFAULT_TRAIN,
    validation_file: Path = DEFAULT_VALIDATION,
    test_file: Path = DEFAULT_TEST,
    manifest_file: Path = DEFAULT_MANIFEST,
    tokenizer_file: Path = DEFAULT_TOKENIZER,
    config_file: Path = Path("config.json"),
    report_file: Path = Path("reports/training_preflight.json"),
    max_steps: int = 220_000,
    batch_size: int = 4,
) -> dict[str, Any]:
    config = load_config(config_file)
    model = Monster1.from_config(config)
    parameter_count = count_parameters(model)
    target_train_tokens = parameter_count * MIN_TOKENS_PER_PARAMETER
    sequence_length = int(config.get("training_sequence_length", 128))
    tokens_per_step = batch_size * sequence_length
    required_steps = math.ceil(target_train_tokens / tokens_per_step)
    files = {
        "train": train_file,
        "validation": validation_file,
        "test": test_file,
    }
    missing_files = [str(path) for path in files.values() if not path.is_file()]
    if not manifest_file.is_file():
        missing_files.append(str(manifest_file))
    report: dict[str, Any] = {
        "project": config["name"],
        "status": "blocked_missing_data" if missing_files else "checking",
        "training_ready": False,
        "training_performed": False,
        "parameter_count": parameter_count,
        "vocab_size": int(config["vocab_size"]),
        "target_train_tokens": target_train_tokens,
        "minimum_validation_tokens": MIN_HELDOUT_TOKENS,
        "minimum_test_tokens": MIN_HELDOUT_TOKENS,
        "tokens_per_training_step": tokens_per_step,
        "steps_needed_for_token_target": required_steps,
        "max_steps_requested": max_steps,
        "required_files": [str(train_file), str(validation_file), str(test_file), str(manifest_file)],
        "missing_files": missing_files,
        "data_token_counts": {},
        "corpus_manifest_issues": [],
        "notes": [
            "The repository's legacy checkpoint is not accepted as a trained model.",
            "Training uses only supplied corpus text; there is no synthetic-data fallback.",
            "Manifests marked synthetic or not-real are rejected before tokenizer fitting or training.",
        ],
    }
    if missing_files:
        report["required_data"] = {
            "train": f"{target_train_tokens:,} SentencePiece tokens of licensed, sourced Persian text",
            "validation": f"at least {MIN_HELDOUT_TOKENS:,} held-out SentencePiece tokens",
            "test": f"at least {MIN_HELDOUT_TOKENS:,} separate held-out SentencePiece tokens",
            "manifest": "source/license names and HTTPS URLs, fa language, collection date, attribution, rights_to_train=true, privacy_reviewed=true, and SHA-256 for each split",
        }
        _write_report(report_file, report)
        return report

    issues = _manifest_issues(manifest_file, files)
    report["corpus_manifest_issues"] = issues
    if issues:
        report["status"] = "blocked_invalid_manifest"
        _write_report(report_file, report)
        return report

    if not tokenizer_file.is_file():
        prefix = tokenizer_file.with_suffix("")
        try:
            train_tokenizer(train_file, prefix, vocab_size=int(config["vocab_size"]))
        except Exception as exc:
            report["status"] = "blocked_tokenizer_training"
            report["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
            _write_report(report_file, report)
            return report

    processor = SentencePieceProcessor(model_file=str(tokenizer_file))
    actual_vocab = processor.vocab_size()
    report["tokenizer_vocab_size"] = actual_vocab
    if actual_vocab != int(config["vocab_size"]):
        report["status"] = "blocked_vocab_mismatch"
        report["tokenizer_error"] = (
            f"Expected {config['vocab_size']} pieces, got {actual_vocab}"
        )
        _write_report(report_file, report)
        return report

    counts = {label: count_sentencepiece_tokens(path, processor) for label, path in files.items()}
    report["data_token_counts"] = counts
    data_ready = (
        counts["train"] >= target_train_tokens
        and counts["validation"] >= MIN_HELDOUT_TOKENS
        and counts["test"] >= MIN_HELDOUT_TOKENS
    )
    compute_ready = max_steps >= required_steps
    report["status"] = "ready" if data_ready and compute_ready else (
        "blocked_insufficient_data" if not data_ready else "blocked_compute_budget"
    )
    report["training_ready"] = data_ready and compute_ready
    report["compute_budget_note"] = (
        f"At batch_size={batch_size} and sequence_length={sequence_length}, "
        f"at least {required_steps:,} steps are needed to consume the target token budget. "
        "The GitHub job is still hard-limited to 300 training minutes."
    )
    _write_report(report_file, report)
    return report


def _encode_to_memmap(text_path: Path, tokenizer_path: Path, output_path: Path, token_count: int) -> np.memmap:
    processor = SentencePieceProcessor(model_file=str(tokenizer_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(token_count,))
    offset = 0
    eos_id = processor.eos_id()
    with text_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            encoded = processor.encode(text, out_type=int)
            encoded.append(eos_id)
            end = offset + len(encoded)
            tokens[offset:end] = np.asarray(encoded, dtype=np.uint16)
            offset = end
    if offset != token_count:
        raise RuntimeError(f"Token count changed while encoding: expected {token_count}, got {offset}")
    tokens.flush()
    return tokens


def _get_batch(tokens: np.memmap, batch_size: int, sequence_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - sequence_length - 1
    if max_start <= 0:
        raise ValueError("Token stream is shorter than the requested sequence length")
    starts = torch.randint(0, max_start, (batch_size,)).tolist()
    batch_x = []
    batch_y = []
    for start in starts:
        values = np.asarray(tokens[start : start + sequence_length + 1], dtype=np.int64)
        batch_x.append(torch.from_numpy(values[:-1]))
        batch_y.append(torch.from_numpy(values[1:]))
    return torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)


def _evaluate(model: Monster1, tokens: np.memmap, device: torch.device, batch_size: int, sequence_length: int, batches: int = 50) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            inputs, targets = _get_batch(tokens, batch_size, sequence_length, device)
            logits = model(inputs)
            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))))
    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> dict[str, Any]:
    report = preflight(
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
        manifest_file=args.manifest,
        tokenizer_file=args.tokenizer,
        config_file=args.config,
        report_file=args.report,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    if not report["training_ready"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    model = Monster1.from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batch_size = args.batch_size
    sequence_length = int(config.get("training_sequence_length", 128))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = SentencePieceProcessor(model_file=str(args.tokenizer))
    counts = report["data_token_counts"]
    with tempfile.TemporaryDirectory(prefix="monster1-tokens-") as tmp:
        train_tokens = _encode_to_memmap(args.train_file, args.tokenizer, Path(tmp) / "train.bin", counts["train"])
        validation_tokens = _encode_to_memmap(args.validation_file, args.tokenizer, Path(tmp) / "validation.bin", counts["validation"])
        target_tokens = int(report["target_train_tokens"])
        started = time.monotonic()
        deadline = started + args.time_budget_minutes * 60
        tokens_seen = 0
        last_validation_loss = float("inf")
        step = 0
        model.train()
        for step in range(1, args.max_steps + 1):
            if time.monotonic() >= deadline:
                break
            inputs, targets = _get_batch(train_tokens, batch_size, sequence_length, device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; refusing to create a final model")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_seen += batch_size * sequence_length
            if step % args.eval_every == 0 or step == args.max_steps:
                last_validation_loss = _evaluate(
                    model, validation_tokens, device, batch_size, sequence_length, batches=args.eval_batches
                )
                print(json.dumps({"step": step, "training_loss": float(loss), "validation_loss": last_validation_loss, "tokens_seen": tokens_seen}))
                model.train()
            if tokens_seen >= target_tokens and step % args.eval_every == 0:
                break

        model.eval()
        elapsed = time.monotonic() - started
        complete = tokens_seen >= target_tokens and math.isfinite(last_validation_loss)
        output_status = "trained" if complete else "incomplete_compute_budget"
        report.update({
            "status": output_status,
            "training_performed": True,
            "steps_completed": step,
            "tokens_seen": tokens_seen,
            "training_loss": float(loss),
            "validation_loss": last_validation_loss,
            "elapsed_seconds": round(elapsed, 2),
            "device": str(device),
        })
        if complete:
            test_tokens = _encode_to_memmap(args.test_file, args.tokenizer, Path(tmp) / "test.bin", counts["test"])
            report["test_loss"] = _evaluate(
                model, test_tokens, device, batch_size, sequence_length, batches=args.eval_batches
            )
            baseline_loss = math.log(int(config["vocab_size"]))
            if not (report["validation_loss"] < baseline_loss and report["test_loss"] < baseline_loss):
                report["status"] = "failed_validation"
                report["training_ready"] = False
                report["validation_note"] = "Held-out loss did not beat the uniform-token baseline; no final model was saved."
            else:
                weights_path = args.output_dir / "monster1.pt"
                torch.save(model.cpu().state_dict(), weights_path)
                manifest = {
                    "name": "Monster-1",
                    "status": "trained",
                    "is_synthetic": False,
                    "real_data": True,
                    "pretrained_model": False,
                    "parameter_count": count_parameters(model),
                    "vocab_size": processor.vocab_size(),
                    "training_tokens_target": target_tokens,
                    "training_tokens_seen": tokens_seen,
                    "validation_loss": report["validation_loss"],
                    "test_loss": report["test_loss"],
                    "corpus_manifest_sha256": sha256_file(args.manifest),
                    "train_sha256": sha256_file(args.train_file),
                    "validation_sha256": sha256_file(args.validation_file),
                    "test_sha256": sha256_file(args.test_file),
                    "weights_sha256": sha256_file(weights_path),
                    "tokenizer_sha256": sha256_file(args.tokenizer),
                    "training_steps": step,
                    "elapsed_seconds": round(elapsed, 2),
                }
                (args.output_dir / "model_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                report["training_ready"] = True
        else:
            report["training_ready"] = False
            report["checkpoint_note"] = "No final monster1.pt is written for an incomplete run."
        _write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _path_arg(parser: argparse.ArgumentParser, name: str, default: Path) -> None:
    parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, default=default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gated from-scratch Monster-1 training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight", help="Check data, provenance, tokenizer, and compute budget")
    _path_arg(preflight_parser, "train_file", DEFAULT_TRAIN)
    _path_arg(preflight_parser, "validation_file", DEFAULT_VALIDATION)
    _path_arg(preflight_parser, "test_file", DEFAULT_TEST)
    _path_arg(preflight_parser, "manifest", DEFAULT_MANIFEST)
    _path_arg(preflight_parser, "tokenizer", DEFAULT_TOKENIZER)
    _path_arg(preflight_parser, "config", Path("config.json"))
    _path_arg(preflight_parser, "report", Path("reports/training_preflight.json"))
    preflight_parser.add_argument("--max-steps", type=int, default=220_000)
    preflight_parser.add_argument("--batch-size", type=int, default=4)
    preflight_parser.add_argument("--github-output", type=Path)

    train_parser = subparsers.add_parser("train", help="Train only after the data and compute gate passes")
    for name, default in (
        ("train_file", DEFAULT_TRAIN),
        ("validation_file", DEFAULT_VALIDATION),
        ("test_file", DEFAULT_TEST),
        ("manifest", DEFAULT_MANIFEST),
        ("tokenizer", DEFAULT_TOKENIZER),
        ("config", Path("config.json")),
        ("report", Path("reports/training_report.json")),
        ("output_dir", DEFAULT_OUTPUT),
    ):
        _path_arg(train_parser, name, default)
    train_parser.add_argument("--max-steps", type=int, default=220_000)
    train_parser.add_argument("--time-budget-minutes", type=int, default=300)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--eval-every", type=int, default=500)
    train_parser.add_argument("--eval-batches", type=int, default=20)
    train_parser.add_argument("--seed", type=int, default=42, help="Reproducible sampling seed; never creates training data")

    args = parser.parse_args()
    if args.command == "preflight":
        report = preflight(
            train_file=args.train_file,
            validation_file=args.validation_file,
            test_file=args.test_file,
            manifest_file=args.manifest,
            tokenizer_file=args.tokenizer,
            config_file=args.config,
            report_file=args.report,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
        )
        if args.github_output:
            args.github_output.parent.mkdir(parents=True, exist_ok=True)
            with args.github_output.open("a", encoding="utf-8") as stream:
                stream.write(f"training_ready={'true' if report['training_ready'] else 'false'}\n")
                stream.write(f"status={report['status']}\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
