from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import generate_dataset


DEFAULT_DATA_DIR = ROOT / "data" / "synthetic_test"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "synthetic_test"
DEFAULT_REPORT = ROOT / "reports" / "synthetic_test" / "training_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_report(report: dict[str, Any], output_dir: Path, mirror_path: Path) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(payload, encoding="utf-8")
    mirror_path.write_text(payload, encoding="utf-8")


def _count_tokens(path: Path, processor: Any) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if text:
                count += len(processor.encode(text, out_type=int)) + 1
    return count


def _update_manifest(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "synthetic_test" or manifest.get("is_synthetic") is not True:
        raise ValueError("Refusing synthetic training: corpus manifest is not explicitly synthetic_test")
    manifest.update(values)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _train_synthetic_tokenizer(train_file: Path, model_prefix: Path, vocab_size: int) -> dict[str, Any]:
    """Fit the existing unigram tokenizer and transparently reserve test-only filler pieces.

    A small synthetic corpus may contain far fewer than 4,096 useful unigram
    pieces. The pilot fit measures that ceiling; explicit unused reserved
    symbols then preserve the configured model/tokenizer tensor shape without
    pretending those symbols were learned from data.
    """
    import sentencepiece as spm
    from sentencepiece import SentencePieceProcessor

    from tokenizer import inspect_corpus

    corpus_stats = inspect_corpus(train_file)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="monster1-synthetic-spm-") as temporary:
        pilot_prefix = Path(temporary) / "pilot"
        spm.SentencePieceTrainer.train(
            input=str(train_file),
            model_prefix=str(pilot_prefix),
            model_type="unigram",
            vocab_size=vocab_size,
            character_coverage=0.9995,
            hard_vocab_limit=False,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            num_threads=2,
            shuffle_input_sentence=False,
            input_sentence_size=0,
            minloglevel=2,
        )
        pilot = SentencePieceProcessor(model_file=str(pilot_prefix.with_suffix(".model")))
        learned_piece_count = pilot.vocab_size()
        filler_count = vocab_size - learned_piece_count
        if filler_count < 0:
            raise RuntimeError(
                f"Synthetic tokenizer pilot unexpectedly exceeds configured vocabulary: {learned_piece_count} > {vocab_size}"
            )
        reserved_symbols = [f"<synthetic_reserved_{index:04d}>" for index in range(filler_count)]
        spm.SentencePieceTrainer.train(
            input=str(train_file),
            model_prefix=str(model_prefix),
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
            user_defined_symbols=reserved_symbols,
            minloglevel=2,
        )
    processor = SentencePieceProcessor(model_file=str(model_prefix.with_suffix(".model")))
    if processor.vocab_size() != vocab_size:
        raise RuntimeError(f"Synthetic tokenizer has {processor.vocab_size()} pieces; expected exactly {vocab_size}")
    return {
        "algorithm": "SentencePiece unigram (same production tokenizer family)",
        "corpus_characters": corpus_stats["characters"],
        "corpus_lines": corpus_stats["nonempty_lines"],
        "learned_piece_count_before_test_reserve": learned_piece_count,
        "reserved_test_only_piece_count": filler_count,
        "reserved_test_only_piece_prefix": "<synthetic_reserved_",
        "reserved_piece_note": "Unused placeholders solely preserve the configured 4,096-way test model head; they were not learned corpus vocabulary and are not used in prompts.",
        "vocab_size": processor.vocab_size(),
    }


def _train_fresh_model(
    data_dir: Path,
    output_dir: Path,
    config_path: Path,
    report: dict[str, Any],
    mirror_report: Path,
    steps: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
    on_phase: Any,
) -> dict[str, Any]:
    on_phase("dependency_imports")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sentencepiece import SentencePieceProcessor

    from model import Monster1, count_parameters, load_config
    from train import _encode_to_memmap, _evaluate, _get_batch

    on_phase("configuration_and_fresh_model_initialization")
    config = load_config(config_path)
    configured_vocab = int(config["vocab_size"])
    model = Monster1.from_config(config)
    if model.vocab_size != configured_vocab:
        raise RuntimeError(f"Model/config vocabulary mismatch: {model.vocab_size} != {configured_vocab}")

    on_phase("tokenizer_training")
    tokenizer_path = output_dir / "tokenizer.model"
    vocab_path = output_dir / "tokenizer.vocab"
    tokenizer_details = _train_synthetic_tokenizer(data_dir / "train.txt", output_dir / "tokenizer", configured_vocab)
    if not tokenizer_path.is_file() or not vocab_path.is_file():
        raise RuntimeError("SentencePiece did not create the expected tokenizer.model and tokenizer.vocab files")
    on_phase("tokenizer_vocab_verification")
    processor = SentencePieceProcessor(model_file=str(tokenizer_path))
    if processor.vocab_size() != configured_vocab:
        raise RuntimeError(f"Tokenizer/config vocabulary mismatch: {processor.vocab_size()} != {configured_vocab}")

    on_phase("split_token_counting_and_encoding")
    split_paths = {name: data_dir / f"{name}.txt" for name in ("train", "validation", "test")}
    token_counts = {name: _count_tokens(path, processor) for name, path in split_paths.items()}
    sequence_length = int(config.get("training_sequence_length", 128))
    for name, count in token_counts.items():
        if count <= sequence_length + 1:
            raise ValueError(f"Synthetic {name} split is too short: {count} tokens for sequence length {sequence_length}")
    manifest_path = data_dir / "corpus_manifest.json"
    corpus_manifest = _update_manifest(
        manifest_path,
        {
            "tokenizer_vocab_size": processor.vocab_size(),
            "model_vocab_size": model.vocab_size,
            "tokenizer_details": tokenizer_details,
            "token_counts_sentencepiece_including_eos": token_counts,
        },
    )
    shutil.copy2(manifest_path, output_dir / "corpus_manifest.json")

    on_phase("from_scratch_training")
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This is a fresh initialization. No checkpoint is read or loaded here.
    model = Monster1.from_config(config).to(device)
    initial_state_hash = hashlib.sha256(
        b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    with tempfile.TemporaryDirectory(prefix="monster1-synthetic-tokens-") as temporary:
        token_dir = Path(temporary)
        train_tokens = _encode_to_memmap(
            split_paths["train"], tokenizer_path, token_dir / "train.bin", token_counts["train"]
        )
        validation_tokens = _encode_to_memmap(
            split_paths["validation"], tokenizer_path, token_dir / "validation.bin", token_counts["validation"]
        )
        test_tokens = _encode_to_memmap(
            split_paths["test"], tokenizer_path, token_dir / "test.bin", token_counts["test"]
        )
        model.train()
        losses: list[float] = []
        started = time.monotonic()
        tokens_seen = 0
        for step in range(1, steps + 1):
            inputs, targets = _get_batch(train_tokens, batch_size, sequence_length, device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite synthetic training loss at step {step}; no weights saved")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            tokens_seen += batch_size * sequence_length

        on_phase("held_out_validation_and_test")
        model.eval()
        validation_loss = _evaluate(model, validation_tokens, device, batch_size, sequence_length, batches=eval_batches)
        test_loss = _evaluate(model, test_tokens, device, batch_size, sequence_length, batches=eval_batches)
        elapsed = round(time.monotonic() - started, 2)

    on_phase("checkpoint_and_training_manifest_save")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "monster1.pt"
    torch.save(model.cpu().state_dict(), weights_path)
    final_state_hash = hashlib.sha256(
        b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in model.parameters())
    ).hexdigest()
    if initial_state_hash == final_state_hash:
        raise RuntimeError("Fresh-model weights did not change during synthetic_test optimization")

    model_manifest = {
        "name": "Monster-1",
        "status": "synthetic_test_only",
        "training_mode": "synthetic_test",
        "is_synthetic": True,
        "not_for_production": True,
        "pretrained_model": False,
        "weights_initialized_from_scratch": True,
        "checkpoint_loaded": False,
        "parameter_count": count_parameters(model),
        "vocab_size": model.vocab_size,
        "tokenizer_vocab_size": processor.vocab_size(),
        "tokenizer_details": tokenizer_details,
        "training_tokens_target": tokens_seen,
        "training_tokens_seen": tokens_seen,
        "data_token_counts": token_counts,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "training_loss_first_step": losses[0],
        "training_loss_last_step": losses[-1],
        "training_steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "corpus_manifest_sha256": _sha256(output_dir / "corpus_manifest.json"),
        "train_sha256": corpus_manifest["sha256_train"],
        "validation_sha256": corpus_manifest["sha256_validation"],
        "test_sha256": corpus_manifest["sha256_test"],
        "weights_sha256": _sha256(weights_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "quality_claim": False,
        "real_data_threshold_satisfied": False,
        "api_engine_ready": False,
    }
    model_manifest_path = output_dir / "model_manifest.json"
    model_manifest_path.write_text(json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report.update(
        {
            "model_status": "synthetic_test_only",
            "pretrained_model_used": False,
            "checkpoint_loaded": False,
            "weights_initialized_from_scratch": True,
            "model_parameter_count": count_parameters(model),
            "model_vocab_size": model.vocab_size,
            "tokenizer_vocab_size": processor.vocab_size(),
            "tokenizer_details": tokenizer_details,
            "data_token_counts_sentencepiece_including_eos": token_counts,
            "training_steps": steps,
            "training_tokens_seen": tokens_seen,
            "training_loss_first_step": losses[0],
            "training_loss_last_step": losses[-1],
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "elapsed_seconds": elapsed,
            "device": str(device),
            "quality_verified": False,
            "real_data_threshold_satisfied": False,
        }
    )
    _save_report(report, output_dir, mirror_report)
    return model_manifest


def _run_onnx_inference(output_dir: Path, report: dict[str, Any]) -> list[dict[str, str]]:
    import numpy as np
    import onnxruntime as ort
    from sentencepiece import SentencePieceProcessor

    tokenizer_path = output_dir / "tokenizer.model"
    processor = SentencePieceProcessor(model_file=str(tokenizer_path))
    session = ort.InferenceSession(str(output_dir / "monster1.onnx"), providers=["CPUExecutionProvider"])
    prompts = [
        "سلام، حالت چطور است؟",
        "لطفاً جمع 12 و 8 را بگو.",
        "یک خط پایتون برای چاپ عدد 7 بنویس.",
    ]
    responses: list[dict[str, str]] = []
    forbidden_ids = {processor.unk_id(), processor.bos_id(), processor.pad_id(), processor.eos_id()}
    model_manifest = json.loads((output_dir / "model_manifest.json").read_text(encoding="utf-8"))
    reserved_count = int(model_manifest.get("tokenizer_details", {}).get("reserved_test_only_piece_count", 0))
    for index in range(reserved_count):
        reserved_id = processor.piece_to_id(f"<synthetic_reserved_{index:04d}>")
        if reserved_id <= processor.unk_id():
            raise RuntimeError(f"Could not resolve reserved synthetic tokenizer piece {index}")
        forbidden_ids.add(reserved_id)
    for prompt in prompts:
        generated = processor.encode(prompt, out_type=int)
        if not generated or processor.unk_id() in generated:
            raise RuntimeError(f"Persian inference prompt was not tokenized cleanly: {prompt!r}")
        for _ in range(8):
            context = generated[-1024:]
            logits = session.run(["logits"], {"input_ids": np.asarray([context], dtype=np.int64)})[0]
            next_logits = logits[0, -1].copy()
            for token_id in forbidden_ids:
                if 0 <= token_id < len(next_logits):
                    next_logits[token_id] = -np.inf
            generated.append(int(np.argmax(next_logits)))
        reply = processor.decode(generated[len(processor.encode(prompt, out_type=int)) :]).strip()
        if not reply:
            raise RuntimeError(f"ONNX produced an empty decoded response for prompt: {prompt!r}")
        responses.append({"prompt": prompt, "response": reply})
    report["inference"] = {
        "status": "passed",
        "runtime": "ONNX Runtime CPU",
        "responses": responses,
        "response_count": len(responses),
        "interpretation": "Raw outputs from a tiny synthetic-data smoke test; they do not establish Persian language quality or production readiness.",
    }
    return responses


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "synthetic_test":
        raise ValueError("Only the explicitly isolated mode 'synthetic_test' is supported by this runner")
    output_dir = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    report_path = args.report.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "project": "Monster-1",
        "mode": "synthetic_test",
        "status": "running",
        "is_synthetic": True,
        "real_data": False,
        "pretrained_model_used": False,
        "checkpoint_loaded": False,
        "engine_ready": False,
        "quality_verified": False,
        "production_artifacts_written": False,
        "workflow_checks": {
            "dependency_install_outcome": os.environ.get("DEPENDENCY_INSTALL_OUTCOME", "not_run"),
            "unit_test_outcome": os.environ.get("UNIT_TEST_OUTCOME", "not_run"),
        },
        "phases": {},
        "artifacts_directory": str(output_dir),
        "notes": [
            "This run uses locally generated synthetic Persian data only.",
            "The real-data training threshold remains unchanged and is not met or bypassed by this mode.",
            "Successful ONNX execution is a mechanics test, not evidence of language quality or production readiness.",
        ],
    }
    _save_report(report, output_dir, report_path)
    phase_state = {"name": "data_generation"}

    def set_phase(name: str) -> None:
        phase_state["name"] = name
        report["current_phase"] = name
        _save_report(report, output_dir, report_path)

    try:
        set_phase("data_generation")
        manifest = generate_dataset(data_dir, seed=args.seed)
        if manifest.get("is_synthetic") is not True or manifest.get("mode") != "synthetic_test":
            raise RuntimeError("Generator did not identify its output as synthetic_test data")
        shutil.copy2(data_dir / "corpus_manifest.json", output_dir / "corpus_manifest.json")
        report["corpus_manifest"] = manifest
        report["phases"]["data_generation"] = {
            "status": "passed",
            "sample_counts": manifest["sample_counts"],
            "topic_sample_counts": manifest["topic_sample_counts"],
            "duplicates_removed": manifest["duplicates_removed"],
        }
        _save_report(report, output_dir, report_path)

        model_manifest = _train_fresh_model(
            data_dir=data_dir,
            output_dir=output_dir,
            config_path=args.config.resolve(),
            report=report,
            mirror_report=report_path,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            seed=args.seed,
            on_phase=set_phase,
        )
        set_phase("tokenizer_training_and_vocab_check")
        report["phases"]["tokenizer_training_and_vocab_check"] = {
            "status": "passed",
            "tokenizer_vocab_size": model_manifest["tokenizer_vocab_size"],
            "model_vocab_size": model_manifest["vocab_size"],
        }
        report["phases"]["from_scratch_training"] = {
            "status": "passed",
            "steps": model_manifest["training_steps"],
            "tokens_seen": model_manifest["training_tokens_seen"],
            "weights_sha256": model_manifest["weights_sha256"],
        }
        _save_report(report, output_dir, report_path)

        set_phase("onnx_export_and_runtime_parity")
        from export_onnx import export_synthetic_test

        onnx_report = export_synthetic_test(
            weights_path=output_dir / "monster1.pt",
            tokenizer_path=output_dir / "tokenizer.model",
            manifest_path=output_dir / "model_manifest.json",
            output_path=output_dir / "monster1.onnx",
            config_path=args.config.resolve(),
        )
        report["onnx"] = onnx_report
        report["phases"]["onnx_export_and_runtime_parity"] = {
            "status": "passed",
            "sha256": onnx_report["onnx_sha256"],
            "sequence_tests": onnx_report["sequence_tests"],
            "generation_steps": onnx_report["generation_steps"],
        }
        _save_report(report, output_dir, report_path)

        set_phase("persian_prompt_inference")
        _run_onnx_inference(output_dir, report)
        report["phases"]["persian_prompt_inference"] = {
            "status": "passed",
            "response_count": report["inference"]["response_count"],
        }
        report["status"] = "completed_synthetic_test"
        report["engine_ready"] = False
        report["quality_verified"] = False
        report["production_artifacts_written"] = False
        report["completed"] = True
        report["failure"] = None
        report.pop("current_phase", None)
        _save_report(report, output_dir, report_path)
        return report
    except Exception as exc:
        current_phase = phase_state["name"]
        report["status"] = "failed"
        report["completed"] = False
        report["engine_ready"] = False
        report["quality_verified"] = False
        report["production_artifacts_written"] = False
        report["failure"] = {
            "phase": current_phase,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-12000:],
        }
        report["phases"].setdefault(current_phase, {})
        report["phases"][current_phase].update(
            {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc)}
        )
        for source in (data_dir / "corpus_manifest.json",):
            if source.is_file():
                shutil.copy2(source, output_dir / "corpus_manifest.json")
        _save_report(report, output_dir, report_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end isolated Monster-1 synthetic_test smoke run")
    parser.add_argument("--mode", choices=("synthetic_test",), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_batches < 1:
        parser.error("steps, batch-size, and eval-batches must all be positive")
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure": {"exception_type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
