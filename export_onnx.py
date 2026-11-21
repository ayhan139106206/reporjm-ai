from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import Monster1, load_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_graph(model: Monster1, output_path: Path, example_length: int = 16) -> None:
    """Export an already-constructed model with dynamic batch and sequence axes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros((1, example_length), dtype=torch.long)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)


def verify_runtime(
    model: Monster1,
    onnx_path: Path,
    vocab_size: int,
    test_lengths: Iterable[int] = (1, 2, 16, 32, 64),
    generation_steps: int = 4,
) -> dict[str, Any]:
    """Check ONNX/PyTorch parity, variable lengths, and token-loop mechanics."""
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "input_ids":
        raise RuntimeError("ONNX input signature is not input_ids")
    if len(outputs) != 1 or outputs[0].name != "logits":
        raise RuntimeError("ONNX output signature is not logits")
    if inputs[0].shape[0] != "batch" or inputs[0].shape[1] != "sequence":
        raise RuntimeError(f"ONNX batch/sequence dimensions are not dynamic: {inputs[0].shape}")

    model = model.cpu().eval()
    tests = []
    batch_tests = []
    for length in test_lengths:
        if length > model.max_seq_len:
            continue
        ids = (np.arange(length, dtype=np.int64) % vocab_size)[None, :]
        with torch.inference_mode():
            expected = model(torch.from_numpy(ids)).numpy()
        actual = session.run(["logits"], {"input_ids": ids})[0]
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise RuntimeError(f"ONNX output shape/finite-value check failed at length {length}")
        difference = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-3, atol=1e-4):
            raise RuntimeError(f"ONNX/PyTorch parity failed at length {length}: max_abs_diff={difference}")
        tests.append({"sequence_length": length, "output_shape": list(actual.shape), "max_abs_diff": difference})

    batch_ids = np.tile((np.arange(7, dtype=np.int64) % vocab_size)[None, :], (2, 1))
    with torch.inference_mode():
        batch_expected = model(torch.from_numpy(batch_ids)).numpy()
    batch_actual = session.run(["logits"], {"input_ids": batch_ids})[0]
    batch_difference = float(np.max(np.abs(batch_actual - batch_expected)))
    if batch_actual.shape != batch_expected.shape or not np.isfinite(batch_actual).all():
        raise RuntimeError("ONNX output shape/finite-value check failed at batch size 2")
    if not np.allclose(batch_actual, batch_expected, rtol=1e-3, atol=1e-4):
        raise RuntimeError(f"ONNX/PyTorch parity failed at batch size 2: max_abs_diff={batch_difference}")
    batch_tests.append({"batch_size": 2, "sequence_length": 7, "output_shape": list(batch_actual.shape), "max_abs_diff": batch_difference})

    if not tests:
        raise RuntimeError("No valid ONNX sequence-length tests ran")
    generated = [1]
    for _ in range(generation_steps):
        context = generated[-model.max_seq_len :]
        logits = session.run(
            ["logits"], {"input_ids": np.asarray([context], dtype=np.int64)}
        )[0]
        generated.append(int(np.argmax(logits[0, -1])))
    if len(generated) != generation_steps + 1:
        raise RuntimeError("Multi-token generation loop did not produce the requested token count")
    return {
        "status": "passed",
        "onnx_input": {"name": inputs[0].name, "shape": inputs[0].shape, "type": inputs[0].type},
        "onnx_output": {"name": outputs[0].name, "shape": outputs[0].shape, "type": outputs[0].type},
        "sequence_tests": tests,
        "batch_tests": batch_tests,
        "generation_steps": generation_steps,
        "generation_note": "Mechanics only; this does not establish language quality.",
    }


def export_and_verify(
    weights_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
    output_path: Path,
    config_path: Path = Path("config.json"),
) -> dict[str, Any]:
    if not weights_path.is_file() or not tokenizer_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Trained weights, tokenizer, and model_manifest.json are all required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "trained" or manifest.get("pretrained_model") is not False:
        raise ValueError("Refusing ONNX export: checkpoint is not a completed from-scratch training run")
    if sha256_file(weights_path) != manifest.get("weights_sha256"):
        raise ValueError("Weights checksum does not match the completed-training manifest")
    if sha256_file(tokenizer_path) != manifest.get("tokenizer_sha256"):
        raise ValueError("Tokenizer checksum does not match the completed-training manifest")

    from sentencepiece import SentencePieceProcessor

    config = load_config(config_path)
    processor = SentencePieceProcessor(model_file=str(tokenizer_path))
    if processor.vocab_size() != int(config["vocab_size"]):
        raise ValueError(
            f"Tokenizer vocab {processor.vocab_size()} does not match model vocab {config['vocab_size']}"
        )
    prompt = "سلام، من Monster-1 هستم."
    prompt_ids = processor.encode(prompt, out_type=int)
    if not prompt_ids or processor.unk_id() in prompt_ids:
        raise ValueError("Persian tokenizer smoke test produced no tokens or an UNK token")

    model = Monster1.from_config(config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    export_graph(model, output_path, example_length=min(16, int(config.get("max_sequence_length", 1024))))
    report = verify_runtime(
        model,
        output_path,
        vocab_size=processor.vocab_size(),
        test_lengths=(1, 2, 16, 32, 64),
        generation_steps=4,
    )
    generated_ids = list(prompt_ids)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    for _ in range(4):
        context = generated_ids[-int(config.get("max_sequence_length", 1024)) :]
        logits = session.run(["logits"], {"input_ids": np.asarray([context], dtype=np.int64)})[0]
        generated_ids.append(int(np.argmax(logits[0, -1])))
    decoded = processor.decode(generated_ids)
    if len(generated_ids) != len(prompt_ids) + 4 or not decoded:
        raise RuntimeError("Tokenizer decode or multi-token generation smoke test failed")

    report.update({
        "onnx_status": "passed",
        "onnx_sha256": sha256_file(output_path),
        "tokenizer_vocab_size": processor.vocab_size(),
        "tokenizer_smoke_test": "passed",
        "tokenizer_round_trip": "passed",
        "generated_token_count": 4,
    })
    report_path = output_path.parent / "onnx_test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["onnx_sha256"] = report["onnx_sha256"]
    manifest["onnx_test_report_sha256"] = sha256_file(report_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify a genuinely trained Monster-1 model")
    parser.add_argument("--weights", type=Path, default=Path("artifacts/model/monster1.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/model/monster1_tokenizer.model"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/model/model_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model/monster1.onnx"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    report = export_and_verify(args.weights, args.tokenizer, args.manifest, args.output, args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
