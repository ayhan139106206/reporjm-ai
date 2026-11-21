from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentencepiece import SentencePieceProcessor

from model import Monster1, count_parameters, load_config


MODEL_DIR = Path(os.environ.get("MONSTER1_MODEL_DIR", "artifacts/model"))
CONFIG_PATH = Path(os.environ.get("MONSTER1_CONFIG", "config.json"))
MAX_GENERATION_TOKENS = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRuntime:
    def __init__(self, model_dir: Path = MODEL_DIR, config_path: Path = CONFIG_PATH) -> None:
        self.config = load_config(config_path)
        self.parameter_count = count_parameters(Monster1.from_config(self.config))
        self.weights_path = model_dir / "monster1.pt"
        self.onnx_path = model_dir / "monster1.onnx"
        self.tokenizer_path = model_dir / "monster1_tokenizer.model"
        self.manifest_path = model_dir / "model_manifest.json"
        required = (self.weights_path, self.onnx_path, self.tokenizer_path, self.manifest_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Missing trained model artifacts: " + ", ".join(missing))

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "trained" or manifest.get("pretrained_model") is not False:
            raise RuntimeError("Model manifest does not attest a completed from-scratch training run")
        if int(manifest.get("parameter_count", -1)) != self.parameter_count:
            raise RuntimeError("Model manifest parameter count does not match config")
        if int(manifest.get("vocab_size", -1)) != int(self.config["vocab_size"]):
            raise RuntimeError("Model manifest vocabulary does not match config")
        if int(manifest.get("training_tokens_seen", 0)) < int(manifest.get("training_tokens_target", 1)):
            raise RuntimeError("Training token target was not met")
        if not (float(manifest.get("validation_loss", float("inf"))) < np.log(self.config["vocab_size"])):
            raise RuntimeError("Validation loss did not beat the uniform-token baseline")
        if not (float(manifest.get("test_loss", float("inf"))) < np.log(self.config["vocab_size"])):
            raise RuntimeError("Test loss did not beat the uniform-token baseline")

        expected_hashes = {
            self.weights_path: manifest.get("weights_sha256"),
            self.tokenizer_path: manifest.get("tokenizer_sha256"),
            self.onnx_path: manifest.get("onnx_sha256"),
        }
        for path, expected in expected_hashes.items():
            if not expected or sha256_file(path) != expected:
                raise RuntimeError(f"Artifact checksum is missing or mismatched: {path.name}")

        self.tokenizer = SentencePieceProcessor(model_file=str(self.tokenizer_path))
        if self.tokenizer.vocab_size() != int(self.config["vocab_size"]):
            raise RuntimeError(
                f"Tokenizer vocab {self.tokenizer.vocab_size()} does not match config vocab {self.config['vocab_size']}"
            )
        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "input_ids"
            or inputs[0].shape[0] != "batch"
            or inputs[0].shape[1] != "sequence"
        ):
            raise RuntimeError("ONNX model must expose a dynamic [batch, sequence] input_ids tensor")
        if (
            len(outputs) != 1
            or outputs[0].name != "logits"
            or outputs[0].shape[0] != "batch"
            or outputs[0].shape[1] != "sequence"
            or outputs[0].shape[2] != int(self.config["vocab_size"])
        ):
            raise RuntimeError("ONNX model must expose logits with shape [batch, sequence, vocab]")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.max_sequence_length = int(self.config.get("max_sequence_length", 1024))
        self.eos_id = self.tokenizer.eos_id()
        self.ready = True

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        token_ids = self.tokenizer.encode(prompt, out_type=int)
        if not token_ids or self.tokenizer.unk_id() in token_ids:
            raise ValueError("The input could not be represented without UNK tokens")
        if len(token_ids) >= self.max_sequence_length:
            token_ids = token_ids[-(self.max_sequence_length - 1) :]
        prompt_length = len(token_ids)
        for _ in range(max_new_tokens):
            context = token_ids[-self.max_sequence_length :]
            input_ids = np.asarray([context], dtype=np.int64)
            logits = self.session.run(
                [self.output_name], {self.input_name: input_ids}
            )[0]
            next_id = int(np.argmax(logits[0, -1]))
            token_ids.append(next_id)
            if next_id == self.eos_id:
                break
        return self.tokenizer.decode(token_ids[prompt_length:]).strip()


_runtime: ModelRuntime | None = None
_runtime_checked = False
_runtime_error = "No completed, verified Monster-1 training artifact is installed."


def get_runtime() -> ModelRuntime | None:
    global _runtime, _runtime_checked, _runtime_error
    if _runtime_checked:
        return _runtime
    _runtime_checked = True
    try:
        _runtime = ModelRuntime()
        _runtime_error = ""
    except Exception as exc:
        _runtime = None
        _runtime_error = f"{type(exc).__name__}: {exc}"
    return _runtime


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_new_tokens: int = Field(default=64, ge=1, le=MAX_GENERATION_TOKENS)


app = FastAPI(title="Monster-1 Persian Model API", version="1.1.0")


@app.get("/v1/health")
def health() -> dict[str, object]:
    runtime = get_runtime()
    return {
        "ok": True,
        "service": "monster-1-api",
        "model_ready": runtime is not None,
        "status": "ready" if runtime is not None else "not_ready",
    }


@app.get("/v1/model")
def model_info() -> dict[str, object]:
    config = load_config(CONFIG_PATH)
    runtime = get_runtime()
    return {
        "name": config["name"],
        "version": config["version"],
        "language": config["language"],
        "architecture": config["architecture"],
        "vocab_size": config["vocab_size"],
        "parameters": count_parameters(Monster1.from_config(config)),
        "pretrained_model": False,
        "engine_ready": runtime is not None,
        "training_status": "trained" if runtime is not None else "not_trained_or_unverified",
    }


@app.post("/v1/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_not_ready", "reason": _runtime_error},
        )
    user_messages = [message.content for message in request.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail={"error": "user_message_required"})
    prompt = "\n".join(user_messages[-4:])
    try:
        reply = runtime.generate(prompt, request.max_new_tokens)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "generation_failed", "reason": f"{type(exc).__name__}: {exc}"},
        ) from exc
    return {"model": "Monster-1", "reply": reply, "generated_tokens_limit": request.max_new_tokens}
