from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


class Monster1(nn.Module):
    """Small, decoder-only Transformer used by the Monster-1 project.

    Parameter names and tensor shapes intentionally remain compatible with the
    original 4-layer checkpoint. The original checkpoint is not considered a
    trained language model unless a verified training manifest accompanies it.
    """

    def __init__(
        self,
        vocab: int = 4096,
        dim: int = 256,
        layers: int = 4,
        heads: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab <= 0 or dim <= 0 or layers <= 0 or heads <= 0:
            raise ValueError("vocab, dim, layers, and heads must be positive")
        if dim % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.vocab_size = vocab
        self.max_seq_len = max_seq_len
        self.token = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(max_seq_len, dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Monster1":
        """Build the model from the project's JSON configuration."""
        return cls(
            vocab=int(config["vocab_size"]),
            dim=int(config["hidden_size"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            max_seq_len=int(config.get("max_sequence_length", 1024)),
            dropout=float(config.get("dropout", 0.1)),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")
        seq_len = input_ids.size(1)
        if not torch.jit.is_tracing():
            if seq_len == 0:
                raise ValueError("sequence length must be greater than zero")
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_sequence_length={self.max_seq_len}"
                )

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token(input_ids) + self.position(positions)
        # Spell out the same post-norm TransformerEncoderLayer math instead of
        # nn.MultiheadAttention's shape-specialized fast paths. This keeps the
        # sequence axis dynamic in ONNX while preserving every checkpoint key.
        for layer in self.transformer.layers:
            residual = hidden
            attn = layer.self_attn
            batch = hidden.size(0)
            head_dim = attn.embed_dim // attn.num_heads
            qkv = nn.functional.linear(hidden, attn.in_proj_weight, attn.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.reshape(batch, -1, attn.num_heads, head_dim).transpose(1, 2)
            key = key.reshape(batch, -1, attn.num_heads, head_dim).transpose(1, 2)
            value = value.reshape(batch, -1, attn.num_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
            causal_mask = torch.triu(
                torch.ones_like(scores, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1)
            attention = nn.functional.dropout(
                attention, p=attn.dropout, training=self.training
            )
            context = torch.matmul(attention, value)
            context = context.transpose(1, 2).contiguous().reshape(batch, -1, attn.embed_dim)
            attended = nn.functional.linear(context, attn.out_proj.weight, attn.out_proj.bias)
            hidden = layer.norm1(residual + layer.dropout1(attended))
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))
            hidden = layer.norm2(hidden + layer.dropout2(feedforward))
        return self.head(self.norm(hidden))


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("vocab_size", "hidden_size", "layers", "heads"):
        if key not in config:
            raise ValueError(f"config is missing required key: {key}")
    return config


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
