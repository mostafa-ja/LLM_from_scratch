"""Training loop for GPT-2 style language models.

The recipe mirrors the GPT-3/GPT-2 training setup used in the notebooks:

* AdamW with ``betas=(0.9, 0.95)`` and decoupled weight decay
* gradient clipping at global norm 1.0
* linear warmup followed by cosine decay
* gradient accumulation to reach a large effective batch size
* bf16 autocast when the GPU supports it
* ``torch.compile`` and fused AdamW as optional speed-ups
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .config import GPTConfig
from .data import (
    DataLoaderLite,
    download_tiny_shakespeare,
    encode_text,
    split_tokens,
)
from .model import GPT


@dataclass
class TrainConfig:
    data_path: str = "data/tiny_shakespeare.txt"
    out_dir: str = "out"
    model: str = "tiny"
    block_size: int = 0
    batch_size: int = 16
    total_batch_size: int = 8192
    max_steps: int = 600
    warmup_steps: int = 50
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    tokenizer: str = "gpt2"
    val_fraction: float = 0.1
    checkpoint: bool = True
    sample_prompt: str = "ROMEO:"
    sample_tokens: int = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def get_lr(
    step: int,
    *,
    max_lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup followed by cosine decay to ``max_lr * min_lr_ratio``."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    min_lr = max_lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    decay_steps = max(1, max_steps - warmup_steps)
    decay_ratio = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (max_lr - min_lr)


def build_model_config(config: TrainConfig) -> GPTConfig:
    model_config = GPTConfig.preset(config.model)
    if config.block_size:
        model_config.block_size = config.block_size
    return model_config


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _autocast(device: torch.device, dtype: torch.dtype):
    enabled = dtype != torch.float32
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def evaluate(
    model: torch.nn.Module,
    tokens: list[int],
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
) -> float:
    loader = DataLoaderLite(tokens, batch_size, block_size, device)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(max_batches):
            x, y = loader.next_batch()
            with _autocast(device, dtype):
                _, loss = model(x, y)
            if loss is not None:
                losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config: GPTConfig,
    train_config: TrainConfig,
    *,
    step: int,
    val_loss: float,
) -> None:
    raw_model = _raw_model(model)
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in raw_model.state_dict().items()
    }
    payload = {
        "model": state,
        "config": model_config.to_dict(),
        "model_name": train_config.model,
        "step": step,
        "val_loss": val_loss,
        "train_config": asdict(train_config),
    }
    torch.save(payload, path)


@torch.no_grad()
def sample_text(
    model: torch.nn.Module,
    prompt: str,
    *,
    max_new_tokens: int,
    tokenizer: str,
    device: torch.device,
    seed: int | None = None,
) -> str:
    import tiktoken

    encoding = tiktoken.get_encoding(tokenizer)
    prompt_ids = encoding.encode_ordinary(prompt) or [encoding.eot_token]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    raw_model = _raw_model(model)
    if not isinstance(raw_model, GPT):
        raise TypeError("sample_text expects a GPT model")
    raw_model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    output = raw_model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=200,
        eot_token=encoding.eot_token,
    )
    return encoding.decode(output[0].tolist())


def train(config: TrainConfig) -> dict[str, Any]:
    """Run training and return metrics, checkpoint path and generated sample."""

    if config.batch_size < 1 or config.total_batch_size < 1:
        raise ValueError("batch_size and total_batch_size must be positive")

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")

    data_path = Path(config.data_path)
    if not data_path.exists():
        download_tiny_shakespeare(data_path)
    text = data_path.read_text(encoding="utf-8")

    model_config = build_model_config(config)
    tokens = encode_text(text, tokenizer=config.tokenizer)
    train_tokens, val_tokens = split_tokens(tokens, config.val_fraction)

    micro_batch_tokens = config.batch_size * model_config.block_size
    if config.total_batch_size % micro_batch_tokens != 0:
        raise ValueError(
            "total_batch_size must be divisible by batch_size * block_size "
            f"({config.total_batch_size} % {micro_batch_tokens} != 0)"
        )
    grad_accum_steps = config.total_batch_size // micro_batch_tokens

    train_loader = DataLoaderLite(
        train_tokens,
        config.batch_size,
        model_config.block_size,
        device,
    )

    model = GPT(model_config).to(device)
    if config.compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decay_params, no_decay_params = _raw_model(model).parameter_groups()
    print(f"Device: {device} | autocast dtype: {dtype}")
    print(f"Model: {config.model} | parameters: {_raw_model(model).num_parameters():,}")
    print(f"Trainable parameter tensors: {len(decay_params)} decay / {len(no_decay_params)} no-decay")
    print(f"Tokens per optimizer step: {config.total_batch_size:,} ({grad_accum_steps} micro-batches)")
    print(f"Train tokens: {len(train_tokens):,} | val tokens: {len(val_tokens):,}")

    history: dict[str, list[float | int]] = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "grad_norm": [],
        "tokens_per_sec": [],
    }
    best_val_loss = float("inf")
    best_step = 0

    model.train()
    for step in range(config.max_steps):
        started = time.perf_counter()
        lr = get_lr(
            step,
            max_lr=config.max_lr,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            with _autocast(device, dtype):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model did not return a loss")

            # A large effective batch is split into micro-batches that fit in
            # memory. Dividing each loss keeps the accumulated gradient equal
            # to the gradient of the mean loss over the full batch.
            loss = loss / grad_accum_steps
            loss.backward()
            accumulated_loss += loss.detach()

        # One optimizer step per effective batch. Clipping tames rare gradient spikes.
        grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-9)
        tokens_per_second = config.total_batch_size / elapsed
        train_loss = float(accumulated_loss.item())

        is_last_step = step == config.max_steps - 1
        should_evaluate = (
            config.eval_interval > 0
            and (step % config.eval_interval == 0 or is_last_step)
        )

        if should_evaluate:
            val_loss = evaluate(
                model,
                val_tokens,
                batch_size=config.batch_size,
                block_size=model_config.block_size,
                device=device,
                dtype=dtype,
                max_batches=config.eval_batches,
            )
            history["step"].append(step + 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr)
            history["grad_norm"].append(float(grad_norm))
            history["tokens_per_sec"].append(tokens_per_second)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step + 1
                if config.checkpoint:
                    save_checkpoint(
                        out_dir / "best.pt",
                        model,
                        model_config,
                        config,
                        step=step + 1,
                        val_loss=val_loss,
                    )

            print(
                f"step {step + 1:5d}/{config.max_steps} | "
                f"loss {train_loss:.4f} | val {val_loss:.4f} | "
                f"lr {lr:.2e} | grad {float(grad_norm):.2f} | "
                f"{tokens_per_second:,.0f} tok/s"
            )
        elif config.log_interval > 0 and step % config.log_interval == 0:
            print(
                f"step {step + 1:5d}/{config.max_steps} | "
                f"loss {train_loss:.4f} | lr {lr:.2e} | "
                f"{tokens_per_second:,.0f} tok/s"
            )

    if config.checkpoint:
        save_checkpoint(
            out_dir / "ckpt.pt",
            model,
            model_config,
            config,
            step=config.max_steps,
            val_loss=best_val_loss,
        )

    metrics = {
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "parameters": _raw_model(model).num_parameters(),
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    generated = ""
    if config.sample_tokens > 0:
        generated = sample_text(
            model,
            config.sample_prompt,
            max_new_tokens=config.sample_tokens,
            tokenizer=config.tokenizer,
            device=device,
            seed=config.seed,
        )
        (out_dir / "sample.txt").write_text(generated, encoding="utf-8")
        print("\nSample:\n" + generated)

    return {
        "metrics": metrics,
        "checkpoint": str(out_dir / "ckpt.pt") if config.checkpoint else None,
        "best_checkpoint": str(out_dir / "best.pt") if config.checkpoint else None,
        "sample": generated,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=TrainConfig.data_path)
    parser.add_argument("--out-dir", default=TrainConfig.out_dir)
    parser.add_argument(
        "--model",
        default=TrainConfig.model,
        choices=["nano", "tiny", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    parser.add_argument("--block-size", type=int, default=TrainConfig.block_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--total-batch-size", type=int, default=TrainConfig.total_batch_size)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--max-lr", type=float, default=TrainConfig.max_lr)
    parser.add_argument("--min-lr-ratio", type=float, default=TrainConfig.min_lr_ratio)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=TrainConfig.eval_batches)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument(
        "--dtype",
        default=TrainConfig.dtype,
        choices=["auto", "float32", "bfloat16", "float16"],
    )
    parser.add_argument("--compile", action="store_true", default=TrainConfig.compile)
    parser.add_argument("--tokenizer", default=TrainConfig.tokenizer)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument(
        "--no-checkpoint",
        action="store_false",
        dest="checkpoint",
        default=TrainConfig.checkpoint,
    )
    parser.add_argument("--sample-prompt", default=TrainConfig.sample_prompt)
    parser.add_argument("--sample-tokens", type=int, default=TrainConfig.sample_tokens)
    args = parser.parse_args(argv)

    config = TrainConfig(**vars(args))
    train(config)


if __name__ == "__main__":
    main()
