from __future__ import annotations

import json

import pytest
import torch

from llm_from_scratch.train import TrainConfig, get_lr, train


def test_get_lr_warmup_and_decay() -> None:
    kwargs = {
        "max_lr": 6e-4,
        "warmup_steps": 10,
        "max_steps": 100,
        "min_lr_ratio": 0.1,
    }
    assert get_lr(0, **kwargs) == pytest.approx(6e-5)
    assert get_lr(9, **kwargs) == pytest.approx(6e-4)
    assert get_lr(100, **kwargs) == pytest.approx(6e-5)
    assert get_lr(500, **kwargs) == pytest.approx(6e-5)

    values = [get_lr(step, **kwargs) for step in range(10, 100)]
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_training_smoke(tmp_path) -> None:
    text_path = tmp_path / "corpus.txt"
    text_path.write_text(
        "the quick brown fox jumps over the lazy dog. " * 250,
        encoding="utf-8",
    )

    config = TrainConfig(
        data_path=str(text_path),
        out_dir=str(tmp_path / "out"),
        model="nano",
        block_size=32,
        batch_size=2,
        total_batch_size=128,
        max_steps=2,
        warmup_steps=1,
        max_lr=1e-3,
        eval_interval=1,
        eval_batches=1,
        log_interval=1,
        device="cpu",
        dtype="float32",
        tokenizer="gpt2",
        sample_prompt="the",
        sample_tokens=8,
    )

    result = train(config)

    assert result["metrics"]["parameters"] > 0
    assert len(result["metrics"]["history"]["step"]) == 2
    assert result["checkpoint"] is not None
    assert (tmp_path / "out" / "ckpt.pt").exists()
    assert (tmp_path / "out" / "best.pt").exists()
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "sample.txt").exists()

    checkpoint = torch.load(
        tmp_path / "out" / "ckpt.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "model" in checkpoint
    assert checkpoint["config"]["block_size"] == 32
    assert checkpoint["step"] == 2

    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_val_loss"] < float("inf")
