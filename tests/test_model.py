from __future__ import annotations

import pytest
import torch

from llm_from_scratch.config import GPTConfig
from llm_from_scratch.model import GPT


@pytest.fixture()
def model_config() -> GPTConfig:
    return GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
    )


def test_forward_shapes_and_loss(model_config: GPTConfig) -> None:
    model = GPT(model_config)
    idx = torch.randint(0, model_config.vocab_size, (3, 16))
    logits, loss = model(idx, idx)

    assert logits.shape == (3, 16, model_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_causal_masking(model_config: GPTConfig) -> None:
    model = GPT(model_config).eval()
    idx = torch.randint(0, model_config.vocab_size, (1, 16))
    changed = idx.clone()
    changed[:, 10:] = (changed[:, 10:] + 1) % model_config.vocab_size

    with torch.no_grad():
        logits, _ = model(idx)
        changed_logits, _ = model(changed)

    torch.testing.assert_close(logits[:, :10], changed_logits[:, :10])
    assert not torch.allclose(logits[:, 10:], changed_logits[:, 10:])


def test_generation_shapes(model_config: GPTConfig) -> None:
    model = GPT(model_config).eval()
    prompt = torch.randint(0, model_config.vocab_size, (1, 5))
    output = model.generate(prompt, max_new_tokens=7, temperature=1.0, top_k=10)

    assert output.shape == (1, 12)
    assert torch.equal(output[:, :5], prompt)


def test_weight_tying(model_config: GPTConfig) -> None:
    model = GPT(model_config)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_parameter_estimation(model_config: GPTConfig) -> None:
    model = GPT(model_config)
    assert model.num_parameters() > 0
    assert model.num_parameters(non_embedding=True) < model.num_parameters()


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        GPTConfig(n_embd=10, n_head=3)
    with pytest.raises(ValueError):
        GPTConfig(dropout=1.0)


def test_optimizer_parameter_groups(model_config: GPTConfig) -> None:
    model = GPT(model_config)
    decay, no_decay = model.parameter_groups()
    assert decay and no_decay
    assert all(parameter.dim() >= 2 for parameter in decay)
    assert all(parameter.dim() < 2 for parameter in no_decay)

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        device_type="cpu",
        fused=False,
    )
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
