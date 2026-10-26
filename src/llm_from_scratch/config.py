"""Model configuration presets for GPT-2 style language models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GPTConfig:
    """Hyper-parameters that define the shape of a GPT model.

    The defaults match GPT-2 small. ``vocab_size`` is padded from 50,257 to
    50,304 so that the vocabulary dimension is a multiple of 64, which keeps
    GPU matrix multiplications aligned.
    """

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GPTConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    @classmethod
    def preset(cls, name: str) -> "GPTConfig":
        """Return a named GPT-2 or small-scale preset."""

        key = name.lower().replace("_", "-")
        presets: dict[str, GPTConfig] = {
            "nano": cls(block_size=128, n_layer=2, n_head=4, n_embd=128),
            "tiny": cls(block_size=256, n_layer=6, n_head=6, n_embd=384),
            "gpt2-small": cls(block_size=1024, n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": cls(block_size=1024, n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": cls(block_size=1024, n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": cls(block_size=1024, n_layer=48, n_head=25, n_embd=1600),
        }
        if key not in presets:
            available = ", ".join(sorted(presets))
            raise ValueError(f"Unknown model preset {name!r}. Choose one of: {available}")
        return presets[key]

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        return cls.preset("gpt2-small")

    @classmethod
    def tiny(cls) -> "GPTConfig":
        return cls.preset("tiny")

    @classmethod
    def nano(cls) -> "GPTConfig":
        return cls.preset("nano")
