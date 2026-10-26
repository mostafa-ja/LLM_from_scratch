"""A compact, GPT-2 style decoder-only Transformer implemented from scratch."""

from __future__ import annotations

import inspect
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Query, key and value projections are computed in one fused linear layer,
    then split across heads. PyTorch's scaled dot-product attention fuses the
    softmax, masking and value aggregation, and dispatches to FlashAttention
    when the hardware and dtype support it.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        # One linear layer computes Q, K and V for all heads at once.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim).
        # Moving the head dimension next to batch lets every head attend in parallel.
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # SDPA combines softmax(QK^T / sqrt(d)) @ V. With is_causal=True the
        # mask is applied inside the kernel, so token t cannot see t+1.
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Merge the heads back into one embedding vector per token.
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """The position-wise feed-forward network used inside every block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """A pre-norm Transformer block with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalise the input, run the sub-layer, then add the residual.
        # Residual paths let gradients flow directly through the whole network.
        if self.gradient_checkpointing and self.training:
            x = x + torch.utils.checkpoint.checkpoint(
                self.attn, self.ln_1(x), use_reentrant=False
            )
            x = x + torch.utils.checkpoint.checkpoint(
                self.mlp, self.ln_2(x), use_reentrant=False
            )
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style language model.

    Parameters are initialised with std=0.02. The output projection of every
    residual branch (``c_proj``) is scaled by ``1 / sqrt(2 * n_layer)`` to keep
    activation variance stable as depth grows. Token embeddings and the LM head
    share weights.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the same matrix maps token id -> vector and vector -> logits.
        # It saves parameters and gives the input and output embeddings a shared meaning.
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 initialises every weight with std=0.02 and zeroes biases.
        self.apply(self._init_weights)
        # Each block has two residual branches (attention + MLP), so shrink
        # residual output projections by 1/sqrt(2 * n_layer) to keep variance stable.
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        # Token identity comes from wte; word order comes from wpe.
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        x = self.transformer.drop(token_embeddings + position_embeddings)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Every position predicts the next token, so we compare all B*T predictions
        # against the shifted targets and average the cross-entropy.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eot_token: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample tokens from the model.

        Args:
            idx: ``(B, T)`` tensor of prompt token ids.
            max_new_tokens: Number of tokens to sample.
            temperature: Softmax temperature. Values below 1 sharpen the
                distribution, values above 1 flatten it.
            top_k: If set, keep only the ``k`` most likely tokens per step.
            top_p: If set, keep the smallest set of tokens whose cumulative
                probability exceeds ``top_p`` (nucleus sampling).
            eot_token: Optional end-of-text token that stops generation when
                every sequence in the batch produces it.
        """

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

        self.eval()
        for _ in range(max_new_tokens):
            # The model has a fixed context window; keep only the most recent tokens.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            # Only the last position is needed to predict the next token.
            # Temperature divides logits before softmax: lower = sharper.
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")

            if top_p is not None:
                # Nucleus sampling: keep the smallest prefix of the sorted
                # distribution whose cumulative probability reaches top_p.
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_indices, sorted_logits
                )

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if eot_token is not None and torch.all(next_token == eot_token):
                break

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        device_type: str = "cuda",
        fused: bool | None = None,
    ) -> torch.optim.AdamW:
        """Create AdamW with decay for matrix weights and no decay for vectors."""

        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        decay = [parameter for parameter in parameters if parameter.dim() >= 2]
        no_decay = [parameter for parameter in parameters if parameter.dim() < 2]

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda" if fused is None else fused

        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            fused=use_fused,
        )

    def num_parameters(self, non_embedding: bool = False) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.transformer.wte.weight.numel()
            count -= self.transformer.wpe.weight.numel()
        return count

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Expose decay / no-decay parameter lists for logging or custom optimizers."""

        parameters: Iterable[nn.Parameter] = (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for parameter in parameters:
            (decay if parameter.dim() >= 2 else no_decay).append(parameter)
        return decay, no_decay
