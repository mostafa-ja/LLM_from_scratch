"""Sample text from a checkpoint produced by ``llm_from_scratch.train``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from .config import GPTConfig
from .model import GPT
from .train import resolve_device


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[GPT, GPTConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if "model" not in payload or "config" not in payload:
        raise ValueError("Checkpoint must contain 'model' and 'config' entries")

    config = GPTConfig.from_dict(payload["config"])
    model = GPT(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, config, payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/best.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    import tiktoken

    device = resolve_device(args.device)
    model, _, payload = load_checkpoint(args.checkpoint, device)
    encoding = tiktoken.get_encoding(args.tokenizer)

    prompt_ids = encoding.encode_ordinary(args.prompt) or [encoding.eot_token]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    torch.manual_seed(args.seed)
    output = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eot_token=encoding.eot_token,
    )
    text = encoding.decode(output[0].tolist())

    print(
        f"checkpoint step={payload.get('step')} | "
        f"val_loss={payload.get('val_loss')} | "
        f"parameters={model.num_parameters():,}",
        file=sys.stderr,
    )
    print(text)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
