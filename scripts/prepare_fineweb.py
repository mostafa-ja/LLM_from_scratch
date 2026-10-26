"""Tokenize and shard a streaming Hugging Face dataset for pretraining.

Example:

    python scripts/prepare_fineweb.py \
        --total-tokens 5_000_000_000 \
        --shard-size 100_000_000 \
        --out-dir data/fineweb_edu

Requires the optional ``data`` dependencies: ``pip install -e ".[data]"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_from_scratch.data import tokenize_documents_to_shards  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="data/fineweb_edu")
    parser.add_argument("--total-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--val-shards", type=int, default=1)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            'Install the data extras first: pip install -e ".[data]"'
        ) from exc

    print(
        f"Streaming {args.dataset} ({args.subset}, {args.split}) -> "
        f"{args.total_tokens:,} tokens in {args.shard_size:,}-token shards"
    )
    dataset = load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        streaming=True,
    )
    manifest = tokenize_documents_to_shards(
        dataset,
        args.out_dir,
        tokenizer=args.tokenizer,
        total_tokens=args.total_tokens,
        shard_size=args.shard_size,
        val_shards=args.val_shards,
        progress=not args.no_progress,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
