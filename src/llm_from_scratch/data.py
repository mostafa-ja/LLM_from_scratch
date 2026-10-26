"""Dataset and data-pipeline utilities.

Three data paths are supported:

* :class:`TokenDataset` - random-access windows over an in-memory token tensor.
* :class:`DataLoaderLite` - deterministic, sequential batches for quick runs.
* :class:`ShardDataset` - memory-mapped ``.npy`` shards for large corpora.

``tokenize_documents_to_shards`` turns a stream of documents into GPT-2 token
shards, following the same approach used for FineWeb-Edu in the notebooks.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str | Path, force: bool = False) -> Path:
    """Download Tiny Shakespeare to ``path`` if it is not already present."""

    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Tiny Shakespeare to {path} ...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


def encode_text(text: str, tokenizer: str = "gpt2", add_eot: bool = False) -> list[int]:
    """Tokenize text with ``tiktoken`` and optionally prepend an EOT token."""

    import tiktoken

    encoding = tiktoken.get_encoding(tokenizer)
    ids = encoding.encode_ordinary(text)
    if add_eot:
        ids = [encoding.eot_token] + ids
    return ids


def split_tokens(tokens: Sequence[int], val_fraction: float = 0.1) -> tuple[list[int], list[int]]:
    """Split a token sequence into contiguous train / validation parts."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    split_index = int(len(tokens) * (1.0 - val_fraction))
    return list(tokens[:split_index]), list(tokens[split_index:])


class TokenDataset(Dataset):
    """Random-access next-token prediction windows."""

    def __init__(self, tokens: Sequence[int], block_size: int, stride: int | None = None) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        if len(self.tokens) <= block_size:
            raise ValueError("Not enough tokens for the requested block_size")
        self.block_size = block_size
        self.stride = stride or block_size
        self.starts = torch.arange(0, len(self.tokens) - block_size, self.stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        end = start + self.block_size
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]


class DataLoaderLite:
    """Minimal sequential loader that walks through a token stream once."""

    def __init__(
        self,
        tokens: Sequence[int],
        batch_size: int,
        block_size: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if batch_size < 1 or block_size < 1:
            raise ValueError("batch_size and block_size must be positive")

        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        self.tokens_per_epoch = (len(self.tokens) - 1) // (batch_size * block_size)
        if self.tokens_per_epoch < 1:
            raise ValueError("Not enough tokens for one batch")
        usable = self.tokens_per_epoch * batch_size * block_size + 1
        self.tokens = self.tokens[:usable]
        self.current_position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_tokens = self.batch_size * self.block_size
        if self.current_position + batch_tokens + 1 > len(self.tokens):
            self.current_position = 0

        start = self.current_position
        end = start + batch_tokens + 1
        buffer = self.tokens[start:end]
        x = buffer[:-1].view(self.batch_size, self.block_size)
        y = buffer[1:].view(self.batch_size, self.block_size)
        self.current_position += batch_tokens

        if self.current_position + batch_tokens + 1 > len(self.tokens):
            self.current_position = 0

        return x.to(self.device), y.to(self.device)

    def reset(self) -> None:
        self.current_position = 0

    def state_dict(self) -> dict[str, int]:
        return {"current_position": self.current_position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.current_position = int(state["current_position"])


class ShardDataset(IterableDataset):
    """Stream ``.npy`` token shards without loading them fully into memory.

    Each item is a pair of 1-D tensors ``(input_ids, target_ids)`` of length
    ``seq_len``. Use a normal PyTorch ``DataLoader`` to batch them.
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        seq_len: int = 1024,
        shuffle_files: bool = False,
        infinite: bool = False,
        seed: int = 0,
    ) -> None:
        if not shard_paths:
            raise ValueError("shard_paths cannot be empty")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.shard_paths = [Path(path) for path in shard_paths]
        self.seq_len = seq_len
        self.shuffle_files = shuffle_files
        self.infinite = infinite
        self.seed = seed

    def _iter_shard(self, path: Path) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        data = np.load(path, mmap_mode="r")
        last_start = len(data) - self.seq_len - 1
        if last_start < 0:
            return
        for start in range(0, last_start + 1, self.seq_len):
            chunk = np.asarray(data[start : start + self.seq_len + 1], dtype=np.int64)
            if chunk.shape[0] != self.seq_len + 1:
                break
            yield torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        paths = [path for path in self.shard_paths if path.exists()]
        if num_workers > 1:
            paths = paths[worker_id::num_workers]

        rng = np.random.default_rng(self.seed + worker_id)
        while True:
            iteration_paths = list(paths)
            if self.shuffle_files:
                rng.shuffle(iteration_paths)
            for path in iteration_paths:
                yield from self._iter_shard(path)
            if not self.infinite:
                break


def tokenize_documents_to_shards(
    documents: Iterable[str | dict[str, Any]],
    out_dir: str | Path,
    *,
    tokenizer: str = "gpt2",
    total_tokens: int = 5_000_000_000,
    shard_size: int = 100_000_000,
    val_shards: int = 1,
    add_eot: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Tokenize documents and write ``uint16`` ``.npy`` shards.

    The first ``val_shards`` shards are named ``val_*.npy``; the rest are
    ``train_*.npy``. A ``manifest.json`` describing the output is written
    alongside the shards.

    ``documents`` may contain strings or dictionaries with a ``"text"`` key,
    which makes it easy to pass a Hugging Face streaming dataset.
    """

    import tiktoken
    from tqdm import tqdm

    if shard_size < 1 or total_tokens < 1:
        raise ValueError("shard_size and total_tokens must be positive")

    encoding = tiktoken.get_encoding(tokenizer)
    if encoding.max_token_value >= 65_536:
        raise ValueError("uint16 shards require a tokenizer vocabulary below 65,536")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # uint16 is enough because the GPT-2 vocabulary is 50,257 tokens.
    # Each shard is a flat array of token ids; training windows are sliced later.
    buffer = np.empty(shard_size, dtype=np.uint16)
    buffered = 0
    shard_index = 0
    written_tokens = 0
    files: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffered, shard_index
        if buffered == 0:
            return
        # The first val_shards shards become validation; the rest are training.
        split = "val" if shard_index < val_shards else "train"
        path = out_path / f"{split}_{shard_index:03d}.npy"
        np.save(path, buffer[:buffered])
        files.append({"path": path.name, "split": split, "tokens": int(buffered)})
        shard_index += 1
        buffered = 0

    bar = (
        tqdm(total=total_tokens, unit="token", unit_scale=True, desc="Tokenizing")
        if progress
        else None
    )
    try:
        for document in documents:
            if written_tokens >= total_tokens:
                break
            text = document["text"] if isinstance(document, dict) else document
            ids = encoding.encode_ordinary(text)
            if add_eot:
                ids = [encoding.eot_token] + ids
            array = np.asarray(ids, dtype=np.uint16)

            remaining = total_tokens - written_tokens
            if array.size > remaining:
                array = array[:remaining]
            written_tokens += int(array.size)
            if bar is not None:
                bar.update(int(array.size))

            cursor = 0
            while cursor < array.size:
                space = shard_size - buffered
                take = min(space, array.size - cursor)
                buffer[buffered : buffered + take] = array[cursor : cursor + take]
                buffered += take
                cursor += take
                if buffered == shard_size:
                    flush()

            if written_tokens >= total_tokens:
                break
    finally:
        flush()
        if bar is not None:
            bar.close()

    manifest = {
        "tokenizer": tokenizer,
        "dtype": "uint16",
        "total_tokens": written_tokens,
        "shard_size": shard_size,
        "val_shards": val_shards,
        "add_eot": add_eot,
        "files": files,
    }
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def shard_paths(directory: str | Path, split: str | None = None) -> list[Path]:
    """Return sorted ``.npy`` shard paths, optionally filtered by split."""

    directory = Path(directory)
    pattern = f"{split}_*.npy" if split else "*.npy"
    return sorted(directory.glob(pattern))
