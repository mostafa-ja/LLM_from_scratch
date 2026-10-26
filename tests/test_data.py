from __future__ import annotations

import json

import numpy as np
import torch

from llm_from_scratch.data import (
    DataLoaderLite,
    ShardDataset,
    TokenDataset,
    split_tokens,
    tokenize_documents_to_shards,
)


def test_data_loader_lite_shapes() -> None:
    loader = DataLoaderLite(list(range(500)), batch_size=4, block_size=16)
    assert loader.tokens_per_epoch > 1

    for _ in range(loader.tokens_per_epoch + 2):
        x, y = loader.next_batch()
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        torch.testing.assert_close(x[:, 1:], y[:, :-1])

    loader.reset()
    assert loader.state_dict()["current_position"] == 0


def test_token_dataset() -> None:
    dataset = TokenDataset(list(range(200)), block_size=16)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    torch.testing.assert_close(x[1:], y[:-1])


def test_split_tokens() -> None:
    train, val = split_tokens(list(range(100)), val_fraction=0.2)
    assert train == list(range(80))
    assert val == list(range(80, 100))


def test_shard_dataset(tmp_path) -> None:
    first = tmp_path / "val_000.npy"
    second = tmp_path / "train_000.npy"
    np.save(first, np.arange(0, 50, dtype=np.uint16))
    np.save(second, np.arange(50, 100, dtype=np.uint16))

    dataset = ShardDataset([first, second], seq_len=8, infinite=False)
    samples = list(dataset)

    assert len(samples) > 0
    for x, y in samples:
        assert x.shape == (8,)
        assert y.shape == (8,)
        torch.testing.assert_close(x[1:], y[:-1])


def test_tokenize_documents_to_shards(tmp_path) -> None:
    documents = ["hello world " * 20, "second document " * 20, "third document " * 20]
    manifest = tokenize_documents_to_shards(
        documents,
        tmp_path,
        total_tokens=100,
        shard_size=32,
        val_shards=1,
        progress=False,
    )

    assert manifest["total_tokens"] == 100
    assert sum(item["tokens"] for item in manifest["files"]) == 100
    assert manifest["files"][0]["split"] == "val"
    assert any(item["split"] == "train" for item in manifest["files"])

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["total_tokens"] == 100

    arrays = [np.load(tmp_path / item["path"]) for item in manifest["files"]]
    assert all(array.dtype == np.uint16 for array in arrays)
    assert sum(len(array) for array in arrays) == 100
