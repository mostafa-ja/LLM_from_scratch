"""A byte-level Byte Pair Encoding (BPE) tokenizer built from scratch.

The implementation follows the GPT-2/GPT-4 recipe:

1. Split text with a regex that keeps whitespace and punctuation attached to
   the following word.
2. Convert each chunk to UTF-8 bytes.
3. Repeatedly merge the most frequent pair of adjacent token ids.
4. Store an ordered merge table so encoding can replay the merges.

It is intentionally compact and readable rather than optimised for
multi-gigabyte corpora. For production data pipelines this repository uses
``tiktoken`` (see ``data.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import regex as re

GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

Pair = tuple[int, int]


class BPETokenizer:
    """Trainable byte-level BPE tokenizer.

    The first 256 token ids are the individual bytes (0-255). Learned merges
    receive ids starting at 256, so ``vocab_size`` is at least 256.
    """

    def __init__(self, pattern: str = GPT2_SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self.merges: dict[Pair, int] = {}
        self.vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _get_stats(ids: Iterable[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
        counts = {} if counts is None else counts
        previous: int | None = None
        for token in ids:
            if previous is not None:
                pair = (previous, token)
                counts[pair] = counts.get(pair, 0) + 1
            previous = token
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(ids):
            if (
                index < len(ids) - 1
                and ids[index] == pair[0]
                and ids[index + 1] == pair[1]
            ):
                merged.append(new_id)
                index += 2
            else:
                merged.append(ids[index])
                index += 1
        return merged

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        ids = list(chunk)
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda candidate: self.merges.get(candidate, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Learn merges until the vocabulary reaches ``vocab_size``.

        If the corpus runs out of mergeable pairs, training stops early and the
        final vocabulary may be smaller than requested.
        """

        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        # Step 1: split text so a merge can never cross a word/punctuation boundary.
        chunks = re.findall(self.pattern, text)
        if not chunks:
            raise ValueError("Cannot train a tokenizer on empty text")

        # Step 2: start from raw UTF-8 bytes. Byte ids 0..255 are always valid,
        # so any Unicode string can be encoded without an <unk> token.
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks]
        merges: dict[Pair, int] = {}
        vocab = {index: bytes([index]) for index in range(256)}

        # Step 3: repeatedly merge the most frequent adjacent pair into a new id.
        for merge_index in range(vocab_size - 256):
            stats: dict[Pair, int] = {}
            for ids in chunk_ids:
                self._get_stats(ids, stats)
            if not stats:
                break

            # Deterministic tie-break: highest count, then smallest pair.
            pair = max(stats.items(), key=lambda item: (item[1], item[0]))[0]
            new_id = 256 + merge_index
            chunk_ids = [self._merge(ids, pair, new_id) for ids in chunk_ids]
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            merges[pair] = new_id

            if verbose:
                print(
                    f"merge {merge_index + 1}/{vocab_size - 256}: "
                    f"{pair} -> {new_id} ({vocab[new_id]!r}) "
                    f"occurrences={stats[pair]}"
                )

        self.merges = merges
        self.vocab = vocab
        return self

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of integer token ids."""

        ids: list[int] = []
        for chunk in re.findall(self.pattern, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids back into a string."""

        raw = b"".join(self.vocab[token] for token in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Persist pattern and merge table as JSON."""

        path = Path(path)
        payload: dict[str, Any] = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [[left, right, index] for (left, right), index in self.merges.items()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer produced by :meth:`save`."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(pattern=payload["pattern"])
        merges = {
            (int(left), int(right)): int(index)
            for left, right, index in payload["merges"]
        }
        tokenizer.merges = merges
        tokenizer.vocab = {index: bytes([index]) for index in range(256)}
        for (left, right), index in merges.items():
            tokenizer.vocab[index] = tokenizer.vocab[left] + tokenizer.vocab[right]
        return tokenizer

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
