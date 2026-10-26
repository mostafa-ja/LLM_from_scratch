from __future__ import annotations

from llm_from_scratch.tokenizer import BPETokenizer

TEXT = (
    "Hello world! This is a small corpus for training a byte-level BPE tokenizer. "
    "It includes punctuation, numbers like 12345, accents (café), and emoji 👋. "
    "안녕하세요. Unicode should always round-trip exactly because the base vocabulary "
    "contains every possible UTF-8 byte. "
) * 8


def test_train_and_roundtrip() -> None:
    tokenizer = BPETokenizer().train(TEXT, vocab_size=320)
    assert tokenizer.vocab_size > 256
    assert len(tokenizer.merges) > 0

    samples = [
        "Hello world!",
        "café 👋 안녕하세요",
        "",
        "12345 + 67890 = 80235",
        "    leading and trailing spaces    ",
    ]
    for sample in samples:
        ids = tokenizer.encode(sample)
        assert tokenizer.decode(ids) == sample


def test_compression() -> None:
    tokenizer = BPETokenizer().train(TEXT, vocab_size=320)
    raw_bytes = len(TEXT.encode("utf-8"))
    encoded = len(tokenizer.encode(TEXT))
    assert encoded < raw_bytes


def test_save_and_load(tmp_path) -> None:
    tokenizer = BPETokenizer().train(TEXT, vocab_size=300)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.pattern == tokenizer.pattern
    assert loaded.merges == tokenizer.merges
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode(TEXT) == tokenizer.encode(TEXT)


def test_rejects_tiny_vocab() -> None:
    try:
        BPETokenizer().train(TEXT, vocab_size=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for vocab_size < 256")
