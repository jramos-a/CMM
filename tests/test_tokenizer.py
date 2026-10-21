"""Tests for the hand-written BPE tokenizer."""

from __future__ import annotations

import os

import pytest

from src.tokenizer import BPETokenizer, bytes_to_unicode, pre_tokenize


@pytest.mark.parametrize("text, pieces", [
    ("What is the sum of 2 and 3?",
     ["What", " is", " the", " sum", " of", " ", "2", " and", " ", "3", "?"]),
    ("I'm 233 years", ["I", "'m", " ", "2", "3", "3", " years"]),
    ("they're WE'VE", ["they", "'re", " WE", "'VE"]),
    ('"hello"', ['"hello', '"']),
    ("a  b", ["a", " ", " b"]),
    ("x\n\n  y", ["x", "\n\n", " ", " y"]),
    ("end  ", ["end", "  "]),
    ("{\"a\": -2}", ["{\"", "a", "\":", " -", "2", "}"]),
    ("_add", ["_add"]),
    ("café 日本", ["café", " 日本"]),
    ("", []),
])
def test_pre_tokenize(text: str, pieces: list[str]) -> None:
    """The splitter reproduces the Qwen pre-tokenization regex."""
    assert pre_tokenize(text) == pieces


def test_pre_tokenize_is_lossless() -> None:
    """Joining the pieces always gives back the input."""
    text = "Mixed \t text\r\n with 123, symbols!? and   spaces \n"
    assert "".join(pre_tokenize(text)) == text


def test_bytes_to_unicode_is_a_bijection() -> None:
    """Every byte maps to a distinct printable character."""
    table = bytes_to_unicode()
    assert len(table) == 256
    assert len(set(table.values())) == 256


def test_encode_applies_merges(tokenizer: BPETokenizer) -> None:
    """Merges are applied in priority order."""
    ids = tokenizer.encode("hello")
    assert len(ids) == 1
    assert tokenizer.decode(ids) == "hello"


@pytest.mark.parametrize("text", [
    "plain text", "café été ☃ \U0001F600", "a\nb\tc",
    '{"name": "fn_add", "x": -2}', "",
])
def test_round_trip(tokenizer: BPETokenizer, text: str) -> None:
    """decode(encode(text)) == text."""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_atomic(tokenizer: BPETokenizer) -> None:
    """Added tokens written in the text become a single id."""
    ids = tokenizer.encode("<|im_start|>hi<|im_end|>")
    assert ids[0] == tokenizer.special_tokens["<|im_start|>"]
    assert ids[-1] == tokenizer.special_tokens["<|im_end|>"]
    assert tokenizer.decode(ids) == "hi"
    assert tokenizer.decode(ids, skip_special=False) == \
        "<|im_start|>hi<|im_end|>"


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1",
                    reason="set RUN_MODEL_TESTS=1 to load the real model")
def test_matches_reference_tokenizer() -> None:
    """Our encode/decode match the SDK's on the real Qwen vocabulary."""
    from src.llm import LLMClient

    llm = LLMClient.load("Qwen/Qwen3-0.6B")
    tok = llm.load_tokenizer()
    for text in [
        "What is the sum of 265 and 345?",
        "Replace all numbers in \"Hello 34 I'm 233 years old\" with X",
        "  spaces\n\n\ttabs\r\n  ", "café \U0001F600 日本語",
        "<|im_start|>user\nhi<|im_end|>\n<think>\n\n</think>",
    ]:
        reference = llm.sdk.encode(text)[0].tolist()
        assert tok.encode(text) == reference
        assert tok.decode(reference, skip_special=False) == text
