"""Shared fixtures: a tiny byte-level tokenizer and fake language models."""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import pytest

from src.constrained_decoder import ConstrainedDecoder, DecodingConfig
from src.llm import LLMClient
from src.schema import FunctionDefinition, ParameterSpec
from src.tokenizer import BPETokenizer, bytes_to_unicode
from src.vocab import TokenTable

MERGES = [
    ("f", "n"), ("_", "a"), ("_a", "d"), ("_ad", "d"), ("{", '"'),
    ("n", "a"), ("m", "e"), ("na", "me"), ('"', ":"), ("Ġ", '"'),
    ("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), (",", "Ġ"),
    ("Ġ", "-"), ("Ã", "©"),
]


def make_tokenizer() -> BPETokenizer:
    """Build a small byte-level BPE tokenizer with a few merges."""
    symbols = list(bytes_to_unicode().values())
    encoder = {s: i for i, s in enumerate(symbols)}
    for left, right in MERGES:
        encoder.setdefault(left + right, len(encoder))
    special = {"<|im_start|>": len(encoder), "<|im_end|>": len(encoder) + 1}
    return BPETokenizer(
        encoder=encoder,
        merge_ranks={pair: rank for rank, pair in enumerate(MERGES)},
        special_tokens=special,
    )


class RandomSDK:
    """Returns pseudo-random logits that depend only on the input ids."""

    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        """Store the vocabulary size and the seed."""
        self.vocab_size = vocab_size
        self.seed = seed

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits seeded by a hash of ``input_ids``."""
        key = zlib.crc32(np.asarray(input_ids, dtype=np.int64).tobytes())
        rng = np.random.default_rng(key ^ self.seed)
        return [float(x) for x in rng.normal(size=self.vocab_size)]


class OracleSDK:
    """Pushes the generation towards ``target`` (a JSON document)."""

    def __init__(self, tokenizer: BPETokenizer, target: str) -> None:
        """Keep the tokenizer used to read back the generated text."""
        self.tokenizer = tokenizer
        self.target = target.encode("utf-8")
        self.token_bytes = [tokenizer.token_bytes(i) or b""
                            for i in range(tokenizer.vocab_size)]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Favour the longest token that continues the target (in bytes)."""
        data = b"".join(self.token_bytes[i] for i in input_ids)
        start = data.find(b'{"name": "')
        generated = data[start:] if start >= 0 else b""
        remaining = self.target[len(generated):] \
            if self.target.startswith(generated) else b""
        logits = np.zeros(self.tokenizer.vocab_size)
        for token_id, raw in enumerate(self.token_bytes):
            if raw and remaining.startswith(raw):
                logits[token_id] = 10.0 + len(raw)
        return [float(x) for x in logits]


class FailingSDK:
    """Simulates a model that crashes after ``ok_calls`` forward passes."""

    def __init__(self, vocab_size: int, ok_calls: int) -> None:
        """Remember how many calls succeed before failing."""
        self.inner = RandomSDK(vocab_size)
        self.ok_calls = ok_calls

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits, or raise once the budget is exhausted."""
        if self.ok_calls <= 0:
            raise RuntimeError("simulated device failure")
        self.ok_calls -= 1
        return self.inner.get_logits_from_input_ids(input_ids)


def make_decoder(sdk: Any, tokenizer: BPETokenizer,
                 **config: Any) -> ConstrainedDecoder:
    """Wrap a fake SDK into a decoder."""
    return ConstrainedDecoder(
        llm=LLMClient(sdk=sdk, retries=0), tokenizer=tokenizer,
        table=TokenTable.build(tokenizer), config=DecodingConfig(**config))


def matches_spec(value: Any, spec: ParameterSpec) -> bool:
    """Return whether ``value`` satisfies ``spec`` (test helper)."""
    if spec.enum is not None:
        return value in spec.enum
    if spec.type == "string":
        return isinstance(value, str)
    if spec.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if spec.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if spec.type == "boolean":
        return isinstance(value, bool)
    if spec.type == "null":
        return value is None
    if spec.type == "array":
        assert spec.items is not None
        return isinstance(value, list) and all(
            matches_spec(v, spec.items) for v in value)
    props = spec.properties or {}
    return isinstance(value, dict) and list(value) == list(props) and all(
        matches_spec(value[k], s) for k, s in props.items())


SAMPLE_FUNCTIONS = [
    FunctionDefinition.model_validate(entry) for entry in [
        {"name": "fn_add_numbers", "description": "Add two numbers.",
         "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
         "returns": {"type": "number"}},
        {"name": "fn_greet", "description": "Greet someone.",
         "parameters": {"name": {"type": "string"}}},
        {"name": "fn_flags", "description": "Flags and counts.",
         "parameters": {"verbose": {"type": "boolean"},
                        "count": {"type": "integer"},
                        "mode": {"type": "string",
                                 "enum": ["fast", "slow", "fast2"]}}},
        {"name": "fn_nested", "description": "Nested arguments.",
         "parameters": {
             "user": {"type": "object", "properties": {
                 "name": {"type": "string"},
                 "tags": {"type": "array", "items": {"type": "string"}}}},
             "points": {"type": "array", "items": {
                 "type": "object", "properties": {
                     "x": {"type": "number"}, "y": {"type": "number"}}}},
             "matrix": {"type": "array", "items": {
                 "type": "array", "items": {"type": "integer"}}},
             "nothing": {"type": "null"}}},
        {"name": "fn_noargs", "description": "No arguments."},
    ]
]


@pytest.fixture(scope="session")
def tokenizer() -> BPETokenizer:
    """Session-wide tiny tokenizer."""
    return make_tokenizer()
